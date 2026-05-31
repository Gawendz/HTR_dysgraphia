from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import compute_metrics, normalize_text, prepare_image, summarize_group_predictions, summarize_predictions  # noqa: E402


DEFAULT_PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-path", default="", help="LoRA adapter directory. Leave empty to evaluate base model.")
    parser.add_argument("--test-json", default=str(PROJECT_ROOT / "FineTuning" / "data" / "qwen" / "test.json"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "FineTuning" / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "FineTuning" / "Qwen3VL_LoRA" / "evaluation"))
    parser.add_argument("--run-name", default="qwen3vl_lora")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def clean_prediction(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(transcription|text)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    return normalize_text(text)


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_records(path: Path, limit: int | None) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[:limit] if limit else records


def main() -> None:
    args = parse_args()
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    requested_device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype, device_map=None, low_cpu_mem_usage=True).eval()
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).eval()
    model.to(requested_device)
    load_seconds = time.perf_counter() - load_start

    rows = []
    records = load_records(Path(args.test_json), args.limit)
    for idx, item in enumerate(records, start=1):
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = Path(args.data_root) / image_path
        image = prepare_image(image_path, auto_invert=False, rgb=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        if requested_device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        if requested_device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - infer_start
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        prediction = clean_prediction(processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        reference = item["conversations"][1]["value"]
        metadata = item.get("metadata", {})
        dataset = metadata.get("dataset", "polish_forms_test")
        source_dataset = metadata.get("source_dataset", "polish_forms")
        preprocessing_variant = metadata.get("preprocessing_variant", "native")
        rows.append(
            {
                "model_id": args.run_name,
                "sample_id": item["id"],
                "dataset": dataset,
                "source_dataset": source_dataset,
                "preprocessing_variant": preprocessing_variant,
                "group": metadata.get("difficulty_group", ""),
                "reference": reference,
                "prediction": prediction,
                "inference_seconds": elapsed,
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if requested_device == "cuda" else None,
                "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if requested_device == "cuda" else None,
                "image_width": image.width,
                "image_height": image.height,
                **compute_metrics(reference, prediction),
            }
        )
        if idx % 25 == 0 or idx == len(records):
            print(f"processed {idx}/{len(records)}", flush=True)

    predictions = pd.DataFrame(rows)
    summary = summarize_predictions(predictions)
    group_summary = summarize_group_predictions(predictions)
    stem = args.run_name
    predictions.to_csv(output_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(output_dir / f"{stem}_group_summary.csv", index=False, encoding="utf-8")
    (output_dir / f"{stem}_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "adapter_path": args.adapter_path,
                "load_seconds": load_seconds,
                "device": requested_device,
                "dtype": str(dtype).replace("torch.", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(output_dir / f"{stem}_summary.csv")


if __name__ == "__main__":
    main()
