from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import psutil
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_benchmark_utils import (  # noqa: E402
    compute_metrics,
    load_iam_manifest,
    load_malaysian_manifest,
    normalize_text,
    prepare_image,
    summarize_group_predictions,
    summarize_predictions,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results" / "malaysian_inverted"
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
DEFAULT_MODELS = ["Qwen/Qwen3-VL-2B-Instruct"]
PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


@dataclass
class ModelMetadata:
    model_id: str
    status: str
    error: str
    load_seconds: float | None
    device: str
    dtype: str
    rss_before_model_mb: float
    rss_after_model_mb: float | None
    rss_model_delta_mb: float | None
    cuda_peak_allocated_mb: float | None
    cuda_peak_reserved_mb: float | None
    prompt: str
    max_pixels: int
    max_new_tokens: int


def clean_prediction(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(transcription|text)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    return normalize_text(text)


def resolve_dtype(name: str, device: str) -> torch.dtype | str:
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float16": torch.float16, "float32": torch.float32}[name]


def transcribe_one(model, processor, image: Image.Image, args: argparse.Namespace) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": args.max_pixels},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return clean_prediction(output_text[0] if output_text else "")


def run_model(model_id: str, manifest: pd.DataFrame, memory_images: dict[str, Image.Image], args: argparse.Namespace) -> tuple[pd.DataFrame, ModelMetadata]:
    process = psutil.Process(os.getpid())
    requested_device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    rss_before_model = process.memory_info().rss / (1024**2)
    load_start = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        ).eval()
        if requested_device == "cuda":
            model.to("cuda")
        else:
            model.to("cpu")
        load_seconds = time.perf_counter() - load_start
        rss_after_model = process.memory_info().rss / (1024**2)

        rows = []
        for idx, sample in enumerate(manifest.to_dict("records"), start=1):
            source = memory_images[sample["image_key"]] if sample["dataset"] == "iam" else sample["image_path"]
            preprocess_start = time.perf_counter()
            image = prepare_image(source, auto_invert=bool(sample["auto_invert"]), rgb=True)
            preprocess_seconds = time.perf_counter() - preprocess_start
            rss_before = process.memory_info().rss / (1024**2)
            if torch.cuda.is_available() and requested_device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            prediction = transcribe_one(model, processor, image, args)
            if torch.cuda.is_available() and requested_device == "cuda":
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - infer_start
            rss_after = process.memory_info().rss / (1024**2)
            cuda_peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if requested_device == "cuda" and torch.cuda.is_available() else None
            cuda_peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if requested_device == "cuda" and torch.cuda.is_available() else None
            pred_info = {
                "prediction": prediction,
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "cuda_peak_allocated_mb": cuda_peak_allocated,
                "cuda_peak_reserved_mb": cuda_peak_reserved,
                "image_width": image.width,
                "image_height": image.height,
            }
            rows.append({"model_id": model_id, **sample, **pred_info, **compute_metrics(sample["reference"], prediction)})
            if idx % args.log_every == 0 or idx == len(manifest):
                partial_output_path = getattr(args, "partial_output_path", None)
                if partial_output_path:
                    partial_output_path = Path(partial_output_path)
                    partial_output_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(partial_output_path, index=False, encoding="utf-8")
                print(f"processed {idx}/{len(manifest)} samples", flush=True)
        status = "ok"
        error = ""
        cuda_peak_allocated_model = max((row.get("cuda_peak_allocated_mb") or 0.0) for row in rows) if requested_device == "cuda" and rows else None
        cuda_peak_reserved_model = max((row.get("cuda_peak_reserved_mb") or 0.0) for row in rows) if requested_device == "cuda" and rows else None
    except Exception as exc:
        rows = []
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        load_seconds = None
        rss_after_model = None
        cuda_peak_allocated_model = None
        cuda_peak_reserved_model = None
        print(f"MODEL_FAILED {model_id}: {error}", flush=True)
    finally:
        try:
            del model
            del processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    metadata = ModelMetadata(
        model_id=model_id,
        status=status,
        error=error,
        load_seconds=load_seconds,
        device=requested_device,
        dtype=str(dtype).replace("torch.", ""),
        rss_before_model_mb=rss_before_model,
        rss_after_model_mb=rss_after_model,
        rss_model_delta_mb=(rss_after_model - rss_before_model) if rss_after_model is not None else None,
        cuda_peak_allocated_mb=cuda_peak_allocated_model,
        cuda_peak_reserved_mb=cuda_peak_reserved_model,
        prompt=PROMPT,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    return pd.DataFrame(rows), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--malaysian-limit", type=int, default=None)
    parser.add_argument("--iam-limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--malaysian-variants", nargs="+", default=["auto_invert", "raw"], choices=["auto_invert", "raw"])
    return parser.parse_args()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    malaysian = load_malaysian_manifest(args.malaysian_limit, variants=args.malaysian_variants)
    iam, iam_images = load_iam_manifest(args.iam_limit)
    manifest = pd.concat([malaysian, iam], ignore_index=True)
    manifest.to_csv(MANIFEST_DIR / "qwen3vl_iam_malaysian_inverted_manifest.csv", index=False, encoding="utf-8")

    predictions_all: list[pd.DataFrame] = []
    metadata_all: list[dict] = []
    for model_id in args.models:
        print(f"=== {model_id} ===", flush=True)
        pred_df, metadata = run_model(model_id, manifest, iam_images, args)
        metadata_all.append(asdict(metadata))
        if not pred_df.empty:
            predictions_all.append(pred_df)

    predictions = pd.concat(predictions_all, ignore_index=True) if predictions_all else pd.DataFrame()
    summary = summarize_predictions(predictions) if not predictions.empty else pd.DataFrame()
    group_summary = summarize_group_predictions(predictions) if not predictions.empty else pd.DataFrame()
    predictions.to_csv(RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_group_summary.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_metadata.json").write_text(
        json.dumps(
            {
                "models": args.models,
                "metadata": metadata_all,
                "malaysian_limit": args.malaysian_limit,
                "iam_limit": args.iam_limit,
                "malaysian_variants": args.malaysian_variants,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "python": sys.version,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_metrics_raw.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        group_summary.to_excel(writer, sheet_name="Group Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame(metadata_all).to_excel(writer, sheet_name="Run metadata", index=False)
        manifest.to_excel(writer, sheet_name="Manifest", index=False)
    print(RESULTS_DIR / "qwen3vl_iam_malaysian_inverted_metrics_raw.xlsx")


if __name__ == "__main__":
    main()

