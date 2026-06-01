from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import compute_metrics, normalize_text, prepare_image, summarize_group_predictions, summarize_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-csv", default=str(PROJECT_ROOT / "FineTuning" / "data" / "trocr" / "test.csv"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "FineTuning" / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "FineTuning" / "TrOCR" / "evaluation"))
    parser.add_argument("--run-name", default="trocr_finetuned")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_start = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to(device=device, dtype=dtype).eval()
    load_seconds = time.perf_counter() - load_start
    manifest = pd.read_csv(args.test_csv)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    rows = []
    for start_idx in range(0, len(manifest), args.batch_size):
        batch = manifest.iloc[start_idx : start_idx + args.batch_size]
        images = []
        image_sizes = []
        for row in batch.to_dict("records"):
            image_path = Path(row["image_relpath"])
            if not image_path.is_absolute():
                image_path = Path(args.data_root) / image_path
            image = prepare_image(image_path, auto_invert=False, rgb=True)
            images.append(image)
            image_sizes.append((image.width, image.height))
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=args.max_new_tokens, num_beams=4)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - infer_start
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else None
        for row, pred, (width, height) in zip(batch.to_dict("records"), predictions, image_sizes):
            pred = normalize_text(pred)
            dataset = row.get("dataset", "polish_forms_test")
            source_dataset = row.get("source_dataset", "polish_forms")
            preprocessing_variant = row.get("preprocessing_variant", "native")
            group = row.get("difficulty_group", row.get("group", ""))
            rows.append(
                {
                    "model_id": args.run_name,
                    "sample_id": row["sample_id"],
                    "dataset": dataset,
                    "source_dataset": source_dataset,
                    "preprocessing_variant": preprocessing_variant,
                    "group": group,
                    "reference": row["text"],
                    "prediction": pred,
                    "inference_seconds": elapsed / max(1, len(batch)),
                    "cuda_peak_allocated_mb": peak_allocated,
                    "cuda_peak_reserved_mb": peak_reserved,
                    "image_width": width,
                    "image_height": height,
                    **compute_metrics(row["text"], pred),
                }
            )
        done = min(start_idx + len(batch), len(manifest))
        if done % 50 == 0 or done == len(manifest):
            print(f"processed {done}/{len(manifest)}", flush=True)

    predictions_df = pd.DataFrame(rows)
    summary = summarize_predictions(predictions_df)
    group_summary = summarize_group_predictions(predictions_df)
    stem = args.run_name
    predictions_df.to_csv(output_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(output_dir / f"{stem}_group_summary.csv", index=False, encoding="utf-8")
    (output_dir / f"{stem}_metadata.json").write_text(
        json.dumps({"model_path": args.model_path, "load_seconds": load_seconds, "device": str(device)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_dir / f"{stem}_summary.csv")


if __name__ == "__main__":
    main()
