from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import psutil
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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

DEFAULT_MODELS = [
    "microsoft/trocr-small-handwritten",
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-large-handwritten",
]
POLISH_PROBE = "\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b"


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
    polish_probe_roundtrip: str | None
    polish_probe_exact: bool | None


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {"float16": torch.float16, "float32": torch.float32}[name]


def predict_unique(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    manifest: pd.DataFrame,
    memory_images: dict[str, Image.Image],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_new_tokens: int,
    log_every: int,
) -> dict[str, dict]:
    process = psutil.Process(os.getpid())
    unique_sources = manifest[["dataset", "image_key", "image_path", "auto_invert"]].drop_duplicates("image_key").to_dict("records")
    predictions: dict[str, dict] = {}
    for start in range(0, len(unique_sources), batch_size):
        batch = unique_sources[start : start + batch_size]
        images = []
        preprocess_seconds = []
        widths = []
        heights = []
        for sample in batch:
            t0 = time.perf_counter()
            source = memory_images[sample["image_key"]] if sample["dataset"] == "iam" else sample["image_path"]
            image = prepare_image(source, auto_invert=bool(sample["auto_invert"]), rgb=True)
            preprocess_seconds.append(time.perf_counter() - t0)
            widths.append(image.width)
            heights.append(image.height)
            images.append(image)

        rss_before = process.memory_info().rss / (1024**2)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        infer_start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds_batch = time.perf_counter() - infer_start
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        rss_after = process.memory_info().rss / (1024**2)
        cuda_peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        cuda_peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None

        per_image_seconds = inference_seconds_batch / max(1, len(batch))
        for sample, pred, pre_s, width, height in zip(batch, decoded, preprocess_seconds, widths, heights):
            predictions[sample["image_key"]] = {
                "prediction": normalize_text(pred),
                "preprocess_seconds": pre_s,
                "inference_seconds": per_image_seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "cuda_peak_allocated_mb": cuda_peak_allocated,
                "cuda_peak_reserved_mb": cuda_peak_reserved,
                "image_width": width,
                "image_height": height,
            }
        done = min(start + batch_size, len(unique_sources))
        if done % log_every == 0 or done == len(unique_sources):
            print(f"processed unique images {done}/{len(unique_sources)}", flush=True)
    return predictions


def run_model(model_id: str, manifest: pd.DataFrame, memory_images: dict[str, Image.Image], args: argparse.Namespace) -> tuple[pd.DataFrame, ModelMetadata]:
    process = psutil.Process(os.getpid())
    requested_device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    dtype = resolve_dtype(args.dtype, device)
    rss_before_model = process.memory_info().rss / (1024**2)
    load_start = time.perf_counter()
    try:
        processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
        torch.set_num_threads(max(1, args.threads))
        model.to(device=device, dtype=dtype)
        load_seconds = time.perf_counter() - load_start
        rss_after_model = process.memory_info().rss / (1024**2)
        try:
            polish_roundtrip = processor.tokenizer.decode(processor.tokenizer.encode(POLISH_PROBE), skip_special_tokens=True)
        except Exception as exc:
            polish_roundtrip = f"TOKENIZER_ERROR: {type(exc).__name__}: {exc}"

        cache = predict_unique(
            model=model,
            processor=processor,
            manifest=manifest,
            memory_images=memory_images,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            log_every=args.log_every,
        )
        rows = []
        for sample in manifest.to_dict("records"):
            pred_info = cache[sample["image_key"]]
            rows.append({"model_id": model_id, **sample, **pred_info, **compute_metrics(sample["reference"], pred_info["prediction"])})
        status = "ok"
        error = ""
        cuda_peak_allocated = max((v["cuda_peak_allocated_mb"] or 0.0) for v in cache.values()) if device.type == "cuda" and cache else None
        cuda_peak_reserved = max((v["cuda_peak_reserved_mb"] or 0.0) for v in cache.values()) if device.type == "cuda" and cache else None
    except Exception as exc:
        rows = []
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        load_seconds = None
        rss_after_model = None
        polish_roundtrip = None
        cuda_peak_allocated = None
        cuda_peak_reserved = None
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
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        rss_before_model_mb=rss_before_model,
        rss_after_model_mb=rss_after_model,
        rss_model_delta_mb=(rss_after_model - rss_before_model) if rss_after_model is not None else None,
        cuda_peak_allocated_mb=cuda_peak_allocated,
        cuda_peak_reserved_mb=cuda_peak_reserved,
        polish_probe_roundtrip=polish_roundtrip,
        polish_probe_exact=(polish_roundtrip == POLISH_PROBE) if polish_roundtrip is not None else None,
    )
    return pd.DataFrame(rows), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--malaysian-limit", type=int, default=None)
    parser.add_argument("--iam-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--malaysian-variants", nargs="+", default=["auto_invert", "raw"], choices=["auto_invert", "raw"])
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    malaysian = load_malaysian_manifest(args.malaysian_limit, variants=args.malaysian_variants)
    iam, iam_images = load_iam_manifest(args.iam_limit)
    manifest = pd.concat([malaysian, iam], ignore_index=True)
    manifest.to_csv(MANIFEST_DIR / "trocr_iam_malaysian_inverted_manifest.csv", index=False, encoding="utf-8")

    all_predictions: list[pd.DataFrame] = []
    all_metadata: list[dict] = []
    for model_id in args.models:
        print(f"=== {model_id} ===", flush=True)
        pred_df, metadata = run_model(model_id, manifest, iam_images, args)
        all_metadata.append(asdict(metadata))
        if not pred_df.empty:
            all_predictions.append(pred_df)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = summarize_predictions(predictions) if not predictions.empty else pd.DataFrame()
    group_summary = summarize_group_predictions(predictions) if not predictions.empty else pd.DataFrame()
    predictions.to_csv(RESULTS_DIR / "trocr_iam_malaysian_inverted_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(RESULTS_DIR / "trocr_iam_malaysian_inverted_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(RESULTS_DIR / "trocr_iam_malaysian_inverted_group_summary.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "trocr_iam_malaysian_inverted_metadata.json").write_text(
        json.dumps(
            {
                "models": args.models,
                "metadata": all_metadata,
                "malaysian_limit": args.malaysian_limit,
                "iam_limit": args.iam_limit,
                "malaysian_variants": args.malaysian_variants,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "python": sys.version,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(RESULTS_DIR / "trocr_iam_malaysian_inverted_metrics_raw.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        group_summary.to_excel(writer, sheet_name="Group Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame(all_metadata).to_excel(writer, sheet_name="Run metadata", index=False)
        manifest.to_excel(writer, sheet_name="Manifest", index=False)
    print(RESULTS_DIR / "trocr_iam_malaysian_inverted_metrics_raw.xlsx")


if __name__ == "__main__":
    main()

