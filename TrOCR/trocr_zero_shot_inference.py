from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import psutil
import torch
from datasets import load_dataset
from jiwer import wer as jiwer_wer
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MALAYSIAN_CSV = DATASET_ROOT / "malaysian_dataset.csv"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"

DEFAULT_MODELS = [
    "microsoft/trocr-small-handwritten",
    "microsoft/trocr-base-handwritten",
    "microsoft/trocr-large-handwritten",
]
POLISH_PROBE = "\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b"


@dataclass
class ModelRunMetadata:
    model_id: str
    status: str
    error: str
    load_seconds: float | None
    rss_before_model_mb: float
    rss_after_model_mb: float | None
    rss_model_delta_mb: float | None
    polish_probe_roundtrip: str | None
    polish_probe_exact: bool | None


def normalize_text(value: str, form: str = "NFC") -> str:
    value = unicodedata.normalize(form, value or "")
    value = re.sub(r"\s+", " ", value.strip())
    return value


def compute_metrics(reference: str, prediction: str) -> dict[str, float | int | bool]:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    ref_words = ref.split()
    pred_words = pred.split()
    ref_chars = max(1, len(ref))
    ref_word_count = max(1, len(ref_words))
    cer = Levenshtein.distance(ref, pred) / ref_chars
    wer = Levenshtein.distance(ref_words, pred_words) / ref_word_count
    return {
        "cer": cer,
        "wer": wer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "word_accuracy": max(0.0, 1.0 - wer),
        "exact_match": ref == pred,
        "exact_match_casefold": ref.casefold() == pred.casefold(),
        "reference_chars": len(ref),
        "prediction_chars": len(pred),
        "reference_words": len(ref_words),
        "prediction_words": len(pred_words),
        "edit_distance_chars": Levenshtein.distance(ref, pred),
        "edit_distance_words": Levenshtein.distance(ref_words, pred_words),
        "jiwer_wer": jiwer_wer(ref, pred),
    }


def load_malaysian_manifest(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(MALAYSIAN_CSV)
    counts = df.groupby("image_path").size().to_dict()
    rows = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        rel_path = Path(row["image_path"])
        stem = rel_path.stem
        match = re.search(r"\((\d+)\)", stem)
        rows.append(
            {
                "dataset": "malaysian",
                "sample_id": f"{stem}_line_{row['text_id']}_row_{row_index + 1}",
                "image_key": str(DATASET_ROOT / rel_path),
                "image_path": str(DATASET_ROOT / rel_path),
                "relative_image_path": row["image_path"],
                "group": row["group"],
                "image_number": int(match.group(1)) if match else None,
                "reference_source": "csv_row",
                "text_id": int(row["text_id"]),
                "reference": row["text"],
                "csv_candidate_count": int(counts[row["image_path"]]),
            }
        )
    manifest = pd.DataFrame(rows)
    if limit:
        manifest = manifest.head(limit).copy()
    return manifest


def load_iam_manifest(limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Image.Image]]:
    split = "test" if not limit else f"test[:{limit}]"
    ds = load_dataset("Teklia/IAM-line", split=split)
    images: dict[str, Image.Image] = {}
    rows = []
    for idx, item in enumerate(ds):
        image_key = f"iam_test_{idx:05d}"
        images[image_key] = item["image"]
        rows.append(
            {
                "dataset": "iam",
                "sample_id": image_key,
                "image_key": image_key,
                "image_path": "",
                "relative_image_path": "",
                "group": "IAM-test",
                "image_number": idx,
                "reference_source": "Teklia/IAM-line test split",
                "text_id": None,
                "reference": item["text"],
                "csv_candidate_count": 1,
            }
        )
    return pd.DataFrame(rows), images


def prepare_image(image_source: str | Path | Image.Image, auto_invert: bool = True) -> Image.Image:
    if isinstance(image_source, Image.Image):
        im = image_source.convert("L")
    else:
        im = Image.open(image_source).convert("L")
    if auto_invert and im.resize((1, 1)).getpixel((0, 0)) < 128:
        im = ImageOps.invert(im)
    return im.convert("RGB")


def predict_unique_images(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    unique_sources: list[dict],
    iam_images: dict[str, Image.Image],
    batch_size: int,
    max_new_tokens: int,
    auto_invert: bool,
    log_every: int,
) -> dict[str, dict]:
    process = psutil.Process(os.getpid())
    predictions: dict[str, dict] = {}
    for start in range(0, len(unique_sources), batch_size):
        batch_sources = unique_sources[start : start + batch_size]
        images = []
        preprocess_seconds = []
        widths = []
        heights = []
        for sample in batch_sources:
            t0 = time.perf_counter()
            image_source = iam_images[sample["image_key"]] if sample["dataset"] == "iam" else sample["image_path"]
            image = prepare_image(image_source, auto_invert=auto_invert)
            preprocess_seconds.append(time.perf_counter() - t0)
            widths.append(image.width)
            heights.append(image.height)
            images.append(image)

        rss_before = process.memory_info().rss / (1024**2)
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        infer_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
        inference_seconds_batch = time.perf_counter() - infer_start
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        rss_after = process.memory_info().rss / (1024**2)

        per_image_seconds = inference_seconds_batch / max(1, len(batch_sources))
        for sample, pred, pre_s, width, height in zip(batch_sources, decoded, preprocess_seconds, widths, heights):
            predictions[sample["image_key"]] = {
                "prediction": normalize_text(pred),
                "preprocess_seconds": pre_s,
                "inference_seconds": per_image_seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "image_width": width,
                "image_height": height,
            }
        done = min(start + batch_size, len(unique_sources))
        if done % log_every == 0 or done == len(unique_sources):
            print(f"processed unique images {done}/{len(unique_sources)}", flush=True)
    return predictions


def run_model(model_id: str, manifest: pd.DataFrame, iam_images: dict[str, Image.Image], args: argparse.Namespace) -> tuple[pd.DataFrame, ModelRunMetadata]:
    process = psutil.Process(os.getpid())
    rss_before_model = process.memory_info().rss / (1024**2)
    load_start = time.perf_counter()
    try:
        processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
        torch.set_num_threads(max(1, args.threads))
        load_seconds = time.perf_counter() - load_start
        rss_after_model = process.memory_info().rss / (1024**2)
        try:
            polish_roundtrip = processor.tokenizer.decode(
                processor.tokenizer.encode(POLISH_PROBE),
                skip_special_tokens=True,
            )
        except Exception as exc:
            polish_roundtrip = f"TOKENIZER_ERROR: {type(exc).__name__}: {exc}"

        unique_sources = (
            manifest[["dataset", "image_key", "image_path"]]
            .drop_duplicates("image_key")
            .to_dict("records")
        )
        cache = predict_unique_images(
            model,
            processor,
            unique_sources,
            iam_images,
            args.batch_size,
            args.max_new_tokens,
            not args.no_auto_invert,
            args.log_every,
        )

        rows = []
        for sample in manifest.to_dict("records"):
            pred_info = cache[sample["image_key"]]
            metrics = compute_metrics(sample["reference"], pred_info["prediction"])
            rows.append(
                {
                    "model_id": model_id,
                    **sample,
                    **pred_info,
                    **metrics,
                }
            )
        status = "ok"
        error = ""
    except Exception as exc:
        load_seconds = None
        rss_after_model = None
        polish_roundtrip = None
        rows = []
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        print(f"MODEL_FAILED {model_id}: {error}", flush=True)
    finally:
        try:
            del model
            del processor
        except Exception:
            pass
        gc.collect()

    metadata = ModelRunMetadata(
        model_id=model_id,
        status=status,
        error=error,
        load_seconds=load_seconds,
        rss_before_model_mb=rss_before_model,
        rss_after_model_mb=rss_after_model,
        rss_model_delta_mb=(rss_after_model - rss_before_model) if rss_after_model is not None else None,
        polish_probe_roundtrip=polish_roundtrip,
        polish_probe_exact=(polish_roundtrip == POLISH_PROBE) if polish_roundtrip is not None else None,
    )
    return pd.DataFrame(rows), metadata


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["model_id", "dataset"])
        .agg(
            samples=("sample_id", "count"),
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            wer_median=("wer", "median"),
            char_accuracy_mean=("char_accuracy", "mean"),
            word_accuracy_mean=("word_accuracy", "mean"),
            exact_match_rate=("exact_match", "mean"),
            inference_seconds_total=("inference_seconds", "sum"),
            inference_seconds_mean=("inference_seconds", "mean"),
            inference_seconds_median=("inference_seconds", "median"),
            rss_after_mb_max=("rss_after_mb", "max"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--malaysian-limit", type=int, default=None)
    parser.add_argument("--iam-limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-auto-invert", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    malaysian = load_malaysian_manifest(args.malaysian_limit)
    iam, iam_images = load_iam_manifest(args.iam_limit)
    manifest = pd.concat([malaysian, iam], ignore_index=True)
    manifest.to_csv(MANIFEST_DIR / "trocr_eval_manifest.csv", index=False, encoding="utf-8")

    all_predictions = []
    all_metadata = []
    for model_id in args.models:
        print(f"=== {model_id} ===", flush=True)
        pred_df, metadata = run_model(model_id, manifest, iam_images, args)
        all_metadata.append(asdict(metadata))
        if not pred_df.empty:
            all_predictions.append(pred_df)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = summarize(predictions)
    predictions.to_csv(RESULTS_DIR / "trocr_zero_shot_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(RESULTS_DIR / "trocr_zero_shot_summary.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "trocr_zero_shot_metadata.json").write_text(
        json.dumps(
            {
                "models": args.models,
                "metadata": all_metadata,
                "malaysian_limit": args.malaysian_limit,
                "iam_limit": args.iam_limit,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "torch_version": torch.__version__,
                "python": sys.version,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(RESULTS_DIR / "trocr_zero_shot_metrics_raw.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame(all_metadata).to_excel(writer, sheet_name="Run metadata", index=False)
        manifest.to_excel(writer, sheet_name="Manifest", index=False)
    print(RESULTS_DIR / "trocr_zero_shot_metrics_raw.xlsx")


if __name__ == "__main__":
    main()
