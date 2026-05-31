from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import psutil
import torch
from datasets import load_dataset
from jiwer import wer as jiwer_wer
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein

from kraken.configs import RecognitionInferenceConfig
from kraken.containers import BBoxLine, Segmentation
from kraken.tasks import RecognitionTaskModel


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MALAYSIAN_CSV = DATASET_ROOT / "malaysian_dataset.csv"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MODEL_DOI = "10.5281/zenodo.13788177"
MODEL_FILENAME = "McCATMuS_nfd_nofix_V1.mlmodel"


@dataclass
class RunConfig:
    malaysian_limit: int | None
    iam_limit: int | None
    malaysian_label_mode: str
    model_path: str
    device: str
    auto_invert: bool
    normalize_form: str


def normalize_text(value: str, form: str = "NFC") -> str:
    value = unicodedata.normalize(form, value or "")
    value = re.sub(r"\s+", " ", value.strip())
    return value


def token_distance(ref_tokens: list[str], pred_tokens: list[str]) -> int:
    return Levenshtein.distance(ref_tokens, pred_tokens)


def compute_metrics(reference: str, prediction: str) -> dict[str, float | int | bool]:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    ref_chars = max(1, len(ref))
    ref_words = ref.split()
    pred_words = pred.split()
    word_count = max(1, len(ref_words))
    cer = Levenshtein.distance(ref, pred) / ref_chars
    wer = token_distance(ref_words, pred_words) / word_count
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
        "edit_distance_words": token_distance(ref_words, pred_words),
        "jiwer_wer": jiwer_wer(ref, pred),
    }


def locate_model() -> Path:
    local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    htrmopo_dir = local_appdata / "htrmopo" / "htrmopo"
    candidates = list(htrmopo_dir.rglob(MODEL_FILENAME)) if htrmopo_dir.exists() else []
    if candidates:
        return candidates[0]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(["kraken", "get", MODEL_DOI], check=True, env=env)
    candidates = list(htrmopo_dir.rglob(MODEL_FILENAME)) if htrmopo_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(f"Could not locate downloaded model {MODEL_FILENAME}")
    return candidates[0]


def load_malaysian_manifest(limit: int | None = None, label_mode: str = "csv_rows") -> pd.DataFrame:
    df = pd.read_csv(MALAYSIAN_CSV)
    if label_mode == "csv_rows":
        rows = []
        per_image_counts = df.groupby("image_path").size().to_dict()
        for row_index, row in df.reset_index(drop=True).iterrows():
            rel_path = Path(row["image_path"])
            stem = rel_path.stem
            match = re.search(r"\((\d+)\)", stem)
            image_number = int(match.group(1)) if match else None
            rows.append(
                {
                    "dataset": "malaysian",
                    "sample_id": f"{stem}_line_{row['text_id']}_row_{row_index + 1}",
                    "image_path": str(DATASET_ROOT / rel_path),
                    "relative_image_path": row["image_path"],
                    "group": row["group"],
                    "image_number": image_number,
                    "reference_source": "csv_row",
                    "text_id": row["text_id"],
                    "reference": row["text"],
                    "csv_candidate_text_ids": str(row["text_id"]),
                    "csv_candidate_count": int(per_image_counts[row["image_path"]]),
                }
            )
        manifest = pd.DataFrame(rows)
    elif label_mode == "inferred_unique":
        texts = (
            df[["text_id", "text"]]
            .drop_duplicates()
            .sort_values(["text_id", "text"])
            .drop_duplicates("text_id", keep="first")
            .set_index("text_id")["text"]
            .to_dict()
        )
        rows = []
        for image_path, group_df in df.groupby("image_path", sort=True):
            rel_path = Path(image_path)
            stem = rel_path.stem
            match = re.search(r"\((\d+)\)", stem)
            image_number = int(match.group(1)) if match else None
            inferred_text_id = ((image_number - 1) % 3) + 1 if image_number else None
            rows.append(
                {
                    "dataset": "malaysian",
                    "sample_id": stem,
                    "image_path": str(DATASET_ROOT / rel_path),
                    "relative_image_path": image_path,
                    "group": group_df["group"].iloc[0],
                    "image_number": image_number,
                    "reference_source": "filename_mod3_inferred",
                    "text_id": inferred_text_id,
                    "reference": texts.get(inferred_text_id, ""),
                    "csv_candidate_text_ids": "|".join(str(x) for x in sorted(group_df["text_id"].unique())),
                    "csv_candidate_count": int(len(group_df)),
                }
            )
        manifest = pd.DataFrame(rows).sort_values(["group", "image_number"]).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported Malaysian label mode: {label_mode}")
    if limit:
        manifest = manifest.head(limit).copy()
    return manifest


def load_iam_manifest(limit: int | None = None) -> pd.DataFrame:
    split = "test" if not limit else f"test[:{limit}]"
    ds = load_dataset("Teklia/IAM-line", split=split)
    rows = []
    for idx, item in enumerate(ds):
        rows.append(
            {
                "dataset": "iam",
                "sample_id": f"iam_test_{idx:05d}",
                "image": item["image"],
                "image_path": "",
                "relative_image_path": "",
                "group": "IAM-test",
                "image_number": idx,
                "reference_source": "Teklia/IAM-line test split",
                "text_id": None,
                "reference": item["text"],
                "csv_candidate_text_ids": "",
                "csv_candidate_count": 1,
            }
        )
    return pd.DataFrame(rows)


def prepare_image(image_or_path: str | Path | Image.Image, auto_invert: bool = True) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        im = image_or_path.convert("L")
    else:
        im = Image.open(image_or_path).convert("L")
    if auto_invert:
        # Malaysian images are white ink on black background. Kraken models expect
        # dark ink on light background, so invert only clearly dark-background lines.
        stat = ImageOps.grayscale(im).resize((1, 1)).getpixel((0, 0))
        if stat < 128:
            im = ImageOps.invert(im)
    return im


def predict_line(
    model: RecognitionTaskModel,
    image: Image.Image,
    config: RecognitionInferenceConfig,
) -> str:
    seg = Segmentation(
        type="bbox",
        text_direction=config.text_direction,
        imagename="memory",
        script_detection=False,
        lines=[BBoxLine(id=f"_{uuid.uuid4()}", bbox=(0, 0, image.width, image.height))],
    )
    records = list(model.predict(im=image, segmentation=seg, config=config))
    return records[0].prediction if records else ""


def iter_records(malaysian: pd.DataFrame, iam: pd.DataFrame) -> Iterable[dict]:
    for row in malaysian.to_dict("records"):
        yield row
    for row in iam.to_dict("records"):
        yield row


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    process = psutil.Process(os.getpid())
    rss_before_model = process.memory_info().rss / (1024**2)
    model_path = Path(args.model_path) if args.model_path else locate_model()

    load_start = time.perf_counter()
    model = RecognitionTaskModel.load_model(model_path)
    load_seconds = time.perf_counter() - load_start
    rss_after_model = process.memory_info().rss / (1024**2)
    config = RecognitionInferenceConfig()

    malaysian = load_malaysian_manifest(args.malaysian_limit, args.malaysian_label_mode)
    iam = load_iam_manifest(args.iam_limit)

    rows: list[dict] = []
    prediction_cache: dict[str, tuple[str, float, float, float, int, int]] = {}
    for idx, sample in enumerate(iter_records(malaysian, iam), start=1):
        image_source = sample.get("image") if sample["dataset"] == "iam" else sample["image_path"]
        cache_key = ""
        if sample["dataset"] == "malaysian":
            cache_key = f"{sample['image_path']}|auto_invert={not args.no_auto_invert}"

        if cache_key and cache_key in prediction_cache:
            prediction, preprocess_seconds, inference_seconds, rss_before, rss_after, image_width, image_height = prediction_cache[cache_key]
        else:
            preprocess_start = time.perf_counter()
            image = prepare_image(image_source, auto_invert=not args.no_auto_invert)
            preprocess_seconds = time.perf_counter() - preprocess_start
            rss_before = process.memory_info().rss / (1024**2)

            infer_start = time.perf_counter()
            prediction = predict_line(model, image, config)
            inference_seconds = time.perf_counter() - infer_start
            rss_after = process.memory_info().rss / (1024**2)
            image_width = image.width
            image_height = image.height
            if cache_key:
                prediction_cache[cache_key] = (
                    prediction,
                    preprocess_seconds,
                    inference_seconds,
                    rss_before,
                    rss_after,
                    image_width,
                    image_height,
                )

        metrics = compute_metrics(sample["reference"], prediction)
        rows.append(
            {
                **{k: v for k, v in sample.items() if k != "image"},
                "prediction": normalize_text(prediction),
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "image_width": image_width,
                "image_height": image_height,
                **metrics,
            }
        )
        if idx % args.log_every == 0:
            print(f"processed {idx} samples", flush=True)

    predictions = pd.DataFrame(rows)
    summary = (
        predictions.groupby("dataset")
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
    run_config = RunConfig(
        malaysian_limit=args.malaysian_limit,
        iam_limit=args.iam_limit,
        malaysian_label_mode=args.malaysian_label_mode,
        model_path=str(model_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        auto_invert=not args.no_auto_invert,
        normalize_form="NFC",
    )
    run_metadata = {
        "config": asdict(run_config),
        "model_doi": MODEL_DOI,
        "model_filename": MODEL_FILENAME,
        "model_load_seconds": load_seconds,
        "model_disk_size_mb": model_path.stat().st_size / (1024**2),
        "rss_before_model_mb": rss_before_model,
        "rss_after_model_mb": rss_after_model,
        "rss_model_delta_mb": rss_after_model - rss_before_model,
        "torch_version": torch.__version__,
        "python": sys.version,
    }

    predictions_path = RESULTS_DIR / "kraken_zero_shot_predictions.csv"
    summary_path = RESULTS_DIR / "kraken_zero_shot_summary.csv"
    metadata_path = RESULTS_DIR / "kraken_zero_shot_metadata.json"
    excel_path = RESULTS_DIR / "kraken_zero_shot_metrics_raw.xlsx"
    manifest_path = EXPERIMENT_ROOT / "manifests" / "malaysian_eval_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    malaysian.to_csv(manifest_path, index=False, encoding="utf-8")
    metadata_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([run_metadata]).to_excel(writer, sheet_name="Run metadata", index=False)
        malaysian.to_excel(writer, sheet_name="Malaysian manifest", index=False)

    print(f"predictions={predictions_path}")
    print(f"summary={summary_path}")
    print(f"metadata={metadata_path}")
    print(f"raw_excel={excel_path}")
    print(f"manifest={manifest_path}")
    return predictions, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malaysian-limit", type=int, default=None)
    parser.add_argument("--iam-limit", type=int, default=200)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-auto-invert", action="store_true")
    parser.add_argument(
        "--malaysian-label-mode",
        choices=["csv_rows", "inferred_unique"],
        default="csv_rows",
        help="csv_rows uses (image_path, text_id, text) exactly as the CSV defines it. inferred_unique is only a diagnostic one-label-per-image mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    run_experiment(parse_args())
