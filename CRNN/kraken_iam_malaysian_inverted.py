from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import psutil
import torch
from PIL import Image

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

from kraken.configs import RecognitionInferenceConfig  # noqa: E402
from kraken.containers import BBoxLine, Segmentation  # noqa: E402
from kraken.tasks import RecognitionTaskModel  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results" / "malaysian_inverted"
MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
MODEL_DOI = "10.5281/zenodo.13788177"
MODEL_FILENAME = "McCATMuS_nfd_nofix_V1.mlmodel"


@dataclass
class ModelMetadata:
    model_id: str
    status: str
    error: str
    load_seconds: float | None
    device: str
    rss_before_model_mb: float
    rss_after_model_mb: float | None
    rss_model_delta_mb: float | None
    model_path: str | None
    model_disk_size_mb: float | None


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


def predict_line(model: RecognitionTaskModel, image: Image.Image, config: RecognitionInferenceConfig) -> str:
    seg = Segmentation(
        type="bbox",
        text_direction=config.text_direction,
        imagename="memory",
        script_detection=False,
        lines=[BBoxLine(id=f"_{uuid.uuid4()}", bbox=(0, 0, image.width, image.height))],
    )
    records = list(model.predict(im=image, segmentation=seg, config=config))
    return records[0].prediction if records else ""


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    process = psutil.Process(os.getpid())
    model_path = Path(args.model_path) if args.model_path else locate_model()
    rss_before_model = process.memory_info().rss / (1024**2)
    load_start = time.perf_counter()
    model = RecognitionTaskModel.load_model(model_path)
    load_seconds = time.perf_counter() - load_start
    rss_after_model = process.memory_info().rss / (1024**2)
    config = RecognitionInferenceConfig()

    malaysian = load_malaysian_manifest(args.malaysian_limit, variants=args.malaysian_variants)
    iam, iam_images = load_iam_manifest(args.iam_limit)
    manifest = pd.concat([malaysian, iam], ignore_index=True)
    manifest.to_csv(MANIFEST_DIR / "kraken_iam_malaysian_inverted_manifest.csv", index=False, encoding="utf-8")

    rows: list[dict] = []
    prediction_cache: dict[str, dict] = {}
    for idx, sample in enumerate(manifest.to_dict("records"), start=1):
        cache_key = sample["image_key"]
        if cache_key in prediction_cache:
            pred_info = prediction_cache[cache_key]
        else:
            source = iam_images[sample["image_key"]] if sample["dataset"] == "iam" else sample["image_path"]
            preprocess_start = time.perf_counter()
            image = prepare_image(source, auto_invert=bool(sample["auto_invert"]), rgb=False)
            preprocess_seconds = time.perf_counter() - preprocess_start
            rss_before = process.memory_info().rss / (1024**2)
            infer_start = time.perf_counter()
            prediction = predict_line(model, image, config)
            inference_seconds = time.perf_counter() - infer_start
            rss_after = process.memory_info().rss / (1024**2)
            pred_info = {
                "prediction": normalize_text(prediction),
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "cuda_peak_allocated_mb": None,
                "cuda_peak_reserved_mb": None,
                "image_width": image.width,
                "image_height": image.height,
            }
            prediction_cache[cache_key] = pred_info

        rows.append({"model_id": "CRNN/Kraken", **sample, **pred_info, **compute_metrics(sample["reference"], pred_info["prediction"])})
        if idx % args.log_every == 0:
            print(f"processed {idx}/{len(manifest)} samples", flush=True)

    predictions = pd.DataFrame(rows)
    summary = summarize_predictions(predictions)
    group_summary = summarize_group_predictions(predictions)
    metadata = ModelMetadata(
        model_id="CRNN/Kraken",
        status="ok",
        error="",
        load_seconds=load_seconds,
        device="cuda" if torch.cuda.is_available() else "cpu",
        rss_before_model_mb=rss_before_model,
        rss_after_model_mb=rss_after_model,
        rss_model_delta_mb=rss_after_model - rss_before_model,
        model_path=str(model_path),
        model_disk_size_mb=model_path.stat().st_size / (1024**2),
    )

    predictions.to_csv(RESULTS_DIR / "kraken_iam_malaysian_inverted_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(RESULTS_DIR / "kraken_iam_malaysian_inverted_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(RESULTS_DIR / "kraken_iam_malaysian_inverted_group_summary.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "kraken_iam_malaysian_inverted_metadata.json").write_text(
        json.dumps(
            {
                "metadata": asdict(metadata),
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
    with pd.ExcelWriter(RESULTS_DIR / "kraken_iam_malaysian_inverted_metrics_raw.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        group_summary.to_excel(writer, sheet_name="Group Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame([asdict(metadata)]).to_excel(writer, sheet_name="Run metadata", index=False)
        manifest.to_excel(writer, sheet_name="Manifest", index=False)
    print(RESULTS_DIR / "kraken_iam_malaysian_inverted_metrics_raw.xlsx")
    return predictions, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--malaysian-limit", type=int, default=None)
    parser.add_argument("--iam-limit", type=int, default=200)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--malaysian-variants", nargs="+", default=["auto_invert", "raw"], choices=["auto_invert", "raw"])
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    run_experiment(parse_args())

