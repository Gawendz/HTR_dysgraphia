from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import (  # noqa: E402
    compute_metrics,
    load_polish_forms_manifest,
    normalize_text,
    prepare_image,
    summarize_group_predictions,
    summarize_predictions,
)

from CRNN.kraken_iam_malaysian_inverted import locate_model, predict_line  # noqa: E402
from kraken.configs import RecognitionInferenceConfig  # noqa: E402
from kraken.tasks import RecognitionTaskModel  # noqa: E402
from Qwen3VL.qwen3vl_iam_malaysian_inverted import DEFAULT_MODELS as DEFAULT_QWEN_MODELS  # noqa: E402
from Qwen3VL.qwen3vl_iam_malaysian_inverted import run_model as run_qwen_model  # noqa: E402
from TrOCR.trocr_iam_malaysian_inverted import DEFAULT_MODELS as DEFAULT_TROCR_MODELS  # noqa: E402
from TrOCR.trocr_iam_malaysian_inverted import run_model as run_trocr_model  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def run_crnn(manifest: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    process = psutil.Process(os.getpid())
    torch.set_num_threads(max(1, args.threads))
    model_path = Path(args.kraken_model_path) if args.kraken_model_path else locate_model()
    rss_before_model = process.memory_info().rss / (1024**2)
    load_start = time.perf_counter()
    model = RecognitionTaskModel.load_model(model_path)
    load_seconds = time.perf_counter() - load_start
    rss_after_model = process.memory_info().rss / (1024**2)
    config = RecognitionInferenceConfig()

    rows: list[dict] = []
    try:
        for idx, sample in enumerate(manifest.to_dict("records"), start=1):
            preprocess_start = time.perf_counter()
            image = prepare_image(sample["image_path"], auto_invert=args.auto_invert, rgb=False)
            preprocess_seconds = time.perf_counter() - preprocess_start
            rss_before = process.memory_info().rss / (1024**2)
            infer_start = time.perf_counter()
            prediction = normalize_text(predict_line(model, image, config))
            inference_seconds = time.perf_counter() - infer_start
            rss_after = process.memory_info().rss / (1024**2)
            pred_info = {
                "prediction": prediction,
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
            rows.append({"model_id": "CRNN/Kraken", **sample, **pred_info, **compute_metrics(sample["reference"], prediction)})
            if idx % args.log_every == 0 or idx == len(manifest):
                print(f"CRNN/Kraken processed {idx}/{len(manifest)} samples", flush=True)
        status = "ok"
        error = ""
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        print(f"CRNN_FAILED: {error}", flush=True)
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()

    metadata = {
        "model_id": "CRNN/Kraken",
        "status": status,
        "error": error,
        "load_seconds": load_seconds,
        "device": "kraken_backend",
        "rss_before_model_mb": rss_before_model,
        "rss_after_model_mb": rss_after_model,
        "rss_model_delta_mb": rss_after_model - rss_before_model,
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
        "model_path": str(model_path),
        "model_disk_size_mb": model_path.stat().st_size / (1024**2),
    }
    return pd.DataFrame(rows), metadata


def save_result_set(result_dir: Path, stem: str, predictions: pd.DataFrame, metadata: list[dict], args: argparse.Namespace) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_predictions(predictions) if not predictions.empty else pd.DataFrame()
    group_summary = summarize_group_predictions(predictions) if not predictions.empty else pd.DataFrame()
    predictions.to_csv(result_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(result_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(result_dir / f"{stem}_group_summary.csv", index=False, encoding="utf-8")
    (result_dir / f"{stem}_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "polish_forms",
                "splits": args.splits,
                "limit": args.limit,
                "auto_invert": args.auto_invert,
                "metadata": metadata,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "python": sys.version,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pd.ExcelWriter(result_dir / f"{stem}_metrics_raw.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        group_summary.to_excel(writer, sheet_name="Group Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        pd.DataFrame(metadata).to_excel(writer, sheet_name="Run metadata", index=False)


def split_label(splits: list[str]) -> str:
    clean = [str(split) for split in splits]
    if len(clean) == 1:
        return clean[0]
    if set(clean) == {"train", "val", "test"} or "all" in set(clean):
        return "all"
    return "_".join(clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["test"], help="Dataset splits to evaluate: train val test or all.")
    parser.add_argument("--models", nargs="+", default=["crnn", "trocr", "qwen3vl"], choices=["crnn", "trocr", "qwen3vl"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--auto-invert", action="store_true", help="Use brightness-based polarity correction. Default keeps Polish forms native.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=96)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--trocr-models", nargs="+", default=DEFAULT_TROCR_MODELS)
    parser.add_argument("--qwen-models", nargs="+", default=DEFAULT_QWEN_MODELS)
    parser.add_argument("--kraken-model-path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    args = parse_args()
    selected_splits = ["train", "val", "test"] if "all" in set(args.splits) else args.splits
    manifest = load_polish_forms_manifest(splits=selected_splits, limit=args.limit)
    label = split_label(args.splits)
    result_root = EXPERIMENT_ROOT / "results" / label
    manifest_dir = EXPERIMENT_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / f"polish_forms_{label}_manifest.csv", index=False, encoding="utf-8")
    print(f"Loaded Polish Forms manifest: {len(manifest)} samples; splits={selected_splits}", flush=True)

    if "crnn" in args.models:
        predictions, metadata = run_crnn(manifest, args)
        save_result_set(result_root / "CRNN", f"crnn_polish_forms_{label}", predictions, [metadata], args)

    if "trocr" in args.models:
        all_predictions: list[pd.DataFrame] = []
        all_metadata: list[dict] = []
        trocr_args = SimpleNamespace(**vars(args))
        for model_id in args.trocr_models:
            print(f"=== TrOCR {model_id} ===", flush=True)
            pred_df, metadata = run_trocr_model(model_id, manifest, {}, trocr_args)
            all_metadata.append(asdict(metadata))
            if not pred_df.empty:
                all_predictions.append(pred_df)
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        save_result_set(result_root / "TrOCR", f"trocr_polish_forms_{label}", predictions, all_metadata, args)

    if "qwen3vl" in args.models:
        all_predictions = []
        all_metadata = []
        qwen_args = SimpleNamespace(**vars(args))
        qwen_args.max_new_tokens = args.qwen_max_new_tokens
        qwen_partial_dir = result_root / "Qwen3VL" / "partials"
        for model_id in args.qwen_models:
            print(f"=== Qwen3VL {model_id} ===", flush=True)
            safe_model_name = model_id.replace("/", "__").replace("\\", "__")
            qwen_args.partial_output_path = qwen_partial_dir / f"qwen3vl_polish_forms_{label}_{safe_model_name}_partial_predictions.csv"
            pred_df, metadata = run_qwen_model(model_id, manifest, {}, qwen_args)
            all_metadata.append(asdict(metadata))
            if not pred_df.empty:
                all_predictions.append(pred_df)
        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        save_result_set(result_root / "Qwen3VL", f"qwen3vl_polish_forms_{label}", predictions, all_metadata, args)

    print(result_root)


if __name__ == "__main__":
    main()
