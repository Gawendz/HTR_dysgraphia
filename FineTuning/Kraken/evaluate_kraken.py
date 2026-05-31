from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from kraken.configs import RecognitionInferenceConfig  # noqa: E402
from kraken.containers import BBoxLine, BaselineLine, Segmentation  # noqa: E402
from kraken.tasks import RecognitionTaskModel  # noqa: E402
from ocr_benchmark_utils import compute_metrics, normalize_text, prepare_image, summarize_group_predictions, summarize_predictions  # noqa: E402


def predict_line(model: RecognitionTaskModel, image: Image.Image, config: RecognitionInferenceConfig) -> str:
    if getattr(model, "seg_type", "") == "baselines":
        right = max(0, image.width - 1)
        bottom = max(0, image.height - 1)
        baseline_y = max(0, image.height // 2)
        line = BaselineLine(
            id="_line",
            baseline=[(0, baseline_y), (right, baseline_y)],
            boundary=[(0, 0), (right, 0), (right, bottom), (0, bottom)],
        )
        seg_type = "baselines"
    else:
        line = BBoxLine(id="_line", bbox=(0, 0, image.width, image.height))
        seg_type = "bbox"
    seg = Segmentation(
        type=seg_type,
        text_direction=config.text_direction,
        imagename="memory",
        script_detection=False,
        lines=[line],
    )
    records = list(model.predict(im=image, segmentation=seg, config=config))
    return records[0].prediction if records else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-csv", default=str(PROJECT_ROOT / "FineTuning" / "data" / "manifests" / "test.csv"))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "FineTuning" / "Kraken" / "evaluation"))
    parser.add_argument("--run-name", default="kraken_finetuned")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_image_path(row: dict, data_root: Path | None) -> Path:
    for key in ("export_image_path", "image_path", "image_relpath"):
        value = row.get(key)
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        path = Path(str(value))
        if not path.is_absolute() and data_root is not None:
            path = data_root / path
        return path
    raise KeyError("Expected one of: export_image_path, image_path, image_relpath")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.test_csv)
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    data_root = Path(args.data_root) if args.data_root else None
    model_path = Path(args.model_path)
    load_start = time.perf_counter()
    model = RecognitionTaskModel.load_model(model_path)
    model.eval()
    load_seconds = time.perf_counter() - load_start
    config = RecognitionInferenceConfig()
    process = psutil.Process()

    rows = []
    for idx, row in enumerate(manifest.to_dict("records"), start=1):
        image_path = resolve_image_path(row, data_root)
        image = prepare_image(image_path, auto_invert=False, rgb=False)
        rss_before = process.memory_info().rss / (1024**2)
        start = time.perf_counter()
        prediction = normalize_text(predict_line(model, image, config))
        seconds = time.perf_counter() - start
        rss_after = process.memory_info().rss / (1024**2)
        dataset = row.get("dataset", "polish_forms_test")
        source_dataset = row.get("source_dataset", "polish_forms")
        preprocessing_variant = row.get("preprocessing_variant", "native")
        group = row.get("difficulty_group", row.get("group", ""))
        reference = row.get("text", row.get("reference", ""))
        rows.append(
            {
                "model_id": args.run_name,
                "sample_id": row["sample_id"],
                "dataset": dataset,
                "source_dataset": source_dataset,
                "preprocessing_variant": preprocessing_variant,
                "group": group,
                "reference": reference,
                "prediction": prediction,
                "inference_seconds": seconds,
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "rss_delta_mb": rss_after - rss_before,
                "image_width": image.width,
                "image_height": image.height,
                **compute_metrics(reference, prediction),
            }
        )
        if idx % 50 == 0 or idx == len(manifest):
            print(f"processed {idx}/{len(manifest)}", flush=True)

    predictions = pd.DataFrame(rows)
    summary = summarize_predictions(predictions)
    group_summary = summarize_group_predictions(predictions)
    stem = args.run_name
    predictions.to_csv(output_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    group_summary.to_csv(output_dir / f"{stem}_group_summary.csv", index=False, encoding="utf-8")
    (output_dir / f"{stem}_metadata.json").write_text(
        json.dumps({"model_path": str(model_path), "load_seconds": load_seconds, "test_csv": args.test_csv}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_dir / f"{stem}_summary.csv")


if __name__ == "__main__":
    main()
