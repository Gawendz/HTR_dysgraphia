from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import load_malaysian_manifest, prepare_image  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "FineTuning" / "malaysian_external_qwen"
PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--variants", nargs="+", default=["auto_invert", "raw"], choices=["auto_invert", "raw"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_malaysian_manifest(limit=args.limit, variants=args.variants)
    export_rows = []
    for row in manifest.to_dict("records"):
        variant = row["preprocessing_variant"]
        image_dir = output_root / "images" / variant
        image_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"{row['sample_id']}.png"
        rel_image = Path("images") / variant / image_name
        target = output_root / rel_image
        image = prepare_image(row["image_path"], auto_invert=bool(row["auto_invert"]), rgb=True)
        image.save(target)
        export_rows.append(
            {
                **row,
                "text": row["reference"],
                "export_image_path": str(target),
                "image_relpath": rel_image.as_posix(),
            }
        )
    export = pd.DataFrame(export_rows)
    export.to_csv(output_root / "manifest.csv", index=False, encoding="utf-8")

    qwen_dir = output_root / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        subset = export[export["preprocessing_variant"].eq(variant)].copy()
        write_qwen_json(subset, qwen_dir / f"malaysian_{variant}.json")
    write_qwen_json(export, qwen_dir / "malaysian_both_variants.json")
    summary = {
        "samples": int(len(export)),
        "variants": export["preprocessing_variant"].value_counts().sort_index().to_dict(),
        "groups": {
            f"{variant}/{group}": int(count)
            for (variant, group), count in export.groupby(["preprocessing_variant", "group"]).size().sort_index().items()
        },
        "prompt": PROMPT,
        "note": "Images are preprocessed and saved per variant, so evaluation should use auto_invert=False.",
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(output_root)


def write_qwen_json(df: pd.DataFrame, path: Path) -> None:
    records = []
    for row in df.to_dict("records"):
        records.append(
            {
                "id": row["sample_id"],
                "image": row["image_relpath"],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": row["reference"]},
                ],
                "metadata": {
                    "dataset": row["dataset"],
                    "source_dataset": row["source_dataset"],
                    "preprocessing_variant": row["preprocessing_variant"],
                    "difficulty_group": row["group"],
                    "group": row["group"],
                    "image_number": row.get("image_number"),
                    "text_id": row.get("text_id"),
                    "line_id": row.get("line_id"),
                    "reference_source": row.get("reference_source", ""),
                    "annotation_status": row.get("annotation_status", ""),
                },
            }
        )
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
