from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import load_malaysian_manifest, prepare_image  # noqa: E402


POLISH_DATA_ROOT = PROJECT_ROOT / "FineTuning" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "FineTuning" / "qwen_mixed_domain"
PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--malaysian-train-ratio", type=float, default=0.70)
    parser.add_argument("--malaysian-val-ratio", type=float, default=0.15)
    parser.add_argument("--variants", nargs="+", default=["raw", "auto_invert"], choices=["raw", "auto_invert"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    polish = load_polish_qwen_records(POLISH_DATA_ROOT / "qwen")
    copy_polish_images(output_root)
    malaysian = prepare_malaysian_records(output_root, args)
    split_summary = split_malaysian_by_image(malaysian, args)
    malaysian = malaysian.merge(split_summary, on=["source_group", "image_number"], how="left")

    qwen_dir = output_root / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)

    write_json(polish["train"] + records_for(malaysian, "train", args.variants), qwen_dir / "mixed_train.json")
    write_json(polish["val"] + records_for(malaysian, "val", args.variants), qwen_dir / "mixed_val.json")
    write_json(polish["test"], qwen_dir / "polish_test.json")
    for split in ["train", "val", "test"]:
        for variant in args.variants:
            write_json(records_for(malaysian, split, [variant]), qwen_dir / f"malaysian_{split}_{variant}.json")
        write_json(records_for(malaysian, split, args.variants), qwen_dir / f"malaysian_{split}_both_variants.json")

    malaysian.to_csv(output_root / "malaysian_manifest.csv", index=False, encoding="utf-8")
    summary = {
        "seed": args.seed,
        "malaysian_train_ratio": args.malaysian_train_ratio,
        "malaysian_val_ratio": args.malaysian_val_ratio,
        "variants": args.variants,
        "polish_counts": {name: len(records) for name, records in polish.items()},
        "malaysian_counts": {
            f"{split}/{variant}": int(len(records_for(malaysian, split, [variant])))
            for split in ["train", "val", "test"]
            for variant in args.variants
        },
        "mixed_train": len(polish["train"] + records_for(malaysian, "train", args.variants)),
        "mixed_val": len(polish["val"] + records_for(malaysian, "val", args.variants)),
        "note": "Malaysian split is by source_group + image_number. Both variants are used as image-domain augmentation for training.",
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_root)


def load_polish_qwen_records(qwen_dir: Path) -> dict[str, list[dict]]:
    return {
        "train": json.loads((qwen_dir / "train.json").read_text(encoding="utf-8")),
        "val": json.loads((qwen_dir / "val.json").read_text(encoding="utf-8")),
        "test": json.loads((qwen_dir / "test.json").read_text(encoding="utf-8")),
    }


def copy_polish_images(output_root: Path) -> None:
    source_dir = POLISH_DATA_ROOT / "images"
    target_dir = output_root / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.png"):
        target = target_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)


def prepare_malaysian_records(output_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    manifest = load_malaysian_manifest(variants=args.variants)
    rows = []
    for row in manifest.to_dict("records"):
        variant = row["preprocessing_variant"]
        image_dir = output_root / "images" / variant
        image_dir.mkdir(parents=True, exist_ok=True)
        image_name = f"{row['sample_id']}.png"
        rel_image = Path("images") / variant / image_name
        target = output_root / rel_image
        image = prepare_image(row["image_path"], auto_invert=bool(row["auto_invert"]), rgb=True)
        image.save(target)
        rows.append(
            {
                **row,
                "source_group": row["group"],
                "text": row["reference"],
                "export_image_path": str(target),
                "image_relpath": rel_image.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def split_malaysian_by_image(malaysian: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    unique = malaysian[["source_group", "image_number"]].drop_duplicates().copy()
    split_rows = []
    for group, group_df in unique.groupby("source_group", sort=True):
        image_numbers = sorted(group_df["image_number"].astype(int).tolist())
        rng = random.Random(f"{args.seed}:{group}")
        rng.shuffle(image_numbers)
        n = len(image_numbers)
        n_train = max(1, round(n * args.malaysian_train_ratio))
        n_val = max(1, round(n * args.malaysian_val_ratio))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_set = set(image_numbers[:n_train])
        val_set = set(image_numbers[n_train : n_train + n_val])
        for image_number in image_numbers:
            if image_number in train_set:
                split = "train"
            elif image_number in val_set:
                split = "val"
            else:
                split = "test"
            split_rows.append({"source_group": group, "image_number": image_number, "malaysian_split": split})
    return pd.DataFrame(split_rows)


def records_for(df: pd.DataFrame, split: str, variants: list[str]) -> list[dict]:
    subset = df[df["malaysian_split"].eq(split) & df["preprocessing_variant"].isin(variants)].copy()
    return [to_qwen_record(row) for row in subset.to_dict("records")]


def to_qwen_record(row: dict) -> dict:
    return {
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
            "malaysian_split": row["malaysian_split"],
            "image_number": row.get("image_number"),
            "text_id": row.get("text_id"),
            "line_id": row.get("line_id"),
            "reference_source": row.get("reference_source", ""),
        },
    }


def write_json(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
