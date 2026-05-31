from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "Formularze" / "processed_320_check" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "FineTuning" / "data"
PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."
KEEP_COLUMNS = [
    "sample_id",
    "writer_id",
    "set_id",
    "line_id",
    "gt_text",
    "reference",
    "gt_status",
    "image_path",
    "sex",
    "difficulty",
    "difficulty_group",
    "birth_year",
    "split",
]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.source_manifest))
    export = build_export_manifest(manifest, output_root, copy_images=not args.no_copy_images)
    train_balanced = oversample_dysgraphia(export[export["split"].eq("train")].copy(), args.dysgraphia_target_ratio, args.seed)

    write_common_manifests(export, train_balanced, output_root)
    write_qwen_data(export, train_balanced, output_root)
    write_trocr_data(export, train_balanced, output_root)
    write_kraken_data(export, train_balanced, output_root)
    write_summary(export, train_balanced, output_root, args)
    print(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dysgraphia-target-ratio", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy-images", action="store_true", help="Keep only manifests. Training then uses original image paths.")
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    df = df[KEEP_COLUMNS].copy()
    df["gt_text"] = df["gt_text"].fillna(df["reference"]).astype(str).map(clean_text)
    df["reference"] = df["reference"].fillna(df["gt_text"]).astype(str).map(clean_text)
    df["image_path"] = df["image_path"].astype(str)
    df = df[df["gt_text"].str.len().gt(0)].copy()
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    df = df.sort_values(["split", "writer_id", "set_id", "line_id", "sample_id"]).reset_index(drop=True)
    missing_images = [p for p in df["image_path"].map(Path) if not p.exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images. First missing: {missing_images[0]}")
    return df


def clean_text(value: str) -> str:
    return " ".join(str(value).replace("\ufeff", "").split()).strip()


def build_export_manifest(df: pd.DataFrame, output_root: Path, copy_images: bool) -> pd.DataFrame:
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for record in df.to_dict("records"):
        source = Path(record["image_path"])
        image_name = f"{record['sample_id']}{source.suffix.lower()}"
        rel_image = Path("images") / image_name
        target = output_root / rel_image
        if copy_images and (not target.exists() or target.stat().st_size != source.stat().st_size):
            shutil.copy2(source, target)
        image_path = target if copy_images else source
        rows.append(
            {
                **record,
                "text": record["gt_text"],
                "source_image_path": str(source),
                "export_image_path": str(image_path),
                "image_relpath": rel_image.as_posix() if copy_images else str(source),
                "sample_id_original": record["sample_id"],
                "duplicate_index": 0,
                "training_variant": "standard",
            }
        )
    return pd.DataFrame(rows)


def oversample_dysgraphia(train: pd.DataFrame, target_ratio: float, seed: int) -> pd.DataFrame:
    if not 0 < target_ratio < 1:
        raise ValueError("--dysgraphia-target-ratio must be between 0 and 1")
    dys = train[train["difficulty_group"].eq("dysgrafia")].copy()
    non_dys = train[~train["difficulty_group"].eq("dysgrafia")].copy()
    if dys.empty or non_dys.empty:
        return train
    target_dys_count = math.ceil(target_ratio * len(non_dys) / (1 - target_ratio))
    extra_count = max(0, target_dys_count - len(dys))
    if extra_count == 0:
        out = train.copy()
        out["training_variant"] = "dysgraphia_oversampled"
        return out
    extra = dys.sample(n=extra_count, replace=True, random_state=seed).copy()
    duplicate_counts: dict[str, int] = {}
    duplicate_ids = []
    for sample_id in extra["sample_id_original"].tolist():
        duplicate_counts[sample_id] = duplicate_counts.get(sample_id, 0) + 1
        duplicate_ids.append(duplicate_counts[sample_id])
    extra["duplicate_index"] = duplicate_ids
    extra["sample_id"] = extra.apply(lambda row: f"{row['sample_id_original']}__dup{int(row['duplicate_index']):03d}", axis=1)
    out = pd.concat([train, extra], ignore_index=True)
    out["training_variant"] = "dysgraphia_oversampled"
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_common_manifests(export: pd.DataFrame, train_balanced: pd.DataFrame, output_root: Path) -> None:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        export[export["split"].eq(split)].to_csv(manifest_dir / f"{split}.csv", index=False, encoding="utf-8")
    train_balanced.to_csv(manifest_dir / "train_dysgraphia_oversampled.csv", index=False, encoding="utf-8")
    export.to_csv(manifest_dir / "all.csv", index=False, encoding="utf-8")


def write_qwen_data(export: pd.DataFrame, train_balanced: pd.DataFrame, output_root: Path) -> None:
    qwen_dir = output_root / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        write_qwen_json(export[export["split"].eq(split)], qwen_dir / f"{split}.json")
    write_qwen_json(train_balanced, qwen_dir / "train_dysgraphia_oversampled.json")
    registry = {
        "polish_forms_train": {"annotation_path": "qwen/train.json", "data_path": "."},
        "polish_forms_val": {"annotation_path": "qwen/val.json", "data_path": "."},
        "polish_forms_test": {"annotation_path": "qwen/test.json", "data_path": "."},
        "polish_forms_train_dysgraphia_oversampled": {
            "annotation_path": "qwen/train_dysgraphia_oversampled.json",
            "data_path": ".",
        },
    }
    (qwen_dir / "dataset_info_snippet.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def write_qwen_json(df: pd.DataFrame, path: Path) -> None:
    records = []
    for row in df.to_dict("records"):
        records.append(
            {
                "id": row["sample_id"],
                "image": row["image_relpath"],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{PROMPT}"},
                    {"from": "gpt", "value": row["text"]},
                ],
                "metadata": {
                    "writer_id": row["writer_id"],
                    "split": row["split"],
                    "difficulty_group": row["difficulty_group"],
                    "sample_id_original": row["sample_id_original"],
                    "duplicate_index": int(row["duplicate_index"]),
                },
            }
        )
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trocr_data(export: pd.DataFrame, train_balanced: pd.DataFrame, output_root: Path) -> None:
    trocr_dir = output_root / "trocr"
    trocr_dir.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "image_relpath", "export_image_path", "text", "difficulty_group", "writer_id", "split", "sample_id_original", "duplicate_index"]
    for split in ["train", "val", "test"]:
        export[export["split"].eq(split)][cols].to_csv(trocr_dir / f"{split}.csv", index=False, encoding="utf-8")
    train_balanced[cols].to_csv(trocr_dir / "train_dysgraphia_oversampled.csv", index=False, encoding="utf-8")


def write_kraken_data(export: pd.DataFrame, train_balanced: pd.DataFrame, output_root: Path) -> None:
    kraken_dir = output_root / "kraken"
    for split in ["train", "val", "test"]:
        write_kraken_split(export[export["split"].eq(split)], kraken_dir / split)
    write_kraken_split(train_balanced, kraken_dir / "train_dysgraphia_oversampled")


def write_kraken_split(df: pd.DataFrame, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for row in df.to_dict("records"):
        source = Path(row["export_image_path"])
        suffix = source.suffix.lower()
        name = row["sample_id"]
        image_target = split_dir / f"{name}{suffix}"
        text_target = split_dir / f"{name}.gt.txt"
        if not image_target.exists() or image_target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, image_target)
        text_target.write_text(row["text"], encoding="utf-8")
        manifest_lines.append(str(image_target))
    (split_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def write_summary(export: pd.DataFrame, train_balanced: pd.DataFrame, output_root: Path, args: argparse.Namespace) -> None:
    summary = {
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "output_root": str(output_root),
        "prompt": PROMPT,
        "dysgraphia_target_ratio": args.dysgraphia_target_ratio,
        "seed": args.seed,
        "counts_by_split_group": nested_counts(export, ["split", "difficulty_group"]),
        "balanced_train_counts_by_group": train_balanced["difficulty_group"].value_counts().sort_index().to_dict(),
        "balanced_train_rows": int(len(train_balanced)),
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def nested_counts(df: pd.DataFrame, cols: list[str]) -> dict:
    return {str(key): int(value) for key, value in df.groupby(cols).size().sort_index().items()}


if __name__ == "__main__":
    main()
