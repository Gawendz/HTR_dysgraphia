# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from process_forms import DOUBLE_LINE_WRITERS, clean_line_crop, line_crop_box, line_ink_metrics, render_templates, save_rgb
from review_forms_app import ReviewStore, is_truthy, normalize_gt_status, normalize_scalar


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def refresh_crops(args: argparse.Namespace) -> tuple[int, int]:
    processed_dir = args.processed_dir.resolve()
    forms_path = processed_dir / "forms.csv"
    manifest_all_path = processed_dir / "manifest_all.csv"
    if not forms_path.exists():
        raise FileNotFoundError(forms_path)
    if not manifest_all_path.exists():
        raise FileNotFoundError(manifest_all_path)

    forms = pd.read_csv(forms_path, encoding="utf-8-sig")
    lines = pd.read_csv(manifest_all_path, encoding="utf-8-sig")
    templates = render_templates(args.template_pdf.resolve())

    aligned_paths = {
        str(row["writer_id"]): Path(str(row["aligned_path"]))
        for row in forms.to_dict("records")
        if normalize_scalar(row.get("writer_id")) and normalize_scalar(row.get("aligned_path"))
    }
    aligned_cache: dict[Path, np.ndarray] = {}

    refreshed = 0
    missing = 0
    allowed_writer_ids = {normalize_scalar(writer_id) for writer_id in args.writer_id}

    for idx, row in lines.iterrows():
        writer_id = normalize_scalar(row.get("writer_id"))
        if allowed_writer_ids and writer_id not in allowed_writer_ids:
            continue
        set_id = normalize_scalar(row.get("set_id"))
        line_id_raw = normalize_scalar(row.get("line_id"))
        if not writer_id or set_id not in templates or not line_id_raw:
            missing += 1
            continue

        aligned_path = aligned_paths.get(writer_id)
        if aligned_path is None or not aligned_path.exists():
            missing += 1
            continue

        try:
            line_index = int(float(line_id_raw)) - 1
        except ValueError:
            missing += 1
            continue
        if line_index < 0:
            missing += 1
            continue

        if aligned_path not in aligned_cache:
            aligned_cache[aligned_path] = load_rgb(aligned_path)
        aligned = aligned_cache[aligned_path]

        x0, y0, x1, y1 = line_crop_box(line_index, writer_id=writer_id)
        crop = aligned[y0:y1, x0:x1]
        template = templates[set_id].image_rgb
        template_crop = template[y0:y1, x0:x1]
        metrics = line_ink_metrics(crop, template_crop, args.line_ink_threshold)
        has_handwriting = (
            metrics["line_ink_ratio"] >= args.min_line_ink_ratio
            and metrics["line_ink_bbox_ratio"] >= args.min_line_ink_bbox_ratio
        )

        output_crop = (
            clean_line_crop(crop, template_crop, aggressive=writer_id in DOUBLE_LINE_WRITERS)
            if args.clean_lines or writer_id in DOUBLE_LINE_WRITERS
            else crop
        )
        flat_path = Path(str(row["flat_image_path"]))
        save_rgb(flat_path, output_crop)

        lines.at[idx, "crop_x0"] = x0
        lines.at[idx, "crop_y0"] = y0
        lines.at[idx, "crop_x1"] = x1
        lines.at[idx, "crop_y1"] = y1
        for key, value in metrics.items():
            lines.at[idx, key] = value
        lines.at[idx, "line_has_handwriting"] = bool(has_handwriting)
        lines.at[idx, "line_status"] = "filled" if has_handwriting else "empty_or_uncertain"

        gt_text = normalize_scalar(row.get("gt_text")) or normalize_scalar(row.get("reference"))
        raw_gt_status = normalize_scalar(row.get("gt_status")).lower()
        gt_status = normalize_gt_status(row.get("gt_status"))
        if gt_status in {"uncertain", "exclude"} or (raw_gt_status == "empty_line" and not gt_text):
            lines.at[idx, "line_exportable"] = False
        else:
            lines.at[idx, "line_exportable"] = bool(has_handwriting or args.include_empty_lines)

        refreshed += 1

    lines.to_csv(manifest_all_path, index=False, encoding="utf-8-sig")
    return refreshed, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh line crops in an existing processed forms directory.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--template-pdf", type=Path, required=True)
    parser.add_argument("--clean-lines", action="store_true")
    parser.add_argument("--include-empty-lines", action="store_true")
    parser.add_argument("--line-ink-threshold", type=int, default=185)
    parser.add_argument("--min-line-ink-ratio", type=float, default=0.003)
    parser.add_argument("--min-line-ink-bbox-ratio", type=float, default=0.01)
    parser.add_argument("--writer-id", action="append", default=[], help="Refresh only this writer_id. Can be used multiple times.")
    parser.add_argument("--no-rebuild-exports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refreshed, missing = refresh_crops(args)
    print(f"Refreshed line crops: {refreshed}")
    if missing:
        print(f"Skipped lines: {missing}")
    if not args.no_rebuild_exports:
        store = ReviewStore(args.processed_dir)
        result = store.rebuild_exports()
        print(result["message"])
        print(f"Exported lines: {result['exported_lines']}")


if __name__ == "__main__":
    main()
