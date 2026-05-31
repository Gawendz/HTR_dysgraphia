from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset
from jiwer import wer as jiwer_wer
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
MALAYSIAN_CSV = DATASET_ROOT / "malaysian_dataset.csv"
MALAYSIAN_MANUAL_LINES_DIR = DATASET_ROOT / "malaysian_manual_lines \u2014 kopia"
POLISH_FORMS_PROCESSED_DIR = PROJECT_ROOT / "Formularze" / "processed_320_check"
POLISH_FORMS_MANIFEST = POLISH_FORMS_PROCESSED_DIR / "manifest.csv"
IAM_DATASET_ID = "Teklia/IAM-line"


def normalize_text(value: str, form: str = "NFC") -> str:
    if value is None or pd.isna(value):
        value = ""
    value = str(value)
    value = unicodedata.normalize(form, value or "")
    return re.sub(r"\s+", " ", value.strip())


def levenshtein_operation_counts(source, target) -> dict[str, int]:
    substitutions = 0
    insertions = 0
    deletions = 0
    correct = 0
    for opcode in Levenshtein.opcodes(source, target):
        src_len = opcode.src_end - opcode.src_start
        dst_len = opcode.dest_end - opcode.dest_start
        if opcode.tag == "equal":
            correct += src_len
        elif opcode.tag == "replace":
            substitutions += min(src_len, dst_len)
            if dst_len > src_len:
                insertions += dst_len - src_len
            elif src_len > dst_len:
                deletions += src_len - dst_len
        elif opcode.tag == "insert":
            insertions += dst_len
        elif opcode.tag == "delete":
            deletions += src_len
    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "correct": correct,
    }


def compute_metrics(reference: str, prediction: str) -> dict[str, float | int | bool]:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    ref_words = ref.split()
    pred_words = pred.split()
    reference_chars_safe = max(1, len(ref))
    reference_words_safe = max(1, len(ref_words))
    edit_distance_chars = Levenshtein.distance(ref, pred)
    edit_distance_words = Levenshtein.distance(ref_words, pred_words)
    char_ops = levenshtein_operation_counts(ref, pred)
    word_ops = levenshtein_operation_counts(ref_words, pred_words)
    cer = edit_distance_chars / reference_chars_safe
    wer = edit_distance_words / reference_words_safe
    cla = (len(ref) - char_ops["substitutions"] - char_ops["insertions"]) / reference_chars_safe
    crw = word_ops["correct"] / reference_words_safe
    return {
        "cer": cer,
        "wer": wer,
        "cla": cla,
        "crw": crw,
        "char_accuracy": max(0.0, 1.0 - cer),
        "word_accuracy": max(0.0, 1.0 - wer),
        "exact_match": ref == pred,
        "exact_match_casefold": ref.casefold() == pred.casefold(),
        "reference_chars": len(ref),
        "prediction_chars": len(pred),
        "reference_words": len(ref_words),
        "prediction_words": len(pred_words),
        "edit_distance_chars": edit_distance_chars,
        "edit_distance_words": edit_distance_words,
        "char_substitutions": char_ops["substitutions"],
        "char_insertions": char_ops["insertions"],
        "char_deletions": char_ops["deletions"],
        "char_correct": char_ops["correct"],
        "word_substitutions": word_ops["substitutions"],
        "word_insertions": word_ops["insertions"],
        "word_deletions": word_ops["deletions"],
        "correct_recognized_words": word_ops["correct"],
        "jiwer_wer": jiwer_wer(ref, pred),
    }


def enrich_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    required_cols = [
        "cla",
        "crw",
        "char_substitutions",
        "char_insertions",
        "char_deletions",
        "char_correct",
        "word_substitutions",
        "word_insertions",
        "word_deletions",
        "correct_recognized_words",
    ]
    if all(col in out.columns for col in required_cols):
        return out
    if "reference" not in out.columns or "prediction" not in out.columns:
        for col in required_cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out
    metrics = [compute_metrics(row["reference"], row["prediction"]) for _, row in out.iterrows()]
    for col in required_cols:
        if col not in out.columns:
            out[col] = [metric[col] for metric in metrics]
    return out


def load_malaysian_manifest(
    limit: int | None = None,
    variants: Iterable[str] = ("auto_invert", "raw"),
    label_mode: str = "manual_lines",
    include_skipped: bool = False,
) -> pd.DataFrame:
    if label_mode == "manual_lines":
        return load_malaysian_manual_lines_manifest(limit, variants, include_skipped=include_skipped)

    df = pd.read_csv(MALAYSIAN_CSV)
    counts = df.groupby("image_path").size().to_dict()
    text_by_id = (
        df[["text_id", "text"]]
        .drop_duplicates()
        .sort_values(["text_id", "text"])
        .drop_duplicates("text_id", keep="first")
        .set_index("text_id")["text"]
        .to_dict()
    )
    rows: list[dict] = []
    for variant in variants:
        auto_invert = variant == "auto_invert"
        dataset_name = f"malaysian_{variant}"
        if label_mode == "csv_rows":
            source_rows = df.reset_index(drop=True).to_dict("records")
        elif label_mode == "filename_mod3":
            source_rows = []
            for image_path, group_df in df.groupby("image_path", sort=True):
                rel_path = Path(image_path)
                match = re.search(r"\((\d+)\)", rel_path.stem)
                image_number = int(match.group(1)) if match else None
                text_id = ((image_number - 1) % 3) + 1 if image_number else None
                source_rows.append(
                    {
                        "image_path": image_path,
                        "text": text_by_id.get(text_id, ""),
                        "group": group_df["group"].iloc[0],
                        "text_id": text_id,
                    }
                )
        else:
            raise ValueError(f"Unsupported Malaysian label_mode: {label_mode}")

        for row_index, row in enumerate(source_rows):
            rel_path = Path(row["image_path"])
            stem = rel_path.stem
            match = re.search(r"\((\d+)\)", stem)
            rows.append(
                {
                    "dataset": dataset_name,
                    "source_dataset": "malaysian",
                    "preprocessing_variant": variant,
                    "auto_invert": auto_invert,
                    "sample_id": f"{stem}_line_{row['text_id']}_row_{row_index + 1}_{variant}",
                    "image_key": f"{DATASET_ROOT / rel_path}|{variant}",
                    "image_path": str(DATASET_ROOT / rel_path),
                    "relative_image_path": row["image_path"],
                    "group": row["group"],
                    "image_number": int(match.group(1)) if match else None,
                    "reference_source": label_mode,
                    "text_id": int(row["text_id"]),
                    "reference": row["text"],
                    "csv_candidate_count": int(counts[row["image_path"]]),
                }
            )
    manifest = pd.DataFrame(rows)
    if limit:
        limited: list[pd.DataFrame] = []
        for _, part in manifest.groupby("preprocessing_variant", sort=False):
            limited.append(part.head(limit).copy())
        manifest = pd.concat(limited, ignore_index=True)
    return manifest


def load_malaysian_manual_lines_manifest(
    limit: int | None = None,
    variants: Iterable[str] = ("auto_invert", "raw"),
    include_skipped: bool = False,
) -> pd.DataFrame:
    annotation_path = MALAYSIAN_MANUAL_LINES_DIR / "annotations.csv"
    df = pd.read_csv(annotation_path)
    if not include_skipped:
        df = df[df["status"].eq("ok")].copy()
    pattern = re.compile(r"^(LPD|PD)_(\d+)_t(\d+)_l(\d+)\.png$")
    rows: list[dict] = []
    for variant in variants:
        auto_invert = variant == "auto_invert"
        dataset_name = f"malaysian_{variant}"
        for idx, row in df.reset_index(drop=True).iterrows():
            filename = str(row["filename"])
            match = pattern.match(filename)
            if not match:
                continue
            group, image_number, text_id, line_id = match.groups()
            image_path = MALAYSIAN_MANUAL_LINES_DIR / filename
            rows.append(
                {
                    "dataset": dataset_name,
                    "source_dataset": "malaysian",
                    "preprocessing_variant": variant,
                    "auto_invert": auto_invert,
                    "sample_id": f"{Path(filename).stem}_{variant}",
                    "image_key": f"{image_path}|{variant}",
                    "image_path": str(image_path),
                    "relative_image_path": str(Path(MALAYSIAN_MANUAL_LINES_DIR.name) / filename),
                    "group": group,
                    "image_number": int(image_number),
                    "reference_source": "manual_lines_annotations",
                    "text_id": int(text_id),
                    "line_id": int(line_id),
                    "reference": row["text"] if pd.notna(row["text"]) else "",
                    "csv_candidate_count": 1,
                    "annotation_status": row["status"],
                }
            )
    manifest = pd.DataFrame(rows)
    if limit:
        limited: list[pd.DataFrame] = []
        for _, part in manifest.groupby("preprocessing_variant", sort=False):
            limited.append(part.head(limit).copy())
        manifest = pd.concat(limited, ignore_index=True)
    return manifest


def load_iam_manifest(limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Image.Image]]:
    split = "test" if not limit else f"test[:{limit}]"
    ds = load_dataset(IAM_DATASET_ID, split=split)
    images: dict[str, Image.Image] = {}
    rows: list[dict] = []
    for idx, item in enumerate(ds):
        image_key = f"iam_test_{idx:05d}"
        images[image_key] = item["image"]
        rows.append(
            {
                "dataset": "iam",
                "source_dataset": "iam",
                "preprocessing_variant": "native",
                "auto_invert": False,
                "sample_id": image_key,
                "image_key": image_key,
                "image_path": "",
                "relative_image_path": "",
                "group": "IAM-test",
                "image_number": idx,
                "reference_source": f"{IAM_DATASET_ID} test split",
                "text_id": None,
                "reference": item["text"],
                "csv_candidate_count": 1,
            }
        )
    return pd.DataFrame(rows), images


def load_polish_forms_manifest(
    splits: Iterable[str] = ("test",),
    limit: int | None = None,
    manifest_path: str | Path = POLISH_FORMS_MANIFEST,
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    split_set = {str(split) for split in splits}
    if split_set and "all" not in split_set:
        df = df[df["split"].astype(str).isin(split_set)].copy()
    rows: list[dict] = []
    for _, row in df.reset_index(drop=True).iterrows():
        image_path = Path(str(row["image_path"]))
        reference = row["gt_text"] if "gt_text" in row and pd.notna(row["gt_text"]) else row["reference"]
        split = str(row["split"])
        sample_id = str(row["sample_id"])
        difficulty_group = str(row.get("difficulty_group", "unknown") or "unknown")
        rows.append(
            {
                "dataset": f"polish_forms_{split}",
                "source_dataset": "polish_forms",
                "preprocessing_variant": "native",
                "auto_invert": False,
                "sample_id": sample_id,
                "image_key": str(image_path),
                "image_path": str(image_path),
                "relative_image_path": str(image_path.relative_to(PROJECT_ROOT)) if image_path.is_absolute() and PROJECT_ROOT in image_path.parents else str(image_path),
                "group": difficulty_group,
                "writer_id": row.get("writer_id", ""),
                "set_id": row.get("set_id", ""),
                "line_id": row.get("line_id", ""),
                "reference_source": "polish_forms_manifest",
                "reference": reference if pd.notna(reference) else "",
                "gt_status": row.get("gt_status", ""),
                "split": split,
                "difficulty": row.get("difficulty", ""),
                "difficulty_group": difficulty_group,
                "sex": row.get("sex", ""),
                "birth_year": row.get("birth_year", ""),
            }
        )
    out = pd.DataFrame(rows)
    if limit:
        out = out.head(limit).copy()
    return out


def prepare_image(image_source: str | Path | Image.Image, auto_invert: bool = True, rgb: bool = True) -> Image.Image:
    if isinstance(image_source, Image.Image):
        image = image_source.convert("L")
    else:
        image = Image.open(image_source).convert("L")
    if auto_invert and image.resize((1, 1)).getpixel((0, 0)) < 128:
        image = ImageOps.invert(image)
    return image.convert("RGB") if rgb else image


def summarize_predictions(predictions: pd.DataFrame, model_column: str = "model_id") -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    predictions = enrich_prediction_metrics(predictions)
    for col in ["rss_after_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"]:
        if col not in predictions.columns:
            predictions[col] = pd.NA
    group_cols = [model_column, "dataset", "source_dataset", "preprocessing_variant"]
    if "group" in predictions.columns:
        for col in group_cols:
            if col not in predictions.columns:
                raise KeyError(col)
    grouped = predictions.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        samples=("sample_id", "count"),
        cer_mean=("cer", "mean"),
        cer_median=("cer", "median"),
        wer_mean=("wer", "mean"),
        wer_median=("wer", "median"),
        cla_mean=("cla", "mean"),
        crw_mean=("crw", "mean"),
        char_accuracy_mean=("char_accuracy", "mean"),
        word_accuracy_mean=("word_accuracy", "mean"),
        exact_match_rate=("exact_match", "mean"),
        reference_chars=("reference_chars", "sum"),
        edit_distance_chars=("edit_distance_chars", "sum"),
        reference_words=("reference_words", "sum"),
        edit_distance_words=("edit_distance_words", "sum"),
        char_substitutions=("char_substitutions", "sum"),
        char_insertions=("char_insertions", "sum"),
        char_deletions=("char_deletions", "sum"),
        char_correct=("char_correct", "sum"),
        word_substitutions=("word_substitutions", "sum"),
        word_insertions=("word_insertions", "sum"),
        word_deletions=("word_deletions", "sum"),
        correct_recognized_words=("correct_recognized_words", "sum"),
        inference_seconds_total=("inference_seconds", "sum"),
        inference_seconds_mean=("inference_seconds", "mean"),
        inference_seconds_median=("inference_seconds", "median"),
        rss_after_mb_max=("rss_after_mb", "max"),
        cuda_peak_allocated_mb_max=("cuda_peak_allocated_mb", "max"),
        cuda_peak_reserved_mb_max=("cuda_peak_reserved_mb", "max"),
    ).reset_index()
    summary["corpus_cer"] = summary["edit_distance_chars"] / summary["reference_chars"].clip(lower=1)
    summary["corpus_wer"] = summary["edit_distance_words"] / summary["reference_words"].clip(lower=1)
    summary["corpus_cla"] = (summary["reference_chars"] - summary["char_substitutions"] - summary["char_insertions"]) / summary["reference_chars"].clip(lower=1)
    summary["corpus_crw"] = summary["correct_recognized_words"] / summary["reference_words"].clip(lower=1)
    summary["corpus_char_accuracy"] = (1.0 - summary["corpus_cer"]).clip(lower=0.0)
    summary["corpus_word_accuracy"] = (1.0 - summary["corpus_wer"]).clip(lower=0.0)
    metric_cols = [
        "samples",
        "cer_mean",
        "cer_median",
        "corpus_cer",
        "wer_mean",
        "wer_median",
        "corpus_wer",
        "cla_mean",
        "corpus_cla",
        "crw_mean",
        "corpus_crw",
        "char_accuracy_mean",
        "corpus_char_accuracy",
        "word_accuracy_mean",
        "corpus_word_accuracy",
        "exact_match_rate",
        "reference_chars",
        "edit_distance_chars",
        "char_substitutions",
        "char_insertions",
        "char_deletions",
        "char_correct",
        "reference_words",
        "edit_distance_words",
        "word_substitutions",
        "word_insertions",
        "word_deletions",
        "correct_recognized_words",
        "inference_seconds_total",
        "inference_seconds_mean",
        "inference_seconds_median",
        "rss_after_mb_max",
        "cuda_peak_allocated_mb_max",
        "cuda_peak_reserved_mb_max",
    ]
    return summary[group_cols + metric_cols]


def summarize_group_predictions(predictions: pd.DataFrame, model_column: str = "model_id") -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    predictions = enrich_prediction_metrics(predictions)
    for col in ["cuda_peak_allocated_mb", "cuda_peak_reserved_mb"]:
        if col not in predictions.columns:
            predictions[col] = pd.NA
    group_cols = [model_column, "dataset", "source_dataset", "preprocessing_variant", "group"]
    summary = predictions.groupby(group_cols, dropna=False).agg(
        samples=("sample_id", "count"),
        cer_mean=("cer", "mean"),
        cla_mean=("cla", "mean"),
        corpus_edit_distance_chars=("edit_distance_chars", "sum"),
        corpus_reference_chars=("reference_chars", "sum"),
        corpus_char_substitutions=("char_substitutions", "sum"),
        corpus_char_insertions=("char_insertions", "sum"),
        wer_mean=("wer", "mean"),
        crw_mean=("crw", "mean"),
        corpus_edit_distance_words=("edit_distance_words", "sum"),
        corpus_reference_words=("reference_words", "sum"),
        corpus_correct_recognized_words=("correct_recognized_words", "sum"),
        inference_seconds_mean=("inference_seconds", "mean"),
        cuda_peak_allocated_mb_max=("cuda_peak_allocated_mb", "max"),
        cuda_peak_reserved_mb_max=("cuda_peak_reserved_mb", "max"),
    ).reset_index()
    summary["corpus_cer"] = summary["corpus_edit_distance_chars"] / summary["corpus_reference_chars"].clip(lower=1)
    summary["corpus_wer"] = summary["corpus_edit_distance_words"] / summary["corpus_reference_words"].clip(lower=1)
    summary["corpus_cla"] = (summary["corpus_reference_chars"] - summary["corpus_char_substitutions"] - summary["corpus_char_insertions"]) / summary["corpus_reference_chars"].clip(lower=1)
    summary["corpus_crw"] = summary["corpus_correct_recognized_words"] / summary["corpus_reference_words"].clip(lower=1)
    return summary
