# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = (
    "Read the handwritten birth year from this image. "
    "Return only compact JSON with keys year and confidence, for example "
    '{"year":"2010","confidence":0.93}. '
    'If you are uncertain, return {"year":"","confidence":0.0}.'
)


def resolve_dtype(name: str, device: str) -> torch.dtype | str:
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {"float16": torch.float16, "float32": torch.float32}[name]


def normalize_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def is_truthy(value: Any) -> bool:
    return normalize_scalar(value).lower() in {"true", "1", "yes", "tak"}


def slug(value: object, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = (
        text.replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ź", "z")
        .replace("ż", "z")
    )
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or default


def prepare_crop(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Small year crops are easier for VLMs after padding and moderate upscaling.
    padded = Image.new("RGB", (image.width + 80, image.height + 80), "white")
    padded.paste(image, (40, 40))
    scale = max(1, min(6, 640 // max(1, padded.width)))
    if scale > 1:
        padded = padded.resize((padded.width * scale, padded.height * scale), Image.Resampling.LANCZOS)
    return padded


def clean_model_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_year_prediction(raw_text: str, min_year: int, max_year: int) -> tuple[str, float, str]:
    text = clean_model_text(raw_text)
    json_confidence: float | None = None
    json_year = ""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            json_year = normalize_scalar(parsed.get("year"))
            try:
                json_confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                json_confidence = None
    except json.JSONDecodeError:
        pass

    candidates = []
    if json_year:
        candidates.extend(re.findall(r"(?<!\d)(?:19\d{2}|20\d{2})(?!\d)", json_year))
    candidates.extend(re.findall(r"(?<!\d)(?:19\d{2}|20\d{2})(?!\d)", text))

    plausible = []
    for candidate in candidates:
        year = int(candidate)
        if min_year <= year <= max_year:
            plausible.append(candidate)

    unique = list(dict.fromkeys(plausible))
    if not unique:
        return "", float(json_confidence or 0.0), "needs_review"

    year = unique[0]
    if json_confidence is not None:
        confidence = max(0.0, min(1.0, json_confidence))
    else:
        confidence = 0.85 if len(unique) == 1 else 0.55

    status = "auto_qwen" if len(unique) == 1 else "qwen_needs_review"
    return year, confidence, status


def transcribe_year(model, processor, image: Image.Image, args: argparse.Namespace) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": args.max_pixels},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0].strip() if output_text else ""


def load_qwen(args: argparse.Namespace):
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True,
    ).eval()
    if device == "cpu":
        model.to("cpu")
    return model, processor, device, str(dtype).replace("torch.", "")


def predict_birth_years(forms: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    model = None
    processor = None
    device = ""
    dtype = ""
    try:
        model, processor, device, dtype = load_qwen(args)
        for idx, row in enumerate(forms.to_dict("records"), start=1):
            writer_id = normalize_scalar(row.get("writer_id"))
            existing_year = normalize_scalar(row.get("birth_year"))
            crop_path = Path(normalize_scalar(row.get("birth_year_crop_path")))

            if existing_year and not args.force:
                rows.append(
                    {
                        "writer_id": writer_id,
                        "birth_year": existing_year,
                        "birth_year_status": normalize_scalar(row.get("birth_year_status")) or "existing",
                        "birth_year_confidence": row.get("birth_year_confidence", ""),
                        "birth_year_raw_prediction": "",
                        "birth_year_model": "",
                        "birth_year_crop_path": str(crop_path),
                    }
                )
                continue

            if not crop_path.exists():
                rows.append(
                    {
                        "writer_id": writer_id,
                        "birth_year": "",
                        "birth_year_status": "missing_crop",
                        "birth_year_confidence": 0.0,
                        "birth_year_raw_prediction": "",
                        "birth_year_model": args.model,
                        "birth_year_crop_path": str(crop_path),
                    }
                )
                continue

            image = prepare_crop(crop_path)
            raw = transcribe_year(model, processor, image, args)
            year, confidence, status = parse_year_prediction(raw, args.min_year, args.max_year)
            if year and confidence < args.min_confidence:
                status = "qwen_needs_review"
            if not year:
                status = "needs_review"
            rows.append(
                {
                    "writer_id": writer_id,
                    "birth_year": year,
                    "birth_year_status": status,
                    "birth_year_confidence": confidence,
                    "birth_year_raw_prediction": raw,
                    "birth_year_model": args.model,
                    "birth_year_crop_path": str(crop_path),
                    "birth_year_device": device,
                    "birth_year_dtype": dtype,
                }
            )
            if idx % args.log_every == 0 or idx == len(forms):
                print(f"birth-year OCR {idx}/{len(forms)}", flush=True)
    finally:
        try:
            del model
            del processor
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return pd.DataFrame(rows)


def safe_clear_dir(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not str(resolved).lower().startswith(str(resolved_root).lower()):
        raise RuntimeError(f"Refusing to delete outside processed directory: {resolved}")
    shutil.rmtree(resolved)


def recompute_needs_review(forms: pd.DataFrame) -> pd.Series:
    review = pd.Series(False, index=forms.index)
    for column in ["sex_conflict", "difficulty_conflict"]:
        if column in forms.columns:
            review |= forms[column].astype(str).str.lower().isin(["true", "1"])
    if "sex" in forms.columns:
        review |= forms["sex"].fillna("").astype(str).eq("unknown")
    if "difficulty_group" in forms.columns:
        review |= forms["difficulty_group"].fillna("").astype(str).eq("unknown")
    review |= forms["birth_year"].fillna("").astype(str).str.strip().eq("")
    if "birth_year_status" in forms.columns:
        review |= forms["birth_year_status"].fillna("").astype(str).str.contains("review|missing", case=False, regex=True)
    return review


def apply_predictions(forms: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    forms = forms.copy()
    predictions = predictions.set_index("writer_id")
    update_columns = [
        "birth_year",
        "birth_year_status",
        "birth_year_confidence",
        "birth_year_raw_prediction",
        "birth_year_model",
        "birth_year_device",
        "birth_year_dtype",
    ]
    for column in update_columns:
        if column not in forms.columns:
            forms[column] = ""
    for idx, row in forms.iterrows():
        writer_id = normalize_scalar(row.get("writer_id"))
        if writer_id not in predictions.index:
            continue
        pred = predictions.loc[writer_id]
        for column in update_columns:
            if column in pred:
                forms.at[idx, column] = pred[column]
    forms["needs_review"] = recompute_needs_review(forms)
    return forms


def update_lines_from_forms(lines: pd.DataFrame, forms: pd.DataFrame) -> pd.DataFrame:
    lines = lines.copy()
    form_columns = [
        "birth_year",
        "birth_year_status",
        "birth_year_confidence",
        "birth_year_raw_prediction",
        "birth_year_model",
        "needs_review",
    ]
    available = ["writer_id"] + [column for column in form_columns if column in forms.columns]
    merged = lines.drop(columns=[c for c in form_columns if c in lines.columns], errors="ignore").merge(
        forms[available], on="writer_id", how="left"
    )
    return merged


def rebuild_exports(lines: pd.DataFrame, forms: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    for name in ["dataset", "trocr", "qwen", "kraken_gt"]:
        safe_clear_dir(processed_dir / name, processed_dir)

    all_lines = lines.copy()
    if "line_exportable" in all_lines.columns:
        export_lines = all_lines[all_lines["line_exportable"].map(is_truthy)].copy()
    else:
        export_lines = all_lines.copy()

    rows = []
    for row in export_lines.to_dict("records"):
        split = slug(row.get("split"), "train")
        difficulty = slug(row.get("difficulty_group"))
        sex = slug(row.get("sex"))
        year = slug(row.get("birth_year"), "year_unknown")
        src = Path(row["flat_image_path"])
        dst = processed_dir / "dataset" / split / difficulty / sex / f"year_{year}" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row["image_path"] = str(dst)
        rows.append(row)
    rebuilt = pd.DataFrame(rows, columns=list(all_lines.columns))

    all_lines.to_csv(processed_dir / "manifest_all.csv", index=False, encoding="utf-8-sig")
    rebuilt.to_csv(processed_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    forms.to_csv(processed_dir / "forms.csv", index=False, encoding="utf-8-sig")
    with (processed_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rebuilt.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for split, split_df in rebuilt.groupby("split", sort=False):
        trocr_dir = processed_dir / "trocr"
        trocr_dir.mkdir(parents=True, exist_ok=True)
        split_df[["image_path", "reference"]].rename(columns={"reference": "text"}).to_csv(
            trocr_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
        )

        qwen_dir = processed_dir / "qwen"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        with (qwen_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_df.to_dict("records"):
                item = {
                    "image": row["image_path"],
                    "prompt": "Transcribe exactly the handwritten Polish text in this image. Return only the transcription.",
                    "response": row["reference"],
                    "metadata": {
                        "sample_id": row["sample_id"],
                        "writer_id": row["writer_id"],
                        "set_id": row["set_id"],
                        "line_id": int(row["line_id"]),
                        "sex": row.get("sex", ""),
                        "birth_year": row.get("birth_year", ""),
                        "difficulty_group": row.get("difficulty_group", ""),
                        "difficulty": row.get("difficulty", ""),
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        kraken_dir = processed_dir / "kraken_gt" / str(split)
        kraken_dir.mkdir(parents=True, exist_ok=True)
        for row in split_df.to_dict("records"):
            src = Path(row["image_path"])
            dst_image = kraken_dir / src.name
            shutil.copy2(src, dst_image)
            dst_image.with_suffix(".gt.txt").write_text(str(row["reference"]), encoding="utf-8")
    return rebuilt


def write_statistics(forms: pd.DataFrame, lines: pd.DataFrame, processed_dir: Path, all_lines: pd.DataFrame | None = None) -> None:
    stats_dir = processed_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    forms_count = forms.copy()
    forms_count["birth_year"] = forms_count["birth_year"].fillna("").replace("", "unknown")
    forms_count["sex"] = forms_count["sex"].fillna("").replace("", "unknown")
    forms_count["difficulty_group"] = forms_count["difficulty_group"].fillna("").replace("", "unknown")

    by_year_sex = pd.crosstab(forms_count["birth_year"], forms_count["sex"], margins=True, margins_name="total")
    by_year_sex.to_csv(stats_dir / "forms_by_year_sex.csv", encoding="utf-8-sig")

    by_year_difficulty_sex = (
        forms_count.groupby(["birth_year", "difficulty_group", "sex"], dropna=False)
        .size()
        .reset_index(name="forms")
        .sort_values(["birth_year", "difficulty_group", "sex"])
    )
    by_year_difficulty_sex.to_csv(stats_dir / "forms_by_year_difficulty_sex.csv", index=False, encoding="utf-8-sig")

    by_difficulty_sex = pd.crosstab(forms_count["difficulty_group"], forms_count["sex"], margins=True, margins_name="total")
    by_difficulty_sex.to_csv(stats_dir / "forms_by_difficulty_sex.csv", encoding="utf-8-sig")

    forms_by_split = forms_count.groupby(["split", "difficulty_group", "sex"], dropna=False).size().reset_index(name="forms")
    forms_by_split.to_csv(stats_dir / "forms_by_split_difficulty_sex.csv", index=False, encoding="utf-8-sig")

    lines_count = lines.copy()
    lines_count["birth_year"] = lines_count["birth_year"].fillna("").replace("", "unknown")
    lines_count["sex"] = lines_count["sex"].fillna("").replace("", "unknown")
    lines_count["difficulty_group"] = lines_count["difficulty_group"].fillna("").replace("", "unknown")
    lines_by_split = lines_count.groupby(["split", "difficulty_group", "sex"], dropna=False).size().reset_index(name="line_samples")
    lines_by_split.to_csv(stats_dir / "lines_by_split_difficulty_sex.csv", index=False, encoding="utf-8-sig")

    if all_lines is not None and not all_lines.empty:
        all_lines_count = all_lines.copy()
        if "line_status" in all_lines_count.columns:
            line_status = all_lines_count.groupby(["set_id", "line_id", "line_status"], dropna=False).size().reset_index(name="line_slots")
            line_status.to_csv(stats_dir / "line_slots_by_status.csv", index=False, encoding="utf-8-sig")
            line_status_by_form = all_lines_count.groupby(["writer_id", "line_status"], dropna=False).size().reset_index(name="line_slots")
            line_status_by_form.to_csv(stats_dir / "line_slots_by_writer_status.csv", index=False, encoding="utf-8-sig")

    summary = {
        "forms": int(len(forms)),
        "line_samples": int(len(lines)),
        "line_slots_total": int(len(all_lines)) if all_lines is not None else int(len(lines)),
        "line_slots_empty_or_uncertain": int(all_lines["line_status"].astype(str).eq("empty_or_uncertain").sum())
        if all_lines is not None and "line_status" in all_lines.columns
        else 0,
        "forms_needing_review": int(forms["needs_review"].astype(str).str.lower().isin(["true", "1"]).sum())
        if "needs_review" in forms.columns
        else 0,
        "birth_year_auto_qwen": int(forms["birth_year_status"].astype(str).eq("auto_qwen").sum())
        if "birth_year_status" in forms.columns
        else 0,
        "birth_year_needs_review": int(forms["birth_year_status"].astype(str).str.contains("review|missing", case=False, regex=True).sum())
        if "birth_year_status" in forms.columns
        else 0,
    }
    (stats_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read birth-year crops with Qwen and update form manifests.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-year", type=int, default=1900)
    parser.add_argument("--max-year", type=int, default=date.today().year)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Re-read years even when birth_year is already filled.")
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Use existing Qwen predictions CSV instead of running the model.")
    parser.add_argument("--no-apply", action="store_true", help="Only write prediction CSV; do not update manifests/exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    forms_path = processed_dir / "forms.csv"
    manifest_path = processed_dir / "manifest.csv"
    manifest_all_path = processed_dir / "manifest_all.csv"
    if not forms_path.exists():
        raise FileNotFoundError(forms_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    forms = pd.read_csv(forms_path, encoding="utf-8-sig", dtype=str).fillna("")
    lines_source_path = manifest_all_path if manifest_all_path.exists() else manifest_path
    lines = pd.read_csv(lines_source_path, encoding="utf-8-sig", dtype=str).fillna("")

    if args.predictions_csv:
        predictions = pd.read_csv(args.predictions_csv, encoding="utf-8-sig", dtype=str).fillna("")
    else:
        predictions = predict_birth_years(forms, args)

    pred_path = processed_dir / "birth_year_qwen_predictions.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"Predictions: {pred_path}")

    if args.no_apply:
        return

    updated_forms = apply_predictions(forms, predictions)
    updated_lines = update_lines_from_forms(lines, updated_forms)
    rebuilt_lines = rebuild_exports(updated_lines, updated_forms, processed_dir)
    updated_all_lines_path = processed_dir / "manifest_all.csv"
    updated_all_lines = pd.read_csv(updated_all_lines_path, encoding="utf-8-sig", dtype=str).fillna("") if updated_all_lines_path.exists() else rebuilt_lines
    write_statistics(updated_forms, rebuilt_lines, processed_dir, updated_all_lines)
    print(f"Updated forms: {processed_dir / 'forms.csv'}")
    print(f"Updated manifest: {processed_dir / 'manifest.csv'}")
    print(f"Stats: {processed_dir / 'stats'}")


if __name__ == "__main__":
    main()
