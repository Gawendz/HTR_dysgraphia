# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd


FORM_COLUMNS_TO_LINES = [
    "sex",
    "difficulty",
    "difficulty_group",
    "birth_year",
    "birth_year_status",
    "split",
    "needs_review",
    "reviewed",
    "review_status",
    "review_note",
    "review_updated_at",
]

REVIEW_COLUMNS = {
    "reviewed": False,
    "review_status": "pending",
    "review_note": "",
    "review_updated_at": "",
}

TEXT_EDIT_COLUMNS = [
    "sex",
    "difficulty",
    "difficulty_group",
    "birth_year",
    "birth_year_status",
    "split",
    "review_status",
    "review_note",
    "review_updated_at",
    "reference",
    "gt_text",
    "gt_status",
    "gt_source",
    "line_status",
]


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def is_truthy(value: Any) -> bool:
    return normalize_scalar(value).lower() in {"true", "1", "yes", "tak"}


def normalize_gt_status(value: Any) -> str:
    text = normalize_scalar(value).lower()
    if text in {"uncertain", "niepewne"}:
        return "uncertain"
    if text in {"exclude", "excluded", "wyklucz", "wykluczone"}:
        return "exclude"
    return "ok"


def slug(value: object, default: str = "unknown") -> str:
    text = normalize_scalar(value).lower()
    if not text:
        return default
    table = str.maketrans(
        {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
        }
    )
    text = text.translate(table)
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or default


def clean_json_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    clean = clean.where(pd.notnull(clean), "")
    records = clean.to_dict("records")
    return [{key: clean_json_value(value) for key, value in row.items()} for row in records]


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".bak_before_review_app")
    if not backup.exists():
        shutil.copy2(path, backup)


def ensure_object_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("object")
    return df


def suspicious_reason(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    sex = normalize_scalar(row.get("sex"))
    difficulty_group = normalize_scalar(row.get("difficulty_group"))
    if is_truthy(row.get("sex_conflict")):
        reasons.append("konflikt plci")
    if is_truthy(row.get("difficulty_conflict")):
        reasons.append("konflikt trudnosci")
    if sex in {"", "unknown"}:
        reasons.append("nieznana plec")
    if difficulty_group in {"", "unknown"}:
        reasons.append("nieznany typ trudnosci")
    return reasons


def recompute_needs_review(row: pd.Series) -> bool:
    if normalize_scalar(row.get("review_status")) == "validated":
        return False
    if suspicious_reason(row):
        return True
    year = normalize_scalar(row.get("birth_year"))
    if not year:
        return True
    status = normalize_scalar(row.get("birth_year_status")).lower()
    if "review" in status or "missing" in status:
        return True
    return False


def write_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


class ReviewStore:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir.resolve()
        self.forms_path = self.processed_dir / "forms.csv"
        self.manifest_all_path = self.processed_dir / "manifest_all.csv"
        self.manifest_path = self.processed_dir / "manifest.csv"
        self.manifest_jsonl_path = self.processed_dir / "manifest.jsonl"
        self.lock = threading.Lock()
        self.forms = pd.DataFrame()
        self.lines_all = pd.DataFrame()
        self.lines_exported = pd.DataFrame()
        self.load()

    def load(self) -> None:
        if not self.forms_path.exists():
            raise FileNotFoundError(self.forms_path)
        if not self.manifest_all_path.exists():
            raise FileNotFoundError(self.manifest_all_path)

        self.forms = pd.read_csv(self.forms_path, encoding="utf-8-sig")
        self.lines_all = pd.read_csv(self.manifest_all_path, encoding="utf-8-sig")
        self.lines_exported = (
            pd.read_csv(self.manifest_path, encoding="utf-8-sig") if self.manifest_path.exists() else pd.DataFrame()
        )
        self.forms = ensure_object_columns(self.forms, TEXT_EDIT_COLUMNS)
        self.lines_all = ensure_object_columns(self.lines_all, TEXT_EDIT_COLUMNS)
        if not self.lines_exported.empty:
            self.lines_exported = ensure_object_columns(self.lines_exported, TEXT_EDIT_COLUMNS)

        for column, default in REVIEW_COLUMNS.items():
            if column not in self.forms.columns:
                self.forms[column] = default
            if column not in self.lines_all.columns:
                self.lines_all[column] = default
            if not self.lines_exported.empty and column not in self.lines_exported.columns:
                self.lines_exported[column] = default

        for column in ["reference", "gt_text", "gt_status", "gt_source"]:
            if column not in self.lines_all.columns:
                self.lines_all[column] = ""
            if not self.lines_exported.empty and column not in self.lines_exported.columns:
                self.lines_exported[column] = ""

    def save_csvs(self) -> None:
        backup_once(self.forms_path)
        backup_once(self.manifest_all_path)
        backup_once(self.manifest_path)
        self.forms.to_csv(self.forms_path, index=False, encoding="utf-8-sig")
        self.lines_all.to_csv(self.manifest_all_path, index=False, encoding="utf-8-sig")
        if not self.lines_exported.empty:
            self.lines_exported.to_csv(self.manifest_path, index=False, encoding="utf-8-sig")
            with self.manifest_jsonl_path.open("w", encoding="utf-8") as handle:
                for row in df_records(self.lines_exported):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def list_forms(self, filter_name: str) -> dict[str, Any]:
        forms = self.forms.copy()
        reasons = forms.apply(suspicious_reason, axis=1)
        forms["_reason_count"] = reasons.map(len)
        forms["_reasons"] = reasons.map(lambda items: ", ".join(items))

        if filter_name == "suspicious":
            filtered = forms[forms["_reason_count"] > 0]
        elif filter_name == "unreviewed":
            reviewed = forms.get("reviewed", pd.Series(False, index=forms.index)).map(is_truthy)
            filtered = forms[~reviewed]
        elif filter_name == "empty_lines":
            bad = self.lines_all[self.lines_all["line_status"].astype(str) != "filled"]["writer_id"].astype(str).unique()
            filtered = forms[forms["writer_id"].astype(str).isin(set(bad))]
        else:
            filtered = forms

        payload = filtered[
            [
                "writer_id",
                "set_id",
                "sex",
                "difficulty",
                "difficulty_group",
                "birth_year",
                "reviewed",
                "review_status",
                "_reasons",
            ]
        ].copy()
        return {
            "forms": df_records(payload),
            "counts": {
                "all": int(len(forms)),
                "suspicious": int((forms["_reason_count"] > 0).sum()),
                "unreviewed": int((~forms.get("reviewed", pd.Series(False, index=forms.index)).map(is_truthy)).sum()),
                "empty_lines": int(
                    self.lines_all[self.lines_all["line_status"].astype(str) != "filled"]["writer_id"].nunique()
                ),
            },
        }

    def get_form(self, writer_id: str) -> dict[str, Any]:
        match = self.forms[self.forms["writer_id"].astype(str) == writer_id]
        if match.empty:
            raise KeyError(writer_id)
        form = match.iloc[0].copy()
        form["_reasons"] = suspicious_reason(form)
        lines = self.lines_all[self.lines_all["writer_id"].astype(str) == writer_id].sort_values("line_id")
        return {"form": df_records(pd.DataFrame([form]))[0], "lines": df_records(lines)}

    def update_form(self, payload: dict[str, Any]) -> dict[str, Any]:
        writer_id = normalize_scalar(payload.get("writer_id"))
        if not writer_id:
            raise ValueError("Missing writer_id")

        with self.lock:
            form_idx = self.forms.index[self.forms["writer_id"].astype(str) == writer_id]
            if len(form_idx) == 0:
                raise KeyError(writer_id)
            idx = form_idx[0]
            now = datetime.now().isoformat(timespec="seconds")
            form_payload = payload.get("form", {})

            editable_form_columns = [
                "sex",
                "difficulty",
                "difficulty_group",
                "birth_year",
                "birth_year_status",
                "split",
                "review_note",
                "review_status",
            ]
            for column in editable_form_columns:
                if column in form_payload:
                    self.forms.at[idx, column] = normalize_scalar(form_payload[column])

            reviewed = bool(form_payload.get("reviewed", False))
            self.forms.at[idx, "reviewed"] = reviewed
            if reviewed and normalize_scalar(self.forms.at[idx, "review_status"]) == "pending":
                self.forms.at[idx, "review_status"] = "validated"
            if not reviewed and normalize_scalar(self.forms.at[idx, "review_status"]) == "validated":
                self.forms.at[idx, "review_status"] = "pending"

            sex = normalize_scalar(self.forms.at[idx, "sex"])
            difficulty = normalize_scalar(self.forms.at[idx, "difficulty"])
            self.forms.at[idx, "sex_conflict"] = "+" in sex
            self.forms.at[idx, "difficulty_conflict"] = "+" in difficulty
            if normalize_scalar(self.forms.at[idx, "birth_year"]) and not normalize_scalar(self.forms.at[idx, "birth_year_status"]):
                self.forms.at[idx, "birth_year_status"] = "manual"
            self.forms.at[idx, "review_updated_at"] = now
            self.forms.at[idx, "needs_review"] = recompute_needs_review(self.forms.loc[idx])

            writer_mask_all = self.lines_all["writer_id"].astype(str) == writer_id
            for column in FORM_COLUMNS_TO_LINES:
                if column in self.forms.columns:
                    if column not in self.lines_all.columns:
                        self.lines_all[column] = ""
                    self.lines_all.loc[writer_mask_all, column] = self.forms.at[idx, column]

            line_updates = payload.get("lines", [])
            exported_ids: set[str] = set()
            for line in line_updates:
                sample_id = normalize_scalar(line.get("sample_id"))
                if not sample_id:
                    continue
                exported_ids.add(sample_id)
                mask = self.lines_all["sample_id"].astype(str) == sample_id
                if not mask.any():
                    continue
                for column in ["gt_text", "reference", "line_status"]:
                    if column in line:
                        value = normalize_scalar(line[column])
                        if column not in self.lines_all.columns:
                            self.lines_all[column] = ""
                        self.lines_all.loc[mask, column] = value
                if "gt_text" in line:
                    self.lines_all.loc[mask, "reference"] = normalize_scalar(line["gt_text"])
                    self.lines_all.loc[mask, "gt_source"] = "review_app"
                gt_status = normalize_gt_status(line.get("gt_status"))
                self.lines_all.loc[mask, "gt_status"] = gt_status
                line_status = normalize_scalar(self.lines_all.loc[mask, "line_status"].iloc[0])
                self.lines_all.loc[mask, "line_exportable"] = gt_status == "ok" and line_status == "filled"

            if not self.lines_exported.empty:
                writer_mask_exp = self.lines_exported["writer_id"].astype(str) == writer_id
                for column in FORM_COLUMNS_TO_LINES:
                    if column in self.forms.columns:
                        if column not in self.lines_exported.columns:
                            self.lines_exported[column] = ""
                        self.lines_exported.loc[writer_mask_exp, column] = self.forms.at[idx, column]

                for sample_id in exported_ids:
                    all_row = self.lines_all[self.lines_all["sample_id"].astype(str) == sample_id]
                    exp_mask = self.lines_exported["sample_id"].astype(str) == sample_id
                    if all_row.empty:
                        continue
                    if not is_truthy(all_row.iloc[0].get("line_exportable")):
                        if exp_mask.any():
                            self.lines_exported = self.lines_exported.loc[~exp_mask].copy()
                        continue
                    if not exp_mask.any():
                        continue
                    for column in ["gt_text", "reference", "line_status", "gt_status", "gt_source"]:
                        if column in self.lines_exported.columns and column in all_row.columns:
                            self.lines_exported.loc[exp_mask, column] = all_row.iloc[0][column]

            self.save_csvs()
            return self.get_form(writer_id)

    def rebuild_exports(self) -> dict[str, Any]:
        with self.lock:
            for name in ["dataset", "trocr", "qwen", "kraken_gt"]:
                target = (self.processed_dir / name).resolve()
                if target.exists():
                    if self.processed_dir not in target.parents:
                        raise RuntimeError(f"Refusing to delete outside processed dir: {target}")
                    shutil.rmtree(target)

            export_lines = self.lines_all[self.lines_all["line_exportable"].map(is_truthy)].copy()
            rows = []
            for row in df_records(export_lines):
                split = slug(row.get("split"), "train")
                difficulty = slug(row.get("difficulty_group"))
                sex = slug(row.get("sex"))
                year = slug(row.get("birth_year"), "year_unknown")
                src = Path(str(row["flat_image_path"]))
                dst = self.processed_dir / "dataset" / split / difficulty / sex / f"year_{year}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                row["image_path"] = str(dst)
                rows.append(row)
            self.lines_exported = pd.DataFrame(rows, columns=list(self.lines_all.columns))
            self.save_csvs()
            self.write_training_exports()
            self.write_basic_stats()
            return {
                "exported_lines": int(len(self.lines_exported)),
                "message": "Eksporty przebudowane.",
            }

    def write_training_exports(self) -> None:
        if self.lines_exported.empty:
            return
        for split, split_df in self.lines_exported.groupby("split", sort=False):
            trocr_dir = self.processed_dir / "trocr"
            trocr_dir.mkdir(parents=True, exist_ok=True)
            split_df[["image_path", "reference"]].rename(columns={"reference": "text"}).to_csv(
                trocr_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
            )

            qwen_dir = self.processed_dir / "qwen"
            qwen_dir.mkdir(parents=True, exist_ok=True)
            with (qwen_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for row in df_records(split_df):
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

            kraken_dir = self.processed_dir / "kraken_gt" / str(split)
            kraken_dir.mkdir(parents=True, exist_ok=True)
            for row in df_records(split_df):
                src = Path(str(row["image_path"]))
                dst_image = kraken_dir / src.name
                shutil.copy2(src, dst_image)
                dst_image.with_suffix(".gt.txt").write_text(str(row["reference"]), encoding="utf-8")

    def write_basic_stats(self) -> None:
        stats_dir = self.processed_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        forms = self.forms.copy()
        for column in ["birth_year", "sex", "difficulty_group", "split"]:
            if column not in forms.columns:
                forms[column] = "unknown"
            forms[column] = forms[column].fillna("").replace("", "unknown")

        pd.crosstab(forms["birth_year"], forms["sex"], margins=True, margins_name="total").to_csv(
            stats_dir / "forms_by_year_sex.csv", encoding="utf-8-sig"
        )
        (
            forms.groupby(["birth_year", "difficulty_group", "sex"], dropna=False)
            .size()
            .reset_index(name="forms")
            .sort_values(["birth_year", "difficulty_group", "sex"])
            .to_csv(stats_dir / "forms_by_year_difficulty_sex.csv", index=False, encoding="utf-8-sig")
        )
        (
            forms.groupby(["difficulty_group", "sex"], dropna=False)
            .size()
            .reset_index(name="forms")
            .sort_values(["difficulty_group", "sex"])
            .to_csv(stats_dir / "forms_by_difficulty_sex.csv", index=False, encoding="utf-8-sig")
        )

    def allowed_file(self, raw_path: str) -> Path:
        path = Path(raw_path).resolve()
        allowed_roots = [self.processed_dir]
        raw_parent = self.processed_dir.parent / "raw_examples"
        if raw_parent.exists():
            allowed_roots.append(raw_parent.resolve())
        if not any(path == root or root in path.parents for root in allowed_roots):
            raise PermissionError(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path


APP_HTML = r"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Walidacja formularzy</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, sans-serif; background: #eef1f4; color: #111; }
    header { height: 54px; display: flex; align-items: center; gap: 10px; padding: 8px 14px; background: #20242a; color: white; position: sticky; top: 0; z-index: 5; }
    header select, header input, header button { height: 34px; border-radius: 6px; border: 1px solid #666; padding: 0 10px; }
    header button { background: #f2f4f7; cursor: pointer; }
    header .status { margin-left: auto; font-size: 13px; opacity: .9; }
    main { display: grid; grid-template-columns: minmax(520px, 1fr) minmax(520px, 620px); gap: 12px; padding: 12px; }
    .viewer, .panel, .lines { background: white; border: 1px solid #d7dbe0; border-radius: 8px; }
    .viewer { min-height: calc(100vh - 78px); overflow: auto; display: flex; justify-content: center; align-items: flex-start; padding: 12px; }
    #formImage { width: min(100%, 900px); transform-origin: top center; border: 1px solid #c7ccd1; background: #ddd; }
    .side { display: flex; flex-direction: column; gap: 10px; max-height: calc(100vh - 78px); min-height: calc(100vh - 78px); overflow: hidden; }
    .panel { padding: 12px; }
    h1, h2 { font-size: 16px; margin: 0 0 10px; }
    .row { display: grid; grid-template-columns: 112px 1fr; gap: 8px; align-items: center; margin-bottom: 7px; }
    label { font-size: 13px; color: #343a40; }
    input, select, textarea { width: 100%; border: 1px solid #b8c0c8; border-radius: 6px; padding: 7px; font: inherit; background: white; }
    textarea { min-height: 52px; resize: vertical; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    .actions button, .wideButton { border: 0; border-radius: 6px; padding: 9px; cursor: pointer; background: #285c9f; color: white; font-weight: 700; }
    .actions button.secondary, .wideButton.secondary { background: #e7ebef; color: #111; border: 1px solid #ccd2d8; }
    .pill { display: inline-block; background: #fff1b8; border: 1px solid #d2b847; border-radius: 999px; padding: 3px 8px; font-size: 12px; margin: 0 4px 6px 0; }
    .muted { color: #68717a; font-size: 13px; }
    .lines { padding: 12px; flex: 1; min-height: 0; overflow: auto; }
    .lineCard { border-top: 1px solid #e4e8ec; padding: 12px 0; }
    .lineCard:first-child { border-top: 0; padding-top: 0; }
    .lineCard img { width: 100%; border: 1px solid #cfd5db; background: #eee; }
    .lineMeta { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px; }
    .lineMeta code { background: #edf0f2; padding: 2px 5px; border-radius: 4px; }
    .lineGrid { display: grid; grid-template-columns: 94px 1fr; gap: 8px; align-items: start; margin-top: 8px; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      .side { max-height: none; overflow: visible; }
      .lines { overflow: visible; }
    }
  </style>
</head>
<body>
  <header>
    <strong>Walidacja</strong>
    <select id="filter">
      <option value="suspicious">metadane podejrzane</option>
      <option value="unreviewed">nieprzejrzane</option>
      <option value="empty_lines">z pustymi liniami</option>
      <option value="all">wszystkie</option>
    </select>
    <button id="prevBtn">Poprzedni</button>
    <button id="nextBtn">Następny</button>
    <input id="search" placeholder="writer_id, np. 043_p01">
    <button id="goBtn">Idź</button>
    <button id="zoomOut">-</button>
    <button id="zoomIn">+</button>
    <span class="status" id="status">Ładowanie...</span>
  </header>
  <main>
    <section class="viewer">
      <img id="formImage" alt="formularz">
    </section>
    <aside class="side">
      <section class="panel">
        <h1 id="title">Formularz</h1>
        <div id="reasons"></div>
        <div class="row"><label>Zestaw</label><input id="set_id" readonly></div>
        <div class="row"><label>Płeć</label>
          <select id="sex">
            <option value="unknown">unknown</option>
            <option value="kobieta">kobieta</option>
            <option value="mezczyzna">mezczyzna</option>
            <option value="kobieta+mezczyzna">kobieta+mezczyzna</option>
          </select>
        </div>
        <div class="row"><label>Trudności raw</label><input id="difficulty"></div>
        <div class="row"><label>Grupa</label>
          <select id="difficulty_group">
            <option value="unknown">unknown</option>
            <option value="nie">nie</option>
            <option value="dysgrafia">dysgrafia</option>
            <option value="inne">inne</option>
          </select>
        </div>
        <div class="row"><label>Rok ur.</label><input id="birth_year" inputmode="numeric"></div>
        <div class="row"><label>Split</label>
          <select id="split">
            <option value="train">train</option>
            <option value="val">val</option>
            <option value="test">test</option>
          </select>
        </div>
        <div class="actions">
          <button id="saveBtn">Zapisz</button>
          <button id="saveNextBtn">Zapisz i następny</button>
          <button class="secondary" id="reloadBtn">Odśwież</button>
          <button class="secondary" id="rebuildBtn">Przebuduj eksporty</button>
        </div>
      </section>
      <section class="lines">
        <h2>Linie i ground truth</h2>
        <div id="lines"></div>
      </section>
    </aside>
  </main>
<script>
let forms = [];
let currentIndex = 0;
let current = null;
let zoom = 1;

const $ = (id) => document.getElementById(id);
const media = (path) => `/media?path=${encodeURIComponent(path || '')}`;

function setStatus(text) { $("status").textContent = text; }

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function boolValue(v) {
  return v === true || String(v).toLowerCase() === "true" || String(v) === "1";
}

function gtStatusValue(line) {
  const raw = String(line.gt_status || "").toLowerCase();
  if (raw === "uncertain" || raw === "niepewne") return "uncertain";
  if (raw === "exclude" || raw === "excluded" || raw === "wyklucz") return "exclude";
  if (!boolValue(line.line_exportable)) return "exclude";
  return "ok";
}

async function loadList(keepWriter = null) {
  const filter = $("filter").value;
  const data = await api(`/api/forms?filter=${encodeURIComponent(filter)}`);
  forms = data.forms;
  const counts = data.counts;
  setStatus(`${forms.length} w filtrze | wszystkie ${counts.all} | podejrzane ${counts.suspicious} | nieprzejrzane ${counts.unreviewed}`);
  if (!forms.length) {
    current = null;
    $("formImage").removeAttribute("src");
    $("title").textContent = "Brak formularzy";
    return;
  }
  const wanted = keepWriter ? forms.findIndex(f => f.writer_id === keepWriter) : currentIndex;
  currentIndex = wanted >= 0 ? wanted : Math.min(currentIndex, forms.length - 1);
  await loadForm(forms[currentIndex].writer_id);
}

async function jump(index) {
  if (index < 0 || index >= forms.length) return;
  currentIndex = index;
  await loadForm(forms[currentIndex].writer_id);
}

async function loadForm(writerId) {
  current = await api(`/api/form?writer_id=${encodeURIComponent(writerId)}`);
  const f = current.form;
  $("title").textContent = `${f.writer_id} (${currentIndex + 1}/${forms.length})`;
  $("search").value = f.writer_id;
  $("formImage").src = media(f.aligned_path);

  for (const id of ["set_id","sex","difficulty","difficulty_group","birth_year","split"]) {
    $(id).value = f[id] || "";
  }
  $("reasons").innerHTML = (f._reasons || []).map(r => `<span class="pill">${r}</span>`).join("") || `<span class="muted">Brak podejrzanych checkboxów.</span>`;
  renderLines(current.lines);
}

function renderLines(lines) {
  $("lines").innerHTML = lines.map(line => `
    <div class="lineCard" data-sample-id="${line.sample_id}" data-line-status="${line.line_status || ''}">
      <div class="lineMeta">
        <code>${line.sample_id}</code>
        <span>${line.line_status || ""}</span>
        <span class="muted">line ${line.line_id}</span>
      </div>
      <img src="${media(line.flat_image_path)}" alt="${line.sample_id}">
      <div class="lineGrid">
        <label>GT</label>
        <textarea class="gt">${line.gt_text || line.reference || ""}</textarea>
      </div>
      <div class="lineGrid">
        <label>GT status</label>
        <select class="gtStatus">
          <option value="ok" ${gtStatusValue(line) === "ok" ? "selected" : ""}>ok</option>
          <option value="uncertain" ${gtStatusValue(line) === "uncertain" ? "selected" : ""}>niepewne</option>
          <option value="exclude" ${gtStatusValue(line) === "exclude" ? "selected" : ""}>wyklucz</option>
        </select>
      </div>
      <div class="muted">${line.prompt_reference || ""}</div>
    </div>
  `).join("");
}

function collectPayload() {
  const form = {};
  for (const id of ["sex","difficulty","difficulty_group","birth_year","split"]) {
    form[id] = $(id).value;
  }
  form.birth_year_status = form.birth_year ? "manual" : "";
  form.review_status = "validated";
  form.review_note = "";
  form.reviewed = true;
  const lines = [...document.querySelectorAll(".lineCard")].map(card => ({
    sample_id: card.dataset.sampleId,
    gt_text: card.querySelector(".gt").value,
    gt_status: card.querySelector(".gtStatus").value,
    line_status: card.dataset.lineStatus || "filled",
  }));
  return { writer_id: current.form.writer_id, form, lines };
}

async function save(stay = true) {
  if (!current) return;
  setStatus("Zapisywanie...");
  current = await api("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectPayload()),
  });
  const writer = current.form.writer_id;
  await loadList(writer);
  setStatus(`Zapisano ${writer}`);
  if (!stay) await jump(Math.min(currentIndex + 1, forms.length - 1));
}

$("filter").addEventListener("change", () => loadList());
$("prevBtn").addEventListener("click", () => jump(currentIndex - 1));
$("nextBtn").addEventListener("click", () => jump(currentIndex + 1));
$("goBtn").addEventListener("click", async () => {
  const writer = $("search").value.trim();
  const idx = forms.findIndex(f => f.writer_id === writer);
  if (idx >= 0) await jump(idx);
  else await loadForm(writer);
});
$("saveBtn").addEventListener("click", () => save(true));
$("saveNextBtn").addEventListener("click", () => save(false));
$("reloadBtn").addEventListener("click", () => loadList(current?.form?.writer_id));
$("zoomIn").addEventListener("click", () => { zoom = Math.min(2.4, zoom + 0.1); $("formImage").style.transform = `scale(${zoom})`; });
$("zoomOut").addEventListener("click", () => { zoom = Math.max(0.5, zoom - 0.1); $("formImage").style.transform = `scale(${zoom})`; });
$("rebuildBtn").addEventListener("click", async () => {
  if (!confirm("Przebudować foldery dataset/trocr/qwen/kraken_gt według aktualnych CSV?")) return;
  setStatus("Przebudowuję eksporty...");
  const result = await api("/api/rebuild", { method: "POST" });
  setStatus(`${result.message} Linie: ${result.exported_lines}`);
});
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key.toLowerCase() === "s") { event.preventDefault(); save(true); }
  if (event.altKey && event.key === "ArrowRight") { event.preventDefault(); jump(currentIndex + 1); }
  if (event.altKey && event.key === "ArrowLeft") { event.preventDefault(); jump(currentIndex - 1); }
});

loadList().catch(err => setStatus(err.message));
</script>
</body>
</html>
"""


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = APP_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/forms":
                    filter_name = query.get("filter", ["suspicious"])[0]
                    write_json(self, store.list_forms(filter_name))
                    return
                if parsed.path == "/api/form":
                    writer_id = query.get("writer_id", [""])[0]
                    write_json(self, store.get_form(writer_id))
                    return
                if parsed.path == "/media":
                    raw = query.get("path", [""])[0]
                    path = store.allowed_file(raw)
                    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    data = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                write_json(self, {"error": "Not found"}, 404)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, 500)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/api/save":
                    payload = read_json(self)
                    write_json(self, store.update_form(payload))
                    return
                if parsed.path == "/api/rebuild":
                    write_json(self, store.rebuild_exports())
                    return
                write_json(self, {"error": "Not found"}, 404)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, 500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local review app for processed handwriting forms.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed_320_check"),
        help="Directory created by process_forms.py.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    store = ReviewStore(args.processed_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Review app: http://{args.host}:{args.port}")
    print(f"Processed dir: {store.processed_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
