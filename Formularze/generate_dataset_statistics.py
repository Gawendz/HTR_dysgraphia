# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DIFFICULTY_LABELS = {
    "nie": "brak trudności",
    "dysgrafia": "dysgrafia",
    "dysleksja": "dysleksja",
    "dysortografia": "dysortografia",
    "inne": "inne",
    "unknown": "unknown",
    "": "unknown",
}

GROUP_LABELS = {
    "nie": "brak trudności",
    "dysgrafia": "dysgrafia",
    "inne": "inne",
    "unknown": "unknown",
    "": "unknown",
}

SEX_LABELS = {
    "kobieta": "kobieta",
    "mezczyzna": "mężczyzna",
    "mężczyzna": "mężczyzna",
    "unknown": "unknown",
    "": "unknown",
}

STATUS_LABELS = {
    "ok": "ok",
    "manual_reviewed": "ok",
    "template_auto": "ok",
    "uncertain": "niepewne",
    "exclude": "wyklucz",
    "empty_line": "pusta linia",
}

FIG_DPI = 220


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_columns(df: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    out = df.copy()
    for column in ["sex", "difficulty", "difficulty_group", "split", "gt_status", "line_status"]:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].map(norm_text)

    out["sex_label"] = out["sex"].str.lower().map(SEX_LABELS).fillna(out["sex"].replace("", "unknown"))
    out["difficulty_raw_label"] = (
        out["difficulty"].str.lower().map(DIFFICULTY_LABELS).fillna(out["difficulty"].replace("", "unknown"))
    )
    out["difficulty_group_label"] = (
        out["difficulty_group"].str.lower().map(GROUP_LABELS).fillna(out["difficulty_group"].replace("", "unknown"))
    )
    out["gt_status_label"] = out["gt_status"].str.lower().map(STATUS_LABELS).fillna(out["gt_status"].replace("", "unknown"))

    out["birth_year_num"] = pd.to_numeric(out.get("birth_year", ""), errors="coerce")
    out["birth_year_label"] = out["birth_year_num"].dropna().astype("Int64").astype(str)
    out["birth_year_label"] = out["birth_year_label"].reindex(out.index).fillna("unknown")
    out.loc[out["birth_year_label"].eq("<NA>"), "birth_year_label"] = "unknown"

    out["age_years"] = reference_year - out["birth_year_num"]
    out["age_label"] = out["age_years"].dropna().astype("Int64").astype(str)
    out["age_label"] = out["age_label"].reindex(out.index).fillna("unknown")
    out.loc[out["age_label"].eq("<NA>"), "age_label"] = "unknown"

    if "line_exportable" in out.columns:
        out["line_exportable_bool"] = out["line_exportable"].astype(str).str.lower().isin(["true", "1", "tak", "yes"])
    return out


def count_table(df: pd.DataFrame, columns: str | list[str], label: str) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    result = df.groupby(columns, dropna=False).size().reset_index(name=label)
    result["percent"] = (result[label] / max(1, result[label].sum()) * 100).round(2)
    return result.sort_values([label] + columns, ascending=[False] + [True] * len(columns)).reset_index(drop=True)


def crosstab(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    table = pd.crosstab(df[index], df[columns], margins=True, margins_name="razem")
    return table.reset_index()


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_bar(table: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, color: str = "#3b6ea8") -> None:
    plot_df = table.copy()
    plot_df[x] = plot_df[x].astype(str)
    fig_w = max(7.0, min(13.0, 0.55 * len(plot_df) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    bars = ax.bar(plot_df[x], plot_df[y], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_stacked(pivot: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    plot_df = pivot.copy()
    if "razem" in plot_df.columns:
        plot_df = plot_df.drop(columns=["razem"])
    if "razem" in plot_df.index:
        plot_df = plot_df.drop(index=["razem"])
    fig_w = max(7.0, min(13.0, 0.7 * len(plot_df) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    data = data.fillna("")
    headers = list(data.columns)
    rows = [[str(value) for value in row] for row in data.to_numpy()]
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = "| " + " | ".join(str(header).ljust(width) for header, width in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def count_files(base: Path) -> dict[str, int]:
    qwen_rows = 0
    qwen_dir = base / "qwen"
    if qwen_dir.exists():
        for path in qwen_dir.glob("*.jsonl"):
            qwen_rows += sum(1 for _ in path.open(encoding="utf-8"))

    trocr_rows = 0
    trocr_dir = base / "trocr"
    if trocr_dir.exists():
        for path in trocr_dir.glob("*.csv"):
            trocr_rows += len(pd.read_csv(path))

    return {
        "dataset_png": len(list((base / "dataset").rglob("*.png"))) if (base / "dataset").exists() else 0,
        "kraken_png": len(list((base / "kraken_gt").rglob("*.png"))) if (base / "kraken_gt").exists() else 0,
        "kraken_gt_txt": len(list((base / "kraken_gt").rglob("*.gt.txt"))) if (base / "kraken_gt").exists() else 0,
        "qwen_jsonl_rows": qwen_rows,
        "trocr_csv_rows": trocr_rows,
    }


def generate(processed_dir: Path, output_dir: Path, reference_year: int) -> None:
    forms_path = processed_dir / "forms.csv"
    manifest_all_path = processed_dir / "manifest_all.csv"
    manifest_path = processed_dir / "manifest.csv"
    forms = normalize_columns(pd.read_csv(forms_path, encoding="utf-8-sig"), reference_year)
    all_lines = normalize_columns(pd.read_csv(manifest_all_path, encoding="utf-8-sig"), reference_year)
    exported = normalize_columns(pd.read_csv(manifest_path, encoding="utf-8-sig"), reference_year)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    tables["forms_by_difficulty_raw"] = count_table(forms, "difficulty_raw_label", "forms")
    tables["forms_by_difficulty_group"] = count_table(forms, "difficulty_group_label", "forms")
    tables["forms_by_sex"] = count_table(forms, "sex_label", "forms")
    tables["forms_by_birth_year"] = count_table(forms, "birth_year_label", "forms")
    tables["forms_by_age_rocznikowy"] = count_table(forms, "age_label", "forms")
    tables["forms_by_split"] = count_table(forms, "split", "forms")
    tables["forms_by_gt_review_status"] = count_table(forms, "review_status", "forms")
    tables["forms_by_difficulty_group_and_sex"] = count_table(
        forms, ["difficulty_group_label", "sex_label"], "forms"
    )
    tables["forms_by_birth_year_and_sex"] = crosstab(forms, "birth_year_label", "sex_label")
    tables["forms_by_split_and_difficulty_group"] = crosstab(forms, "split", "difficulty_group_label")
    tables["forms_by_split_difficulty_group_sex"] = count_table(
        forms, ["split", "difficulty_group_label", "sex_label"], "forms"
    )

    tables["line_slots_by_gt_status"] = count_table(all_lines, "gt_status_label", "line_slots")
    tables["line_slots_by_line_status"] = count_table(all_lines, "line_status", "line_slots")
    tables["line_slots_by_exportable"] = count_table(all_lines, "line_exportable_bool", "line_slots")
    tables["all_line_slots_by_difficulty_raw"] = count_table(all_lines, "difficulty_raw_label", "line_slots")
    tables["all_line_slots_by_difficulty_group"] = count_table(all_lines, "difficulty_group_label", "line_slots")

    tables["exported_images_by_difficulty_raw"] = count_table(exported, "difficulty_raw_label", "images")
    tables["exported_images_by_difficulty_group"] = count_table(exported, "difficulty_group_label", "images")
    tables["exported_images_by_sex"] = count_table(exported, "sex_label", "images")
    tables["exported_images_by_birth_year"] = count_table(exported, "birth_year_label", "images")
    tables["exported_images_by_age_rocznikowy"] = count_table(exported, "age_label", "images")
    tables["exported_images_by_split"] = count_table(exported, "split", "images")
    tables["exported_images_by_difficulty_group_and_sex"] = count_table(
        exported, ["difficulty_group_label", "sex_label"], "images"
    )
    tables["exported_images_by_birth_year_and_sex"] = crosstab(exported, "birth_year_label", "sex_label")
    tables["exported_images_by_split_and_difficulty_group"] = crosstab(
        exported, "split", "difficulty_group_label"
    )
    tables["exported_images_by_split_difficulty_group_sex"] = count_table(
        exported, ["split", "difficulty_group_label", "sex_label"], "images"
    )

    for name, table in tables.items():
        write_table(table, tables_dir / f"{name}.csv")

    with pd.ExcelWriter(output_dir / "dataset_statistics.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            sheet = name[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)

    plot_bar(
        tables["forms_by_difficulty_raw"],
        "difficulty_raw_label",
        "forms",
        "Liczba formularzy według deklarowanej trudności",
        "Liczba formularzy",
        figures_dir / "forms_by_difficulty_raw.png",
    )
    plot_bar(
        tables["exported_images_by_difficulty_raw"],
        "difficulty_raw_label",
        "images",
        "Liczba obrazów linii według deklarowanej trudności",
        "Liczba obrazów linii",
        figures_dir / "exported_images_by_difficulty_raw.png",
        color="#667f3f",
    )
    plot_bar(
        tables["forms_by_difficulty_group"],
        "difficulty_group_label",
        "forms",
        "Liczba formularzy według grupy",
        "Liczba formularzy",
        figures_dir / "forms_by_difficulty_group.png",
    )
    plot_bar(
        tables["exported_images_by_difficulty_group"],
        "difficulty_group_label",
        "images",
        "Liczba obrazów linii według grupy",
        "Liczba obrazów linii",
        figures_dir / "exported_images_by_difficulty_group.png",
        color="#667f3f",
    )
    plot_bar(tables["forms_by_sex"], "sex_label", "forms", "Liczba formularzy według płci", "Liczba formularzy", figures_dir / "forms_by_sex.png")
    plot_bar(
        tables["exported_images_by_split"],
        "split",
        "images",
        "Podział obrazów linii na zbiory",
        "Liczba obrazów linii",
        figures_dir / "exported_images_by_split.png",
        color="#9a5b38",
    )
    birth_forms = tables["forms_by_birth_year"].sort_values("birth_year_label")
    birth_forms = birth_forms[birth_forms["birth_year_label"] != "unknown"]
    plot_bar(
        birth_forms,
        "birth_year_label",
        "forms",
        "Liczba formularzy według roku urodzenia",
        "Liczba formularzy",
        figures_dir / "forms_by_birth_year.png",
    )
    birth_images = tables["exported_images_by_birth_year"].sort_values("birth_year_label")
    birth_images = birth_images[birth_images["birth_year_label"] != "unknown"]
    plot_bar(
        birth_images,
        "birth_year_label",
        "images",
        "Liczba obrazów linii według roku urodzenia",
        "Liczba obrazów linii",
        figures_dir / "exported_images_by_birth_year.png",
        color="#667f3f",
    )
    age_forms = tables["forms_by_age_rocznikowy"].sort_values("age_label")
    age_forms = age_forms[age_forms["age_label"] != "unknown"]
    plot_bar(
        age_forms,
        "age_label",
        "forms",
        f"Liczba formularzy według wieku rocznikowego ({reference_year} - rok urodzenia)",
        "Liczba formularzy",
        figures_dir / "forms_by_age_rocznikowy.png",
    )

    forms_sex_pivot = pd.crosstab(forms["difficulty_group_label"], forms["sex_label"])
    plot_stacked(
        forms_sex_pivot,
        "Formularze: grupa trudności i płeć",
        "Liczba formularzy",
        figures_dir / "forms_difficulty_group_by_sex.png",
    )
    images_split_pivot = pd.crosstab(exported["split"], exported["difficulty_group_label"])
    plot_stacked(
        images_split_pivot,
        "Obrazy linii: split i grupa trudności",
        "Liczba obrazów linii",
        figures_dir / "exported_images_split_by_difficulty_group.png",
    )

    verification = count_files(processed_dir)
    summary = {
        "processed_dir": str(processed_dir),
        "reference_year": reference_year,
        "forms": int(len(forms)),
        "line_slots_total": int(len(all_lines)),
        "exported_line_images_manifest": int(len(exported)),
        **verification,
        "non_exported_line_slots": int(len(all_lines) - len(exported)),
        "forms_reviewed": int(forms["reviewed"].astype(str).str.lower().isin(["true", "1", "tak", "yes"]).sum())
        if "reviewed" in forms.columns
        else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Statystyki datasetu",
        "",
        f"Folder danych: `{processed_dir}`",
        f"Rok referencyjny do wieku rocznikowego: `{reference_year}`.",
        "",
        "## Podsumowanie",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    ["formularze / osoby", summary["forms"]],
                    ["wszystkie sloty linii", summary["line_slots_total"]],
                    ["wyeksportowane obrazy linii", summary["exported_line_images_manifest"]],
                    ["linie niewyeksportowane", summary["non_exported_line_slots"]],
                    ["pliki PNG w dataset/", summary["dataset_png"]],
                    ["wiersze TrOCR CSV", summary["trocr_csv_rows"]],
                    ["wiersze Qwen JSONL", summary["qwen_jsonl_rows"]],
                    ["pliki Kraken PNG", summary["kraken_png"]],
                    ["pliki Kraken GT", summary["kraken_gt_txt"]],
                ],
                columns=["metryka", "wartość"],
            )
        ),
        "",
        "## Formularze według deklarowanej trudności",
        "",
        markdown_table(tables["forms_by_difficulty_raw"]),
        "",
        "## Wyeksportowane obrazy linii według deklarowanej trudności",
        "",
        markdown_table(tables["exported_images_by_difficulty_raw"]),
        "",
        "## Formularze według grupy",
        "",
        markdown_table(tables["forms_by_difficulty_group"]),
        "",
        "## Wyeksportowane obrazy linii według grupy",
        "",
        markdown_table(tables["exported_images_by_difficulty_group"]),
        "",
        "## Formularze według płci",
        "",
        markdown_table(tables["forms_by_sex"]),
        "",
        "## Formularze według roku urodzenia",
        "",
        markdown_table(tables["forms_by_birth_year"].sort_values("birth_year_label")),
        "",
        "## Podział formularzy na train/val/test",
        "",
        markdown_table(tables["forms_by_split"]),
        "",
        "## Podział obrazów linii na train/val/test",
        "",
        markdown_table(tables["exported_images_by_split"]),
        "",
        "## Status GT dla wszystkich slotów linii",
        "",
        markdown_table(tables["line_slots_by_gt_status"]),
        "",
        "## Pliki",
        "",
        f"- Tabele CSV: `{tables_dir}`",
        f"- Wykresy PNG: `{figures_dir}`",
        f"- Arkusz XLSX: `{output_dir / 'dataset_statistics.xlsx'}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready dataset statistics tables and figures.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-year", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else processed_dir / "stats_thesis"
    generate(processed_dir, output_dir, args.reference_year)
    print(output_dir)


if __name__ == "__main__":
    main()
