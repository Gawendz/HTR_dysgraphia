from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark_utils import summarize_group_predictions, summarize_predictions  # noqa: E402


MODEL_SPECS = [
    {"family": "CRNN/Kraken", "folder": "CRNN", "stem_prefix": "crnn_polish_forms"},
    {"family": "TrOCR", "folder": "TrOCR", "stem_prefix": "trocr_polish_forms"},
    {"family": "Qwen3VL", "folder": "Qwen3VL", "stem_prefix": "qwen3vl_polish_forms"},
]


def main() -> None:
    split_label = sys.argv[1] if len(sys.argv) > 1 else "test"
    result_root = EXPERIMENT_ROOT / "results" / split_label
    comparison_dir = EXPERIMENT_ROOT / "comparison" / split_label
    analysis_dir = EXPERIMENT_ROOT / "analysis" / split_label
    reset_dir(comparison_dir)
    reset_dir(analysis_dir)

    summaries: list[pd.DataFrame] = []
    group_summaries: list[pd.DataFrame] = []
    predictions_all: list[pd.DataFrame] = []
    for spec in MODEL_SPECS:
        model_dir = result_root / spec["folder"]
        stem = f"{spec['stem_prefix']}_{split_label}"
        prediction_path = model_dir / f"{stem}_predictions.csv"
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        summary = summarize_predictions(predictions)
        group_summary = summarize_group_predictions(predictions)
        for df in [predictions, summary, group_summary]:
            set_family(df, spec["family"])
            add_model_short(df)
        predictions.to_csv(model_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8")
        summary.to_csv(model_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
        group_summary.to_csv(model_dir / f"{stem}_group_summary.csv", index=False, encoding="utf-8")
        summaries.append(summary)
        group_summaries.append(group_summary)
        predictions_all.append(predictions)
        model_analysis = analysis_dir / spec["folder"]
        model_analysis.mkdir(parents=True, exist_ok=True)
        render_table(make_summary_table(summary), f"{spec['family']} - Polish Forms {split_label}", model_analysis / "summary_table.png")
        plot_grouped_cer_wer(summary, f"{spec['family']}: Polish Forms {split_label} CER/WER", model_analysis / "cer_wer.png")
        plot_grouped_cla_crw(summary, f"{spec['family']}: Polish Forms {split_label} CLA/CRW", model_analysis / "cla_crw.png")
        plot_group_metric(group_summary, "corpus_cer", f"{spec['family']}: Corpus CER by Difficulty", "Corpus CER", model_analysis / "difficulty_corpus_cer.png")
        plot_group_metric(group_summary, "corpus_wer", f"{spec['family']}: Corpus WER by Difficulty", "Corpus WER", model_analysis / "difficulty_corpus_wer.png")

    if not summaries:
        raise FileNotFoundError(f"No Polish Forms results found under {result_root}")

    summary_all = pd.concat(summaries, ignore_index=True)
    group_all = pd.concat(group_summaries, ignore_index=True)
    predictions_all_df = pd.concat(predictions_all, ignore_index=True)
    summary_report, group_report = make_report_scope(predictions_all_df, split_label, summary_all, group_all)
    summary_all.to_csv(comparison_dir / "summary_by_dataset.csv", index=False, encoding="utf-8")
    group_all.to_csv(comparison_dir / "group_summary_by_dataset.csv", index=False, encoding="utf-8")
    summary_report.to_csv(comparison_dir / "summary.csv", index=False, encoding="utf-8")
    group_report.to_csv(comparison_dir / "group_summary.csv", index=False, encoding="utf-8")
    predictions_all_df.to_csv(comparison_dir / "predictions.csv", index=False, encoding="utf-8")

    summary_table = make_summary_table(summary_report)
    group_table = make_group_table(group_report)
    summary_table.to_csv(comparison_dir / "summary_table.csv", index=False, encoding="utf-8")
    group_table.to_csv(comparison_dir / "difficulty_group_table.csv", index=False, encoding="utf-8")
    render_table(summary_table, f"Polish Forms {split_label} - Summary", comparison_dir / "summary_table.png")
    render_table(group_table, f"Polish Forms {split_label} - Difficulty Groups", comparison_dir / "difficulty_group_table.png")
    plot_grouped_cer_wer(summary_report, f"Polish Forms {split_label} CER/WER by Model", comparison_dir / "polish_forms_cer_wer.png")
    plot_grouped_cla_crw(summary_report, f"Polish Forms {split_label} CLA/CRW by Model", comparison_dir / "polish_forms_cla_crw.png")
    plot_group_metric(group_report, "corpus_cer", f"Polish Forms {split_label} Corpus CER by Difficulty", "Corpus CER", comparison_dir / "polish_forms_difficulty_cer.png")
    plot_group_metric(group_report, "corpus_wer", f"Polish Forms {split_label} Corpus WER by Difficulty", "Corpus WER", comparison_dir / "polish_forms_difficulty_wer.png")
    plot_bar(
        summary_report.sort_values("inference_seconds_mean")["inference_seconds_mean"],
        model_labels(summary_report.sort_values("inference_seconds_mean")),
        "#808080",
        f"Polish Forms {split_label} Mean Inference Time",
        "Seconds/line",
        comparison_dir / "polish_forms_latency.png",
    )
    plot_vram(summary_report, comparison_dir / "polish_forms_vram.png")
    write_workbook(
        summary_report,
        group_report,
        predictions_all_df,
        comparison_dir / f"polish_forms_zero_shot_{split_label}_results.xlsx",
        summary_by_dataset=summary_all,
        group_by_dataset=group_all,
    )
    (comparison_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "split": split_label,
                "result_root": str(result_root),
                "comparison_dir": str(comparison_dir),
                "analysis_dir": str(analysis_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(comparison_dir)


def add_model_short(df: pd.DataFrame) -> None:
    if "model_id" not in df.columns:
        return
    df["model_short"] = (
        df["model_id"]
        .astype(str)
        .str.replace("microsoft/", "", regex=False)
        .str.replace("-handwritten", "", regex=False)
        .str.replace("Qwen/", "", regex=False)
        .str.replace("CRNN/Kraken", "CRNN/Kraken", regex=False)
    )


def set_family(df: pd.DataFrame, family: str) -> None:
    if "family" in df.columns:
        df["family"] = family
    else:
        df.insert(0, "family", family)


def add_family_from_model(df: pd.DataFrame) -> None:
    if "model_id" not in df.columns:
        return
    family = df["model_id"].astype(str).map(model_family)
    if "family" in df.columns:
        df["family"] = family
    else:
        df.insert(0, "family", family)
    add_model_short(df)


def model_family(model_id: str) -> str:
    if model_id == "CRNN/Kraken":
        return "CRNN/Kraken"
    if model_id.startswith("microsoft/trocr"):
        return "TrOCR"
    if model_id.startswith("Qwen/"):
        return "Qwen3VL"
    return "Other"


def make_report_scope(
    predictions: pd.DataFrame,
    split_label: str,
    summary_by_dataset: pd.DataFrame,
    group_by_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_label != "all":
        return summary_by_dataset, group_by_dataset
    combined = predictions.copy()
    combined["dataset"] = "polish_forms_all"
    combined["source_dataset"] = "polish_forms"
    combined["preprocessing_variant"] = "native"
    summary = summarize_predictions(combined)
    group_summary = summarize_group_predictions(combined)
    for df in [summary, group_summary]:
        add_family_from_model(df)
    return summary, group_summary


def model_labels(df: pd.DataFrame) -> pd.Series:
    family = df["family"].astype(str)
    model = df["model_short"].astype(str)
    return family.where(model.eq(family), family + "\n" + model)


def make_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "family",
        "model_short",
        "samples",
        "corpus_cer",
        "corpus_wer",
        "corpus_cla",
        "corpus_crw",
        "exact_match_rate",
        "inference_seconds_mean",
        "rss_after_mb_max",
        "cuda_peak_reserved_mb_max",
    ]
    out = summary[cols].copy().sort_values("corpus_cer")
    out = out.rename(
        columns={
            "family": "Family",
            "model_short": "Model",
            "samples": "N",
            "corpus_cer": "Corpus CER",
            "corpus_wer": "Corpus WER",
            "corpus_cla": "Corpus CLA",
            "corpus_crw": "Corpus CRW",
            "exact_match_rate": "Exact Match",
            "inference_seconds_mean": "Sec/line",
            "rss_after_mb_max": "RSS max MB",
            "cuda_peak_reserved_mb_max": "VRAM reserv MB",
        }
    )
    return format_numeric(out)


def make_group_table(group_summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["family", "model_short", "group", "samples", "corpus_cer", "corpus_wer", "corpus_cla", "corpus_crw", "inference_seconds_mean"]
    out = group_summary[cols].copy().sort_values(["group", "corpus_cer"])
    out = out.rename(
        columns={
            "family": "Family",
            "model_short": "Model",
            "group": "Difficulty Group",
            "samples": "N",
            "corpus_cer": "Corpus CER",
            "corpus_wer": "Corpus WER",
            "corpus_cla": "Corpus CLA",
            "corpus_crw": "Corpus CRW",
            "inference_seconds_mean": "Sec/line",
        }
    )
    return format_numeric(out)


def format_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "N":
            out[col] = out[col].astype(int).astype(str)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "N/A" if pd.isna(value) else f"{value:.3f}")
    return out


def render_table(df: pd.DataFrame, title: str, path: Path) -> None:
    rows = max(len(df), 1)
    cols = max(len(df.columns), 1)
    fig, ax = plt.subplots(figsize=(min(max(cols * 1.45, 9), 20), min(max(rows * 0.42 + 1.2, 3.2), 14)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6FA")
        else:
            cell.set_facecolor("#FFFFFF")
        cell.set_edgecolor("#D9E2F3")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bar(values: pd.Series, labels: pd.Series, colors, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(values) * 1.35), 5.8))
    ax.bar(np.arange(len(values)), values.astype(float), color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_cer_wer(summary: pd.DataFrame, title: str, path: Path) -> None:
    data = summary.copy().sort_values("corpus_cer")
    labels = model_labels(data)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.5), 5.8))
    ax.bar(x - width / 2, data["corpus_cer"].astype(float), width, label="CER", color="#2F75B5")
    ax.bar(x + width / 2, data["corpus_wer"].astype(float), width, label="WER", color="#C65911")
    ax.set_title(title)
    ax.set_ylabel("Error rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_grouped_cla_crw(summary: pd.DataFrame, title: str, path: Path) -> None:
    data = summary.copy().sort_values("corpus_cla", ascending=False)
    labels = model_labels(data)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.5), 5.8))
    ax.bar(x - width / 2, data["corpus_cla"].astype(float), width, label="CLA", color="#70AD47")
    ax.bar(x + width / 2, data["corpus_crw"].astype(float), width, label="CRW", color="#8064A2")
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_metric(group_summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    data = group_summary.copy().sort_values(["group", metric])
    labels = model_labels(data) + "\n" + data["group"].astype(str)
    colors = data["group"].astype(str).map({"nie": "#2F75B5", "dysgrafia": "#C65911", "inne": "#70AD47"}).fillna("#808080")
    plot_bar(data[metric], labels, colors, title, ylabel, path)


def plot_vram(summary: pd.DataFrame, path: Path) -> None:
    data = summary.copy()
    data["cuda_peak_reserved_mb_max"] = pd.to_numeric(data["cuda_peak_reserved_mb_max"], errors="coerce")
    data = data.dropna(subset=["cuda_peak_reserved_mb_max"])
    data = data[data["cuda_peak_reserved_mb_max"].gt(0)].sort_values("cuda_peak_reserved_mb_max")
    if data.empty:
        return
    plot_bar(data["cuda_peak_reserved_mb_max"], model_labels(data), "#70AD47", "Polish Forms Peak VRAM", "Peak reserved VRAM MB", path)


def write_workbook(
    summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    summary_by_dataset: pd.DataFrame | None = None,
    group_by_dataset: pd.DataFrame | None = None,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        make_summary_table(summary).to_excel(writer, sheet_name="Summary formatted", index=False)
        make_group_table(group_summary).to_excel(writer, sheet_name="Groups formatted", index=False)
        summary.to_excel(writer, sheet_name="Summary raw", index=False)
        group_summary.to_excel(writer, sheet_name="Group raw", index=False)
        if summary_by_dataset is not None:
            summary_by_dataset.to_excel(writer, sheet_name="Summary by dataset", index=False)
        if group_by_dataset is not None:
            group_by_dataset.to_excel(writer, sheet_name="Groups by dataset", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)


def reset_dir(path: Path) -> None:
    resolved = path.resolve()
    root = EXPERIMENT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to reset outside experiment root: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
