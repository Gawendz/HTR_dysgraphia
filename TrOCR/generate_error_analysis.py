from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
ANALYSIS_DIR = EXPERIMENT_ROOT / "analysis"


def normalize_model(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-handwritten", "")


def align_substitutions(reference: str, prediction: str) -> Counter[tuple[str, str]]:
    matcher = SequenceMatcher(a=reference or "", b=prediction or "", autojunk=False)
    substitutions: Counter[tuple[str, str]] = Counter()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        ref_part = reference[i1:i2]
        pred_part = prediction[j1:j2]
        for idx in range(min(len(ref_part), len(pred_part))):
            substitutions[(ref_part[idx], pred_part[idx])] += 1
    return substitutions


def make_summaries(df: pd.DataFrame) -> None:
    group_summary = (
        df.groupby(["model_id", "dataset", "group"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            wer_median=("wer", "median"),
            char_accuracy_mean=("char_accuracy", "mean"),
            word_accuracy_mean=("word_accuracy", "mean"),
            exact_match_rate=("exact_match", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
            inference_seconds_median=("inference_seconds", "median"),
        )
        .reset_index()
    )
    group_summary["model_short"] = group_summary["model_id"].map(normalize_model)
    group_summary.to_csv(ANALYSIS_DIR / "group_summary.csv", index=False)

    text_id_summary = (
        df[df["dataset"] == "malaysian"]
        .groupby(["model_id", "group", "text_id"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
        )
        .reset_index()
    )
    text_id_summary["model_short"] = text_id_summary["model_id"].map(normalize_model)
    text_id_summary.to_csv(ANALYSIS_DIR / "malaysian_text_id_summary.csv", index=False)

    best_model = (
        group_summary[group_summary["dataset"] == "malaysian"]
        .groupby("model_id")["cer_mean"]
        .mean()
        .sort_values()
        .index[0]
    )
    sub_counter: Counter[tuple[str, str]] = Counter()
    for row in df[df["model_id"] == best_model].to_dict("records"):
        sub_counter.update(align_substitutions(str(row["reference"]), str(row["prediction"])))
    pd.DataFrame(
        [{"reference_char": a, "predicted_char": b, "count": count} for (a, b), count in sub_counter.most_common()]
    ).to_csv(ANALYSIS_DIR / "best_model_char_substitutions.csv", index=False)

    ref_chars = [ch for ch, _ in Counter({a: c for (a, _), c in sub_counter.items()}).most_common(24)]
    pred_chars = [ch for ch, _ in Counter({b: c for (_, b), c in sub_counter.items()}).most_common(24)]
    matrix = pd.DataFrame(0, index=ref_chars, columns=pred_chars, dtype=int)
    for (a, b), count in sub_counter.items():
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] += count
    matrix.to_csv(ANALYSIS_DIR / "best_model_char_substitution_matrix_top24.csv")
    (ANALYSIS_DIR / "best_model.json").write_text(json.dumps({"model_id": best_model}, indent=2), encoding="utf-8")


def plot_model_comparison() -> None:
    summary = pd.read_csv(ANALYSIS_DIR / "group_summary.csv")
    dataset_summary = (
        summary.groupby(["model_id", "model_short", "dataset"])
        .agg(cer_mean=("cer_mean", "mean"), wer_mean=("wer_mean", "mean"), inference_seconds_mean=("inference_seconds_mean", "mean"))
        .reset_index()
    )
    labels = dataset_summary["model_short"] + " / " + dataset_summary["dataset"]
    x = range(len(dataset_summary))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - 0.18 for i in x], dataset_summary["cer_mean"], width=0.36, label="CER")
    ax.bar([i + 0.18 for i in x], dataset_summary["wer_mean"], width=0.36, label="WER")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("TrOCR Zero-Shot Error by Model and Dataset")
    ax.set_ylabel("Error rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "trocr_model_dataset_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for dataset in sorted(dataset_summary["dataset"].unique()):
        part = dataset_summary[dataset_summary["dataset"] == dataset]
        ax.plot(part["model_short"], part["inference_seconds_mean"], marker="o", label=dataset)
    ax.set_title("TrOCR Mean Inference Time")
    ax.set_ylabel("Seconds / sample")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "trocr_latency_by_model.png", dpi=180)
    plt.close(fig)


def plot_malaysian_group() -> None:
    summary = pd.read_csv(ANALYSIS_DIR / "group_summary.csv")
    mal = summary[summary["dataset"] == "malaysian"]
    labels = mal["model_short"] + " / " + mal["group"]
    x = range(len(mal))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - 0.18 for i in x], mal["cer_mean"], width=0.36, label="CER")
    ax.bar([i + 0.18 for i in x], mal["wer_mean"], width=0.36, label="WER")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("TrOCR Malaysian LPD vs PD")
    ax.set_ylabel("Error rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "trocr_malaysian_lpd_pd_error.png", dpi=180)
    plt.close(fig)


def plot_confusion_heatmap() -> None:
    matrix = pd.read_csv(ANALYSIS_DIR / "best_model_char_substitution_matrix_top24.csv", index_col=0)
    if matrix.empty:
        return
    best_model = json.loads((ANALYSIS_DIR / "best_model.json").read_text(encoding="utf-8"))["model_id"]
    fig, ax = plt.subplots(figsize=(10.5, 8))
    im = ax.imshow(matrix.values, cmap="Purples")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(["SPACE" if x == " " else repr(x)[1:-1] for x in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels(["SPACE" if x == " " else repr(x)[1:-1] for x in matrix.index])
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("Reference character")
    ax.set_title(f"Best TrOCR Character Substitutions: {normalize_model(best_model)}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "trocr_best_model_char_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "trocr_zero_shot_predictions.csv")
    df["group"] = df["group"].fillna("IAM-test")
    make_summaries(df)
    plot_model_comparison()
    plot_malaysian_group()
    plot_confusion_heatmap()
    print(json.dumps({"analysis_files": sorted(p.name for p in ANALYSIS_DIR.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
