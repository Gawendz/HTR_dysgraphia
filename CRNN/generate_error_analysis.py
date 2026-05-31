from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
ANALYSIS_DIR = EXPERIMENT_ROOT / "analysis"


def load_predictions() -> pd.DataFrame:
    path = RESULTS_DIR / "kraken_zero_shot_predictions.csv"
    df = pd.read_csv(path)
    df["group"] = df["group"].fillna("IAM-test")
    df["text_id"] = df["text_id"].fillna(-1).astype(int)
    return df


def align_errors(reference: str, prediction: str) -> tuple[Counter, Counter, Counter]:
    reference = reference or ""
    prediction = prediction or ""
    matcher = SequenceMatcher(a=reference, b=prediction, autojunk=False)
    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_part = reference[i1:i2]
        pred_part = prediction[j1:j2]
        if tag == "replace":
            paired = min(len(ref_part), len(pred_part))
            for i in range(paired):
                substitutions[(ref_part[i], pred_part[i])] += 1
            for ch in ref_part[paired:]:
                deletions[ch] += 1
            for ch in pred_part[paired:]:
                insertions[ch] += 1
        elif tag == "delete":
            deletions.update(ref_part)
        elif tag == "insert":
            insertions.update(pred_part)
    return substitutions, deletions, insertions


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower().replace(" ", "_"))


def save_metric_summaries(df: pd.DataFrame) -> None:
    group_summary = (
        df.groupby(["dataset", "group"], dropna=False)
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
    group_summary.to_csv(ANALYSIS_DIR / "group_summary.csv", index=False)

    text_id_summary = (
        df[df["dataset"] == "malaysian"]
        .groupby(["group", "text_id"], dropna=False)
        .agg(
            samples=("sample_id", "count"),
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            wer_median=("wer", "median"),
            inference_seconds_mean=("inference_seconds", "mean"),
        )
        .reset_index()
    )
    text_id_summary.to_csv(ANALYSIS_DIR / "malaysian_text_id_summary.csv", index=False)

    bins = [0, 30, 60, 120, 240, 10_000]
    labels = ["<=30", "31-60", "61-120", "121-240", ">240"]
    length_df = df.copy()
    length_df["reference_length_bin"] = pd.cut(length_df["reference_chars"], bins=bins, labels=labels, right=True)
    length_summary = (
        length_df.groupby(["dataset", "group", "reference_length_bin"], observed=True)
        .agg(
            samples=("sample_id", "count"),
            cer_mean=("cer", "mean"),
            wer_mean=("wer", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
        )
        .reset_index()
    )
    length_summary.to_csv(ANALYSIS_DIR / "length_bin_summary.csv", index=False)


def save_error_tables(df: pd.DataFrame) -> None:
    all_subs: Counter[tuple[str, str]] = Counter()
    all_dels: Counter[str] = Counter()
    all_ins: Counter[str] = Counter()
    by_group: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    error_type_rows = []

    for row in df.to_dict("records"):
        subs, dels, ins = align_errors(str(row["reference"]), str(row["prediction"]))
        all_subs.update(subs)
        all_dels.update(dels)
        all_ins.update(ins)
        group_key = f"{row['dataset']}::{row['group']}"
        by_group[group_key].update(subs)
        error_type_rows.append(
            {
                "dataset": row["dataset"],
                "group": row["group"],
                "sample_id": row["sample_id"],
                "substitutions": sum(subs.values()),
                "deletions": sum(dels.values()),
                "insertions": sum(ins.values()),
            }
        )

    top_subs = pd.DataFrame(
        [
            {"reference_char": a, "predicted_char": b, "count": count}
            for (a, b), count in all_subs.most_common()
        ]
    )
    top_subs.to_csv(ANALYSIS_DIR / "char_substitutions.csv", index=False)

    pd.DataFrame(
        [{"reference_char": ch, "count": count} for ch, count in all_dels.most_common()]
    ).to_csv(ANALYSIS_DIR / "char_deletions.csv", index=False)
    pd.DataFrame(
        [{"predicted_char": ch, "count": count} for ch, count in all_ins.most_common()]
    ).to_csv(ANALYSIS_DIR / "char_insertions.csv", index=False)

    error_types = pd.DataFrame(error_type_rows)
    error_type_summary = (
        error_types.groupby(["dataset", "group"])
        .agg(
            substitutions=("substitutions", "sum"),
            deletions=("deletions", "sum"),
            insertions=("insertions", "sum"),
        )
        .reset_index()
    )
    error_type_summary.to_csv(ANALYSIS_DIR / "error_type_summary.csv", index=False)

    ref_chars = [ch for ch, _ in Counter({a: c for (a, _), c in all_subs.items()}).most_common(24)]
    pred_chars = [ch for ch, _ in Counter({b: c for (_, b), c in all_subs.items()}).most_common(24)]
    matrix = pd.DataFrame(0, index=ref_chars, columns=pred_chars, dtype=int)
    for (a, b), count in all_subs.items():
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] += count
    matrix.to_csv(ANALYSIS_DIR / "char_substitution_matrix_top24.csv")

    for group_key, subs in by_group.items():
        if not subs:
            continue
        ref_chars_g = [ch for ch, _ in Counter({a: c for (a, _), c in subs.items()}).most_common(18)]
        pred_chars_g = [ch for ch, _ in Counter({b: c for (_, b), c in subs.items()}).most_common(18)]
        matrix_g = pd.DataFrame(0, index=ref_chars_g, columns=pred_chars_g, dtype=int)
        for (a, b), count in subs.items():
            if a in matrix_g.index and b in matrix_g.columns:
                matrix_g.loc[a, b] += count
        matrix_g.to_csv(ANALYSIS_DIR / f"char_substitution_matrix_{safe_name(group_key)}.csv")


def plot_group_bars(df: pd.DataFrame) -> None:
    group_summary = pd.read_csv(ANALYSIS_DIR / "group_summary.csv")
    group_summary["label"] = group_summary["dataset"] + " / " + group_summary["group"]
    x = range(len(group_summary))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar([i - 0.18 for i in x], group_summary["cer_mean"], width=0.36, label="CER")
    ax.bar([i + 0.18 for i in x], group_summary["wer_mean"], width=0.36, label="WER")
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_summary["label"], rotation=25, ha="right")
    ax.set_ylim(0, max(1.2, group_summary[["cer_mean", "wer_mean"]].max().max() * 1.15))
    ax.set_ylabel("Error rate")
    ax.set_title("Mean CER/WER by Dataset and Group")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "cer_wer_by_dataset_group.png", dpi=180)
    plt.close(fig)


def plot_text_id_bars() -> None:
    summary = pd.read_csv(ANALYSIS_DIR / "malaysian_text_id_summary.csv")
    groups = sorted(summary["group"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, metric in zip(axes, ["cer_mean", "wer_mean"]):
        pivot = summary.pivot(index="text_id", columns="group", values=metric).reindex(columns=groups)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("text_id / line number")
        ax.set_ylabel("Error rate")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Malaysian Error by Line Number and Group", y=1.02)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "malaysian_error_by_text_id.png", dpi=180)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame) -> None:
    mal = df[df["dataset"] == "malaysian"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, metric in zip(axes, ["cer", "wer"]):
        data = [mal[mal["group"] == group][metric].clip(upper=2.0) for group in ["LPD", "PD"]]
        ax.boxplot(data, tick_labels=["LPD", "PD"], showfliers=False)
        ax.set_title(metric.upper() + " distribution")
        ax.set_ylabel("Error rate")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "malaysian_error_distribution_lpd_pd.png", dpi=180)
    plt.close(fig)


def plot_confusion_heatmap() -> None:
    matrix = pd.read_csv(ANALYSIS_DIR / "char_substitution_matrix_top24.csv", index_col=0)
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 8))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([repr(x)[1:-1] if x != " " else "SPACE" for x in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([repr(x)[1:-1] if x != " " else "SPACE" for x in matrix.index])
    ax.set_xlabel("Predicted character")
    ax.set_ylabel("Reference character")
    ax.set_title("Top Character Substitutions")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "char_substitution_heatmap_top24.png", dpi=180)
    plt.close(fig)


def plot_top_substitutions() -> None:
    subs = pd.read_csv(ANALYSIS_DIR / "char_substitutions.csv").head(20)
    if subs.empty:
        return
    subs["pair"] = subs.apply(
        lambda r: f"{'SPACE' if r['reference_char'] == ' ' else r['reference_char']} -> {'SPACE' if r['predicted_char'] == ' ' else r['predicted_char']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(subs["pair"][::-1], subs["count"][::-1])
    ax.set_xlabel("Count")
    ax.set_title("Top 20 Character Substitutions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "top_char_substitutions.png", dpi=180)
    plt.close(fig)


def plot_error_types() -> None:
    errors = pd.read_csv(ANALYSIS_DIR / "error_type_summary.csv")
    errors["label"] = errors["dataset"] + " / " + errors["group"]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bottom = None
    for col in ["substitutions", "deletions", "insertions"]:
        ax.bar(errors["label"], errors[col], bottom=bottom, label=col)
        bottom = errors[col] if bottom is None else bottom + errors[col]
    ax.set_title("Character Error Operation Mix")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "error_operation_mix.png", dpi=180)
    plt.close(fig)


def plot_latency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, part in df.groupby("dataset"):
        ax.scatter(part["image_width"], part["inference_seconds"], s=14, alpha=0.55, label=label)
    ax.set_xlabel("Image width")
    ax.set_ylabel("Inference seconds")
    ax.set_title("Latency vs Image Width")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "latency_vs_image_width.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    save_metric_summaries(df)
    save_error_tables(df)
    plot_group_bars(df)
    plot_text_id_bars()
    plot_distributions(df)
    plot_confusion_heatmap()
    plot_top_substitutions()
    plot_error_types()
    plot_latency(df)
    metadata = {"analysis_files": sorted(p.name for p in ANALYSIS_DIR.iterdir())}
    (ANALYSIS_DIR / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
