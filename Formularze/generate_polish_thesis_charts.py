from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"C:\Magisterka\Formularze\processed_320_check")
OUT_DIR = BASE / "stats_thesis" / "figures"
TABLE_DIR = BASE / "stats_thesis" / "tables"
REFERENCE_YEAR = 2026


SEX_LABELS = {
    "kobieta": "kobieta",
    "mezczyzna": "mężczyzna",
    "mężczyzna": "mężczyzna",
    "unknown": "nieokreślona",
    "kobieta+mezczyzna": "niejednoznaczna",
}

DIFFICULTY_LABELS = {
    "nie": "brak trudności",
    "brak trudności": "brak trudności",
    "dysgrafia": "dysgrafia",
    "inne": "inne trudności",
    "unknown": "nieokreślone",
}

SEX_COLORS = {
    "kobieta": "#D55E00",
    "mężczyzna": "#0072B2",
    "nieokreślona": "#8A8F98",
    "niejednoznaczna": "#9B59B6",
}

DIFFICULTY_COLORS = {
    "brak trudności": "#4E79A7",
    "dysgrafia": "#E15759",
    "inne trudności": "#59A14F",
    "nieokreślone": "#8A8F98",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pct_label(count: int, total: int) -> str:
    percent = 100 * count / total if total else 0
    return f"{count}\n({percent:.1f}%)".replace(".", ",")


def normalize_forms(forms: pd.DataFrame) -> pd.DataFrame:
    df = forms.copy()
    df["sex_pl"] = df["sex"].fillna("unknown").astype(str).str.strip().map(SEX_LABELS).fillna("nieokreślona")
    df["difficulty_pl"] = (
        df["difficulty_group"].fillna("unknown").astype(str).str.strip().map(DIFFICULTY_LABELS).fillna("nieokreślone")
    )
    df["birth_year_num"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age_rocznikowy"] = REFERENCE_YEAR - df["birth_year_num"]
    return df


def normalize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.copy()
    df["sex_pl"] = df["sex"].fillna("unknown").astype(str).str.strip().map(SEX_LABELS).fillna("nieokreślona")
    df["difficulty_pl"] = (
        df["difficulty_group"].fillna("unknown").astype(str).str.strip().map(DIFFICULTY_LABELS).fillna("nieokreślone")
    )
    return df


def bar_age_by_sex(forms: pd.DataFrame) -> Path:
    df = forms.dropna(subset=["age_rocznikowy"]).copy()
    df = df[df["sex_pl"].isin(["kobieta", "mężczyzna"])]
    df["age_rocznikowy"] = df["age_rocznikowy"].astype(int)
    sex_order = ["kobieta", "mężczyzna"]
    table = (
        df.pivot_table(index="age_rocznikowy", columns="sex_pl", values="writer_id", aggfunc="count", fill_value=0)
        .reindex(columns=[s for s in sex_order if s in df["sex_pl"].unique()], fill_value=0)
        .sort_index()
    )
    table.to_csv(TABLE_DIR / "uczestnicy_wiek_rocznikowy_plec.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    bottom = np.zeros(len(table))
    x = np.arange(len(table.index))
    for sex in table.columns:
        values = table[sex].to_numpy()
        ax.bar(x, values, bottom=bottom, label=sex, color=SEX_COLORS.get(sex, "#999999"), width=0.68)
        bottom += values

    totals = table.sum(axis=1).to_numpy()
    for i, total in enumerate(totals):
        ax.text(i, total + 1.0, str(int(total)), ha="center", va="bottom", fontsize=10)

    ax.set_title("Liczba uczestników według wieku rocznikowego i płci")
    ax.set_xlabel("Wiek rocznikowy [lata]")
    ax.set_ylabel("Liczba uczestników")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in table.index])
    ax.set_ylim(0, max(totals) * 1.18)
    ax.yaxis.grid(True, linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Płeć", frameon=False, ncols=min(3, len(table.columns)), loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "uczestnicy_wiek_rocznikowy_plec.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def pie_chart(
    series: pd.Series,
    title: str,
    colors: dict[str, str],
    filename: str,
    note: str | None = None,
) -> Path:
    counts = series.value_counts()
    labels = list(counts.index)
    values = counts.to_numpy()
    total = int(values.sum())
    color_values = [colors.get(label, "#999999") for label in labels]
    counts.to_csv(TABLE_DIR / filename.replace(".png", ".csv"), encoding="utf-8-sig")

    def autopct(pct: float) -> str:
        absolute = int(round(pct * total / 100))
        if pct < 3:
            return ""
        return f"{absolute}\n{pct:.1f}%".replace(".", ",")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    wedges, _, autotexts = ax.pie(
        values,
        colors=color_values,
        startangle=90,
        counterclock=False,
        autopct=autopct,
        pctdistance=0.68,
        textprops={"fontsize": 9, "color": "#111111"},
        wedgeprops={"linewidth": 1.2, "edgecolor": "white", "width": 0.82},
    )
    ax.set_title(title)
    ax.axis("equal")

    for autotext in autotexts:
        autotext.set_fontweight("bold")

    legend_labels = [f"{label}: {pct_label(int(value), total).replace(chr(10), ' ')}" for label, value in zip(labels, values)]
    ax.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12 if note is None else -0.18),
        ncols=min(2, len(labels)),
        frameon=False,
    )
    if note:
        fig.text(0.5, 0.025, note, ha="center", va="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.04 if note is None else 0.11, 1, 1])
    out = OUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def horizontal_line_images_by_difficulty(manifest: pd.DataFrame) -> Path:
    order = ["brak trudności", "dysgrafia", "inne trudności", "nieokreślone"]
    counts = manifest["difficulty_pl"].value_counts().reindex(order).dropna().astype(int)
    counts.to_csv(TABLE_DIR / "obrazy_linii_grupa_trudnosci_pl.csv", encoding="utf-8-sig")

    labels = list(counts.index)
    values = counts.to_numpy()
    total = int(values.sum())
    colors = [DIFFICULTY_COLORS.get(label, "#999999") for label in labels]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Liczba obrazów linii")
    ax.set_title("Liczba obrazów linii według grupy trudności")
    ax.xaxis.grid(True, linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

    xmax = max(values) * 1.18
    ax.set_xlim(0, xmax)
    for i, value in enumerate(values):
        ax.text(value + xmax * 0.015, i, pct_label(int(value), total).replace("\n", " "), va="center", fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / "obrazy_linii_grupa_trudnosci_pl.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    forms = normalize_forms(pd.read_csv(BASE / "forms.csv"))
    manifest = normalize_manifest(pd.read_csv(BASE / "manifest.csv"))

    paths = [
        bar_age_by_sex(forms),
        pie_chart(
            forms["difficulty_pl"],
            "Uczestnicy według deklarowanej grupy trudności",
            DIFFICULTY_COLORS,
            "uczestnicy_grupa_trudnosci_kolowy.png",
        ),
        pie_chart(
            forms.loc[forms["sex_pl"].isin(["kobieta", "mężczyzna"]), "sex_pl"],
            "Uczestnicy według płci",
            SEX_COLORS,
            "uczestnicy_plec_kolowy.png",
        ),
        horizontal_line_images_by_difficulty(manifest),
    ]

    print("Wygenerowano wykresy:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
