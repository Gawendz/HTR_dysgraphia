from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


def rel_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def file_link(path: Path, base: Path) -> str:
    return rel_path(path, base).replace("#", "%23")


def truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "tak"}


def reason_for_form(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    sex = str(row.get("sex", "")).strip()
    difficulty = str(row.get("difficulty_group", "")).strip()
    if truthy(row.get("sex_conflict")):
        reasons.append("konflikt plci")
    if truthy(row.get("difficulty_conflict")):
        reasons.append("konflikt trudnosci")
    if sex in {"", "unknown", "nan"}:
        reasons.append("nieznana plec")
    if difficulty in {"", "unknown", "nan"}:
        reasons.append("nieznany typ trudnosci")
    return reasons


def make_header_crop(aligned_path: Path, out_path: Path) -> None:
    with Image.open(aligned_path) as img:
        w, h = img.size
        crop = img.crop((0, 0, w, int(h * 0.38)))
        crop.save(out_path)


def make_html(processed_dir: Path, output: Path) -> None:
    forms_path = processed_dir / "forms.csv"
    manifest_all_path = processed_dir / "manifest_all.csv"

    if not forms_path.exists():
        raise FileNotFoundError(forms_path)
    if not manifest_all_path.exists():
        raise FileNotFoundError(manifest_all_path)

    forms = pd.read_csv(forms_path)
    manifest_all = pd.read_csv(manifest_all_path)

    assets_dir = output.parent / "review_suspicious_assets"
    header_dir = assets_dir / "headers"
    lines_dir = assets_dir / "lines"
    header_dir.mkdir(parents=True, exist_ok=True)
    lines_dir.mkdir(parents=True, exist_ok=True)

    form_rows = []
    for _, row in forms.iterrows():
        reasons = reason_for_form(row)
        if not reasons:
            continue

        writer_id = str(row["writer_id"])
        aligned_path = Path(str(row["aligned_path"]))
        header_path = header_dir / f"{writer_id}_header.png"
        if aligned_path.exists():
            make_header_crop(aligned_path, header_path)

        form_rows.append((row, reasons, header_path if header_path.exists() else None))

    suspicious_lines = manifest_all[
        (manifest_all["line_status"].astype(str) != "filled")
        | (~manifest_all["line_exportable"].astype(bool))
    ].copy()

    line_rows = []
    for _, row in suspicious_lines.iterrows():
        sample_id = str(row["sample_id"])
        src = Path(str(row["flat_image_path"]))
        dst = lines_dir / f"{sample_id}.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        line_rows.append((row, dst if dst.exists() else None))

    html_parts: list[str] = [
        "<!doctype html>",
        "<html lang='pl'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Podglad podejrzanych formularzy</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f6f7f8;color:#111}",
        "h1,h2{margin:0 0 12px}",
        ".summary{margin:0 0 24px;color:#333}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #ddd;border-radius:8px;padding:14px}",
        ".meta{font-size:14px;line-height:1.45;margin-bottom:10px}",
        ".reason{display:inline-block;background:#fff2b8;border:1px solid #e5ca52;padding:2px 6px;border-radius:4px;margin:2px 4px 2px 0}",
        "img.header{width:100%;max-height:430px;object-fit:contain;background:#eee;border:1px solid #ddd}",
        "img.line{width:100%;background:#eee;border:1px solid #ddd}",
        "a{color:#0645ad;text-decoration:none}",
        "table{border-collapse:collapse;width:100%;background:white;margin-bottom:24px}",
        "td,th{border:1px solid #ddd;padding:8px;text-align:left;font-size:14px}",
        "th{background:#e9ecef}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Podglad podejrzanych formularzy</h1>",
        f"<p class='summary'>Formularze do sprawdzenia metadanych: <b>{len(form_rows)}</b>. Linie puste/niepewne: <b>{len(line_rows)}</b>.</p>",
        "<h2>Formularze z podejrzanymi checkboxami</h2>",
        "<div class='grid'>",
    ]

    for row, reasons, header_path in form_rows:
        writer_id = html.escape(str(row["writer_id"]))
        sex = html.escape(str(row.get("sex", "")))
        difficulty = html.escape(str(row.get("difficulty", "")))
        difficulty_group = html.escape(str(row.get("difficulty_group", "")))
        set_id = html.escape(str(row.get("set_id", "")))
        aligned_path = Path(str(row["aligned_path"]))
        birth_crop = Path(str(row["birth_year_crop_path"]))
        other_crop = Path(str(row["other_text_crop_path"]))

        reason_html = " ".join(f"<span class='reason'>{html.escape(r)}</span>" for r in reasons)
        html_parts.extend(
            [
                "<div class='card'>",
                "<div class='meta'>",
                f"<b>{writer_id}</b> | zestaw {set_id}<br>",
                f"plec: <b>{sex}</b> | trudnosci: <b>{difficulty}</b> | grupa: <b>{difficulty_group}</b><br>",
                f"{reason_html}<br>",
                f"<a href='{file_link(aligned_path, output.parent)}'>pelny formularz</a> | ",
                f"<a href='{file_link(birth_crop, output.parent)}'>crop roku</a> | ",
                f"<a href='{file_link(other_crop, output.parent)}'>crop inne</a>",
                "</div>",
            ]
        )
        if header_path is not None:
            html_parts.append(f"<img class='header' src='{file_link(header_path, output.parent)}'>")
        html_parts.append("</div>")

    html_parts.extend(["</div>", "<h2 style='margin-top:28px'>Linie puste albo niepewne</h2>"])

    if line_rows:
        html_parts.append("<table><thead><tr><th>sample_id</th><th>status</th><th>reference</th><th>podglad</th></tr></thead><tbody>")
        for row, line_path in line_rows:
            sample_id = html.escape(str(row["sample_id"]))
            status = html.escape(str(row["line_status"]))
            reference = html.escape(str(row["prompt_reference"]))
            img_html = ""
            if line_path is not None:
                img_html = f"<img class='line' src='{file_link(line_path, output.parent)}'>"
            html_parts.append(
                f"<tr><td>{sample_id}</td><td>{status}</td><td>{reference}</td><td>{img_html}</td></tr>"
            )
        html_parts.append("</tbody></table>")
    else:
        html_parts.append("<p>Brak podejrzanych linii.</p>")

    html_parts.extend(["</body>", "</html>"])
    output.write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HTML gallery for forms/lines requiring manual review.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed_320_check"),
        help="Directory created by process_forms.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <processed-dir>/review_suspicious.html.",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    output = args.output.resolve() if args.output else processed_dir / "review_suspicious.html"
    make_html(processed_dir, output)
    print(output)


if __name__ == "__main__":
    main()
