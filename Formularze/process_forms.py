# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


OUT_W = 2480
OUT_H = 3508
PAGE_W_PT = 595.28
PAGE_H_PT = 841.89

SET_IDS = ["A", "B", "C"]

REFERENCES: dict[str, list[str]] = {
    "A": [
        "Litwo, Ojczyzno moja! Ty jesteś jak zdrowie.",
        "Mężny bądź, chroń pułk twój i sześć flag.",
        "Pchnąć w tę łódź jeża lub ośm skrzyń.",
        "„Bywam, a może nie bywam? Oto jest pytanie!”",
        "Stary zegar w wieży wybił właśnie godzinę dziewiątą.",
        "Miecz. Na plecach. Dlaczego masz na plecach miecz?",
        "Żółw powoli przeszedł przez krętą ścieżkę w wilgotnym lesie.",
        "UWAGA: SYSTEM WYMAGA AKTUALIZACJI.",
    ],
    "B": [
        "W odległości około 10 metrów od burty woda zakotłowała się.",
        "Mieć pół miliona i nie mieć pół miliona — to cały milion.",
        "„Bóg nie gra w kości” — powiedział kiedyś Einstein.",
        "Kontakt: user_01@poczta.pl (tel: +48 123 456 789).",
        "Spotkanie rozpoczęło się o 18:45 w sali nr 3.",
        "Eksperyment rozpoczął się o godzinie 14:35.",
        "Średnia prędkość wyniosła 3,5 m/s.",
        "PROGRAM WYŚWIETLIŁ KOMUNIKAT ERROR #404.",
    ],
    "C": [
        "Robot przejechał dystans 12 metrów w 8 sekund.",
        "ID: A-204; status=OK; temp=23.5°C.",
        "Gracz zapisał stan gry w pliku save_01.dat.",
        "W folderze znajdziesz plik raport_v2.txt.",
        "O 21:30 rozpoczęło się ważne spotkanie zespołu.",
        "W bibliotece znaleziono starą książkę z 1908 roku.",
        "Gęsta mgła powoli zasłoniła światła starego miasta.",
        "ANALIZA WYKAZAŁA SKUTECZNOŚĆ 87%.",
    ],
}

# Coordinates are in PDF points for the supplied LaTeX/PDF layout.
MARKER_CENTERS_PX = np.array(
    [
        [163.5, 172.9],
        [2242.6, 172.9],
        [2242.6, 3273.6],
        [163.5, 3273.6],
    ],
    dtype=np.float32,
)

LINE_PROMPT_Y_PT = [244.9, 310.8, 376.7, 442.7, 508.6, 574.5, 640.5, 706.4]
LINE_CROP_X_PT = (0.0, PAGE_W_PT)
LINE_CROP_TOP_OFFSET_PT = 12.0
LINE_CROP_BOTTOM_OFFSET_PT = 63.5
DOUBLE_LINE_WRITERS = {"212_p01", "227_p01"}
DOUBLE_LINE_TOP_EXTRA_PX = 35
DOUBLE_LINE_GAP_PX = 5
DOUBLE_LINE_LAST_EXTRA_PX = 130

CHECKBOXES_PT = {
    "sex_female": (88.02, 205.30, 98.50, 218.60),
    "sex_male": (138.08, 205.30, 148.60, 218.60),
    "difficulty_none": (332.92, 218.85, 343.50, 232.16),
    "difficulty_dysgraphia": (362.50, 218.85, 373.00, 232.16),
    "difficulty_other": (420.76, 218.85, 431.20, 232.16),
}

BIRTH_YEAR_BOX_PT = (257.0, 199.0, 360.0, 222.0)
OTHER_TEXT_BOX_PT = (452.0, 216.0, 552.0, 236.0)

SET_MATCH_ROIS_PT = [
    (245.0, 112.0, 350.0, 145.0),  # "Zestaw A/B/C"
    (45.0, 240.0, 555.0, 733.0),  # prompt texts
]
SET_SEARCH_ROI_PT = (290.0, 115.0, 350.0, 150.0)
SET_LETTER_TEMPLATE_ROI_PT = (311.0, 122.0, 330.0, 142.0)

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class TemplatePage:
    set_id: str
    image_rgb: np.ndarray
    gray: np.ndarray
    match_mask: np.ndarray
    set_letter_template: np.ndarray


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


def pt_to_px_x(value: float) -> int:
    return int(round(value / PAGE_W_PT * OUT_W))


def pt_to_px_y(value: float) -> int:
    return int(round(value / PAGE_H_PT * OUT_H))


def box_pt_to_px(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return pt_to_px_x(x0), pt_to_px_y(y0), pt_to_px_x(x1), pt_to_px_y(y1)


def clamp_box(box: tuple[int, int, int, int], width: int = OUT_W, height: int = OUT_H) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def read_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(image)


def render_pdf_pages(path: Path, width: int = OUT_W, height: int = OUT_H) -> Iterable[tuple[str, np.ndarray]]:
    doc = fitz.open(path)
    for page_index, page in enumerate(doc, start=1):
        matrix = fitz.Matrix(width / page.rect.width, height / page.rect.height)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        yield f"{path.stem}_p{page_index:02d}", image


def load_inputs(input_path: Path) -> Iterable[tuple[str, Path, np.ndarray]]:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.rglob("*") if p.is_file())

    for path in paths:
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGES:
            yield path.stem, path, read_image_rgb(path)
        elif suffix == ".pdf":
            for page_id, image in render_pdf_pages(path):
                yield page_id, path, image


def render_templates(template_pdf: Path) -> dict[str, TemplatePage]:
    pages: dict[str, TemplatePage] = {}
    for set_id, (_, image_rgb) in zip(SET_IDS, render_pdf_pages(template_pdf)):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        dark = gray < 180
        roi_mask = np.zeros_like(dark, dtype=bool)
        for roi_pt in SET_MATCH_ROIS_PT:
            x0, y0, x1, y1 = box_pt_to_px(roi_pt)
            roi_mask[y0:y1, x0:x1] = True
        match_mask = dark & roi_mask
        lx0, ly0, lx1, ly1 = box_pt_to_px(SET_LETTER_TEMPLATE_ROI_PT)
        letter_crop = gray[ly0:ly1, lx0:lx1]
        set_letter_template = ((letter_crop < 180).astype(np.uint8)) * 255
        pages[set_id] = TemplatePage(
            set_id=set_id,
            image_rgb=image_rgb,
            gray=gray,
            match_mask=match_mask,
            set_letter_template=set_letter_template,
        )
    if len(pages) != 3:
        raise ValueError(f"Expected a 3-page template PDF, got {len(pages)} page(s): {template_pdf}")
    return pages


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def find_document_quad(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small = cv2.resize(image_rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image_rgb.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < mask.size * 0.25:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = small.shape[0] * small.shape[1]
    for contour in contours[:8]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2) / scale)

    rect = cv2.minAreaRect(contours[0])
    return order_points(cv2.boxPoints(rect) / scale)


def warp_to_a4(image_rgb: np.ndarray) -> np.ndarray:
    quad = find_document_quad(image_rgb)
    destination = np.array([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(image_rgb, matrix, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def detect_registration_markers(image_rgb: np.ndarray, radius: int = 520) -> list[tuple[float, float] | None]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    found: list[tuple[float, float] | None] = []
    for expected_x, expected_y in MARKER_CENTERS_PX:
        x0 = max(0, int(expected_x - radius))
        y0 = max(0, int(expected_y - radius))
        x1 = min(OUT_W, int(expected_x + radius))
        y1 = min(OUT_H, int(expected_y + radius))
        crop = gray[y0:y1, x0:x1]
        _, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        components = cv2.connectedComponentsWithStats(threshold, 8)
        count, _, stats, centroids = components
        best: tuple[float, float, float] | None = None
        for idx in range(1, count):
            cx0, cy0, w, h, area = stats[idx]
            if not (7 <= w <= 90 and 7 <= h <= 90 and 35 <= area <= 4500):
                continue
            aspect = w / max(1, h)
            fill = area / max(1, w * h)
            if not (0.55 <= aspect <= 1.45 and fill >= 0.33):
                continue
            cx = x0 + float(centroids[idx][0])
            cy = y0 + float(centroids[idx][1])
            distance = float(np.hypot(cx - expected_x, cy - expected_y))
            score = distance - fill * 25
            if best is None or score < best[0]:
                best = (score, cx, cy)
        found.append((best[1], best[2]) if best else None)
    return found


def refine_with_markers(image_rgb: np.ndarray) -> tuple[np.ndarray, str, int]:
    refined = image_rgb
    method = "page_contour_only"
    used = 0
    for _ in range(2):
        markers = detect_registration_markers(refined)
        src_points = []
        dst_points = []
        for detected, expected in zip(markers, MARKER_CENTERS_PX):
            if detected is None:
                continue
            src_points.append(detected)
            dst_points.append(expected)
        used = len(src_points)
        if used >= 4:
            homography, _ = cv2.findHomography(np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32), 0)
            if homography is None:
                break
            refined = cv2.warpPerspective(refined, homography, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            method = "marker_homography"
        elif used >= 3:
            affine, _ = cv2.estimateAffine2D(np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32))
            if affine is None:
                break
            refined = cv2.warpAffine(refined, affine, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            method = "marker_affine"
        else:
            break
    return refined, method, used


def align_form(image_rgb: np.ndarray) -> tuple[np.ndarray, str, int]:
    initial = warp_to_a4(image_rgb)
    return refine_with_markers(initial)


def classify_set(image_rgb: np.ndarray, templates: dict[str, TemplatePage]) -> tuple[str, float, dict[str, float]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sx0, sy0, sx1, sy1 = box_pt_to_px(SET_SEARCH_ROI_PT)
    search_crop = gray[sy0:sy1, sx0:sx1]
    scores = {}
    for set_id, template in templates.items():
        search_binary = ((search_crop < 150).astype(np.uint8)) * 255
        result = cv2.matchTemplate(search_binary, template.set_letter_template, cv2.TM_CCOEFF_NORMED)
        scores[set_id] = float(result.max())
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_set, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = best_score - second_score
    return best_set, confidence, scores


def checkbox_density(gray: np.ndarray, box_pt: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = clamp_box(box_pt_to_px(box_pt))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    _, threshold = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = threshold.shape
    inner = threshold[int(h * 0.32) : int(h * 0.68), int(w * 0.32) : int(w * 0.68)]
    if inner.size == 0:
        return 0.0
    return float(cv2.countNonZero(inner) / inner.size)


def checkbox_extra_density(
    gray: np.ndarray,
    template_gray: np.ndarray,
    box_pt: tuple[float, float, float, float],
    pad_px: int = 16,
) -> float:
    x0, y0, x1, y1 = clamp_box(box_pt_to_px(box_pt))
    x0, y0, x1, y1 = clamp_box((x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px))
    crop = gray[y0:y1, x0:x1]
    template_crop = template_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    image_dark = crop < 150
    template_dark = ((template_crop < 190).astype(np.uint8)) * 255
    template_dark = cv2.dilate(template_dark, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1) > 0
    extra = image_dark & ~template_dark
    return float(extra.sum() / extra.size)


def crop_box(image_rgb: np.ndarray, box_pt: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = clamp_box(box_pt_to_px(box_pt))
    return image_rgb[y0:y1, x0:x1]


def detect_metadata(
    image_rgb: np.ndarray,
    template_gray: np.ndarray,
    checkbox_threshold: float,
    checkbox_extra_threshold: float,
) -> dict[str, object]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    densities = {name: checkbox_density(gray, box) for name, box in CHECKBOXES_PT.items()}
    extra_densities = {name: checkbox_extra_density(gray, template_gray, box) for name, box in CHECKBOXES_PT.items()}
    selected = {
        name: densities[name] >= checkbox_threshold or extra_densities[name] >= checkbox_extra_threshold
        for name in CHECKBOXES_PT
    }

    sex_values = []
    if selected["sex_female"]:
        sex_values.append("kobieta")
    if selected["sex_male"]:
        sex_values.append("mezczyzna")
    sex = "+".join(sex_values) if sex_values else "unknown"

    difficulty_values = []
    if selected["difficulty_none"]:
        difficulty_values.append("nie")
    if selected["difficulty_dysgraphia"]:
        difficulty_values.append("dysgrafia")
    if selected["difficulty_other"]:
        difficulty_values.append("inne")

    if selected["difficulty_dysgraphia"]:
        difficulty_group = "dysgrafia"
    elif selected["difficulty_other"]:
        difficulty_group = "inne"
    elif selected["difficulty_none"]:
        difficulty_group = "nie"
    else:
        difficulty_group = "unknown"

    return {
        "sex": sex,
        "difficulty": "+".join(difficulty_values) if difficulty_values else "unknown",
        "difficulty_group": difficulty_group,
        "checkbox_sex_female_density": densities["sex_female"],
        "checkbox_sex_male_density": densities["sex_male"],
        "checkbox_difficulty_none_density": densities["difficulty_none"],
        "checkbox_difficulty_dysgraphia_density": densities["difficulty_dysgraphia"],
        "checkbox_difficulty_other_density": densities["difficulty_other"],
        "checkbox_sex_female_extra_density": extra_densities["sex_female"],
        "checkbox_sex_male_extra_density": extra_densities["sex_male"],
        "checkbox_difficulty_none_extra_density": extra_densities["difficulty_none"],
        "checkbox_difficulty_dysgraphia_extra_density": extra_densities["difficulty_dysgraphia"],
        "checkbox_difficulty_other_extra_density": extra_densities["difficulty_other"],
        "sex_conflict": len(sex_values) > 1,
        "difficulty_conflict": len(difficulty_values) > 1,
    }


def base_line_crop_box(line_index: int) -> tuple[int, int, int, int]:
    y_prompt = LINE_PROMPT_Y_PT[line_index]
    next_prompt = LINE_PROMPT_Y_PT[line_index + 1] if line_index + 1 < len(LINE_PROMPT_Y_PT) else y_prompt + 66.0
    y0 = y_prompt + LINE_CROP_TOP_OFFSET_PT
    y1 = min(y_prompt + LINE_CROP_BOTTOM_OFFSET_PT, next_prompt - 2.0)
    return clamp_box((pt_to_px_x(LINE_CROP_X_PT[0]), pt_to_px_y(y0), pt_to_px_x(LINE_CROP_X_PT[1]), pt_to_px_y(y1)))


def line_crop_box(line_index: int, writer_id: str = "") -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = base_line_crop_box(line_index)
    if writer_id not in DOUBLE_LINE_WRITERS:
        return x0, y0, x1, y1

    y0 = max(0, y0 - DOUBLE_LINE_TOP_EXTRA_PX)
    if line_index + 1 < len(LINE_PROMPT_Y_PT):
        _, next_y0, _, _ = base_line_crop_box(line_index + 1)
        y1 = max(y0 + 1, next_y0 - DOUBLE_LINE_GAP_PX)
    else:
        y1 = min(OUT_H, y1 + DOUBLE_LINE_LAST_EXTRA_PX)
    return clamp_box((0, y0, OUT_W, y1))


def clean_line_crop(crop_rgb: np.ndarray, template_crop_rgb: np.ndarray, aggressive: bool = False) -> np.ndarray:
    clean = crop_rgb.copy()
    template_gray = cv2.cvtColor(template_crop_rgb, cv2.COLOR_RGB2GRAY)
    if not aggressive:
        template_dark = (template_gray < 190).astype(np.uint8) * 255
        template_dark = cv2.dilate(template_dark, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        clean[template_dark > 0] = 255
        return clean

    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    # Aggressive mode for exceptional forms with two-line answers: remove more
    # of the printed prompt/guide line, while keeping likely pen strokes.
    printed_text = (template_gray < 190).astype(np.uint8) * 255
    printed_text = cv2.dilate(printed_text, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)), iterations=2) > 0
    template_marks = (template_gray < 250).astype(np.uint8) * 255
    template_marks = cv2.dilate(template_marks, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)), iterations=2) > 0
    blue_ink = (
        (crop_rgb[:, :, 2].astype(np.int16) > crop_rgb[:, :, 0].astype(np.int16) + 12)
        & (crop_rgb[:, :, 2].astype(np.int16) > crop_rgb[:, :, 1].astype(np.int16) + 4)
        & (crop_gray < 245)
    )
    darker_than_template = crop_gray.astype(np.int16) < (template_gray.astype(np.int16) - 28)
    remove_mask = printed_text | (template_marks & ~blue_ink & ~darker_than_template)
    clean[remove_mask] = 255
    return clean


def line_ink_metrics(
    crop_rgb: np.ndarray,
    template_crop_rgb: np.ndarray,
    ink_threshold: int,
) -> dict[str, float | int]:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_crop_rgb, cv2.COLOR_RGB2GRAY)
    image_dark = gray < ink_threshold

    # Remove printed/gray guide-line pixels expected from the blank template.
    template_marks = (template_gray < 230).astype(np.uint8) * 255
    template_marks = cv2.dilate(template_marks, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=1) > 0

    extra_ink = image_dark & ~template_marks
    # Ignore the outer border of the crop, where perspective interpolation can add dark slivers.
    if extra_ink.shape[0] > 20 and extra_ink.shape[1] > 20:
        extra_ink[:5, :] = False
        extra_ink[-5:, :] = False
        extra_ink[:, :5] = False
        extra_ink[:, -5:] = False

    ink_pixels = int(extra_ink.sum())
    total_pixels = int(extra_ink.size)
    if ink_pixels:
        ys, xs = np.where(extra_ink)
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    else:
        bbox_area = 0
    return {
        "line_ink_pixels": ink_pixels,
        "line_ink_ratio": float(ink_pixels / max(1, total_pixels)),
        "line_ink_bbox_area": bbox_area,
        "line_ink_bbox_ratio": float(bbox_area / max(1, total_pixels)),
    }


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def safe_clear_path(path: Path, root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not str(resolved).lower().startswith(str(resolved_root).lower()):
        raise RuntimeError(f"Refusing to remove outside output directory: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def clear_previous_outputs(output_dir: Path) -> None:
    for name in [
        "aligned",
        "lines_flat",
        "metadata_crops",
        "dataset",
        "trocr",
        "qwen",
        "kraken_gt",
        "stats",
        "forms.csv",
        "manifest.csv",
        "manifest_all.csv",
        "manifest.jsonl",
        "errors.csv",
        "birth_year_qwen_predictions.csv",
    ]:
        safe_clear_path(output_dir / name, output_dir)


def load_overrides(path: Path | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if "writer_id" not in df.columns:
        raise ValueError(f"Metadata override CSV must contain writer_id column: {path}")
    return {str(row["writer_id"]): row.dropna().to_dict() for _, row in df.iterrows()}


def apply_overrides(metadata: dict[str, object], writer_id: str, overrides: dict[str, dict[str, object]]) -> dict[str, object]:
    result = dict(metadata)
    if writer_id in overrides:
        for key, value in overrides[writer_id].items():
            if key != "writer_id":
                result[key] = value
    return result


def process_form(
    source_id: str,
    source_path: Path,
    image_rgb: np.ndarray,
    args: argparse.Namespace,
    templates: dict[str, TemplatePage],
    overrides: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    writer_id = slug(source_id)
    aligned, align_method, marker_count = align_form(image_rgb)
    set_id, set_confidence, set_scores = classify_set(aligned, templates)
    metadata = detect_metadata(aligned, templates[set_id].gray, args.checkbox_threshold, args.checkbox_extra_threshold)
    metadata["birth_year"] = ""
    metadata["birth_year_status"] = "crop_saved_needs_review"
    metadata = apply_overrides(metadata, writer_id, overrides)

    if not str(metadata.get("birth_year", "")).strip():
        metadata["birth_year"] = ""
        metadata["birth_year_status"] = "crop_saved_needs_review"
    else:
        metadata["birth_year_status"] = "from_override"

    aligned_path = args.output / "aligned" / f"{writer_id}.png"
    save_rgb(aligned_path, aligned)

    birth_crop_path = args.output / "metadata_crops" / f"{writer_id}_birth_year.png"
    other_crop_path = args.output / "metadata_crops" / f"{writer_id}_other.png"
    save_rgb(birth_crop_path, crop_box(aligned, BIRTH_YEAR_BOX_PT))
    save_rgb(other_crop_path, crop_box(aligned, OTHER_TEXT_BOX_PT))

    needs_review = (
        marker_count < 3
        or set_confidence < args.min_set_confidence
        or bool(metadata.get("sex_conflict"))
        or bool(metadata.get("difficulty_conflict"))
        or metadata.get("sex") == "unknown"
        or metadata.get("difficulty_group") == "unknown"
        or not str(metadata.get("birth_year", "")).strip()
    )

    form_row = {
        "writer_id": writer_id,
        "source_path": str(source_path),
        "aligned_path": str(aligned_path),
        "set_id": set_id,
        "set_confidence": set_confidence,
        "set_score_A": set_scores.get("A", 0.0),
        "set_score_B": set_scores.get("B", 0.0),
        "set_score_C": set_scores.get("C", 0.0),
        "alignment_method": align_method,
        "markers_used": marker_count,
        "birth_year_crop_path": str(birth_crop_path),
        "other_text_crop_path": str(other_crop_path),
        "needs_review": needs_review,
        **metadata,
    }

    line_rows = []
    references = REFERENCES[set_id]
    template = templates[set_id].image_rgb
    for line_index, reference in enumerate(references):
        line_id = line_index + 1
        x0, y0, x1, y1 = line_crop_box(line_index, writer_id=writer_id)
        crop = aligned[y0:y1, x0:x1]
        raw_crop = crop
        template_crop = template[y0:y1, x0:x1]
        ink_metrics = line_ink_metrics(raw_crop, template_crop, args.line_ink_threshold)
        line_has_handwriting = (
            ink_metrics["line_ink_ratio"] >= args.min_line_ink_ratio
            and ink_metrics["line_ink_bbox_ratio"] >= args.min_line_ink_bbox_ratio
        )
        gt_text = reference if line_has_handwriting else ""
        gt_status = "template_auto" if line_has_handwriting else "empty_line"
        if args.clean_lines:
            crop = clean_line_crop(crop, template_crop, aggressive=writer_id in DOUBLE_LINE_WRITERS)
        filename = f"{writer_id}_{set_id}_l{line_id:02d}.png"
        flat_path = args.output / "lines_flat" / filename
        save_rgb(flat_path, crop)
        line_rows.append(
            {
                "sample_id": f"{writer_id}_{set_id}_l{line_id:02d}",
                "writer_id": writer_id,
                "source_path": str(source_path),
                "set_id": set_id,
                "line_id": line_id,
                "prompt_reference": reference,
                "reference": gt_text,
                "gt_text": gt_text,
                "gt_status": gt_status,
                "gt_source": "form_template",
                "flat_image_path": str(flat_path),
                "image_path": str(flat_path),
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1": x1,
                "crop_y1": y1,
                "line_has_handwriting": line_has_handwriting,
                "line_status": "filled" if line_has_handwriting else "empty_or_uncertain",
                "line_exportable": bool(line_has_handwriting or args.include_empty_lines),
                **ink_metrics,
                "needs_review": needs_review,
                **metadata,
            }
        )
    return form_row, line_rows


def assign_splits(forms_df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    split_map: dict[str, str] = {}
    if forms_df.empty:
        return split_map

    forms_df = forms_df.copy()
    forms_df["stratum"] = forms_df["difficulty_group"].astype(str) + "__" + forms_df["sex"].astype(str)
    for _, group in forms_df.groupby("stratum", sort=False):
        writer_ids = list(group["writer_id"].astype(str))
        rng.shuffle(writer_ids)
        n = len(writer_ids)
        if n < 3:
            counts = {"train": n, "val": 0, "test": 0}
        else:
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            if test_ratio > 0 and n_test == 0:
                n_test = 1
            if val_ratio > 0 and n_val == 0 and n - n_test > 1:
                n_val = 1
            n_train = max(1, n - n_val - n_test)
            while n_train + n_val + n_test > n:
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1
                else:
                    n_train -= 1
            counts = {"train": n_train, "val": n_val, "test": n_test}
        cursor = 0
        for split, count in counts.items():
            for writer_id in writer_ids[cursor : cursor + count]:
                split_map[writer_id] = split
            cursor += count
    return split_map


def copy_split_dataset(lines_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for row in lines_df.to_dict("records"):
        split = slug(row.get("split"), "train")
        difficulty = slug(row.get("difficulty_group"))
        sex = slug(row.get("sex"))
        year = slug(row.get("birth_year"), "year_unknown")
        src = Path(row["flat_image_path"])
        dst = output_dir / "dataset" / split / difficulty / sex / f"year_{year}" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row["image_path"] = str(dst)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(lines_df.columns))


def export_manifests(lines_df: pd.DataFrame, forms_df: pd.DataFrame, output_dir: Path, all_lines_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_lines_df.to_csv(output_dir / "manifest_all.csv", index=False, encoding="utf-8-sig")
    lines_df.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    forms_df.to_csv(output_dir / "forms.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in lines_df.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for split, split_df in lines_df.groupby("split", sort=False):
        trocr_dir = output_dir / "trocr"
        trocr_dir.mkdir(parents=True, exist_ok=True)
        split_df[["image_path", "reference"]].rename(columns={"reference": "text"}).to_csv(
            trocr_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
        )

        qwen_dir = output_dir / "qwen"
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
                        "line_id": row["line_id"],
                        "sex": row["sex"],
                        "birth_year": row.get("birth_year", ""),
                        "difficulty_group": row["difficulty_group"],
                        "difficulty": row["difficulty"],
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        kraken_dir = output_dir / "kraken_gt" / str(split)
        kraken_dir.mkdir(parents=True, exist_ok=True)
        for row in split_df.to_dict("records"):
            src = Path(row["image_path"])
            dst_image = kraken_dir / src.name
            shutil.copy2(src, dst_image)
            dst_image.with_suffix(".gt.txt").write_text(str(row["reference"]), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess handwritten form photos into OCR line datasets.")
    parser.add_argument("--input", type=Path, default=Path("raw"), help="Input image/PDF file or directory with filled forms.")
    parser.add_argument("--output", type=Path, default=Path("processed"), help="Output directory.")
    parser.add_argument("--template-pdf", type=Path, default=Path("Formularz.pdf"), help="Blank 3-page form PDF.")
    parser.add_argument("--metadata-overrides", type=Path, default=None, help="Optional CSV with writer_id,birth_year,... overrides.")
    parser.add_argument("--checkbox-threshold", type=float, default=0.30, help="Inner checkbox ink density threshold.")
    parser.add_argument("--checkbox-extra-threshold", type=float, default=0.085, help="Extra ink threshold after subtracting the blank template.")
    parser.add_argument("--min-set-confidence", type=float, default=0.01, help="Mark form for review below this A/B/C score margin.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-lines", action="store_true", help="Try to remove printed guide lines from line crops.")
    parser.add_argument("--include-empty-lines", action="store_true", help="Export empty/uncertain line crops to model datasets too.")
    parser.add_argument("--line-ink-threshold", type=int, default=185, help="Pixel threshold for handwriting detection; lower means stricter.")
    parser.add_argument("--min-line-ink-ratio", type=float, default=0.003, help="Minimum extra ink ratio to treat a line as filled.")
    parser.add_argument("--min-line-ink-bbox-ratio", type=float, default=0.01, help="Minimum extra ink bounding-box ratio to treat a line as filled.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.input = args.input.resolve()
    args.template_pdf = args.template_pdf.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(args.output)

    templates = render_templates(args.template_pdf)
    overrides = load_overrides(args.metadata_overrides.resolve() if args.metadata_overrides else None)

    form_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    inputs = list(load_inputs(args.input))
    if not inputs:
        raise FileNotFoundError(f"No supported images/PDFs found in {args.input}")

    for index, (source_id, source_path, image_rgb) in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] processing {source_path.name} as {source_id}", flush=True)
        try:
            form_row, rows = process_form(source_id, source_path, image_rgb, args, templates, overrides)
            form_rows.append(form_row)
            line_rows.extend(rows)
        except Exception as exc:
            errors.append({"source_id": source_id, "source_path": str(source_path), "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
            if args.stop_on_error:
                raise

    forms_df = pd.DataFrame(form_rows)
    lines_df = pd.DataFrame(line_rows)
    if lines_df.empty:
        raise RuntimeError("No line crops were produced.")

    split_map = assign_splits(forms_df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    forms_df["split"] = forms_df["writer_id"].map(split_map).fillna("train")
    lines_df["split"] = lines_df["writer_id"].map(split_map).fillna("train")
    exported_lines_df = lines_df[lines_df["line_exportable"].astype(bool)].copy()
    exported_lines_df = copy_split_dataset(exported_lines_df, args.output)
    export_manifests(exported_lines_df, forms_df, args.output, lines_df)

    if errors:
        pd.DataFrame(errors).to_csv(args.output / "errors.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Forms: {len(forms_df)}, exported line samples: {len(exported_lines_df)}, all line slots: {len(lines_df)}")
    print(f"Manifest: {args.output / 'manifest.csv'}")
    print(f"All line slots: {args.output / 'manifest_all.csv'}")
    print(f"Review forms: {int(forms_df['needs_review'].sum()) if 'needs_review' in forms_df else 0}")


if __name__ == "__main__":
    main()
