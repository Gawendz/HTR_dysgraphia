from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "FineTuning" / "data" / "kraken"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "FineTuning" / "data" / "kraken_pagexml"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def read_manifest(split_dir: Path) -> list[Path]:
    manifest = split_dir / "manifest.txt"
    return [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def page_xml_for_line(image_path: Path, text: str) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
    # Use the whole line image as one text line. A baseline through the vertical
    # center is sufficient for Kraken to extract the already-cropped line strip.
    right = max(0, width - 1)
    bottom = max(0, height - 1)
    baseline_y = max(0, height // 2)
    points = f"0,0 {right},0 {right},{bottom} 0,{bottom}"
    baseline = f"0,{baseline_y} {right},{baseline_y}"
    image_name = escape(str(image_path))
    text_xml = escape(text)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{PAGE_NS}">
  <Page imageFilename="{image_name}" imageWidth="{width}" imageHeight="{height}">
    <TextRegion id="r1">
      <Coords points="{points}"/>
      <TextLine id="l1">
        <Coords points="{points}"/>
        <Baseline points="{baseline}"/>
        <TextEquiv>
          <Unicode>{text_xml}</Unicode>
        </TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def convert_split(source_root: Path, output_root: Path, split: str) -> dict[str, int]:
    source_dir = source_root / split
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_paths: list[str] = []
    for image_path in read_manifest(source_dir):
        gt_path = image_path.with_suffix(".gt.txt")
        text = gt_path.read_text(encoding="utf-8").strip()
        xml_path = out_dir / f"{image_path.stem}.xml"
        xml_path.write_text(page_xml_for_line(image_path, text), encoding="utf-8")
        xml_paths.append(str(xml_path))
    (out_dir / "manifest.txt").write_text("\n".join(xml_paths) + "\n", encoding="utf-8")
    return {"split": split, "records": len(xml_paths)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test", "train_dysgraphia_oversampled"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    summary = [convert_split(data_root, output_root, split) for split in args.splits]
    for item in summary:
        print(f"{item['split']}: {item['records']}")
    print(output_root)


if __name__ == "__main__":
    main()
