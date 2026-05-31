from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="Teklia/IAM-line")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output-root", default="/content/FineTuning/qwen_iam_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    images_dir = output_root / "images"
    qwen_dir = output_root / "qwen"
    images_dir.mkdir(parents=True, exist_ok=True)
    qwen_dir.mkdir(parents=True, exist_ok=True)

    split = f"{args.split}[:{args.limit}]" if args.limit else args.split
    dataset = load_dataset(args.dataset_id, split=split)

    records = []
    manifest = []
    for idx, item in enumerate(dataset):
        sample_id = f"iam_{args.split}_{idx:05d}"
        image_name = f"{sample_id}.png"
        image_path = images_dir / image_name
        image = item["image"].convert("RGB")
        image.save(image_path)
        text = str(item["text"])

        records.append(
            {
                "id": sample_id,
                "image": f"images/{image_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{DEFAULT_PROMPT}"},
                    {"from": "gpt", "value": text},
                ],
                "metadata": {
                    "dataset": f"iam_{args.split}_{len(dataset)}",
                    "source_dataset": "iam",
                    "preprocessing_variant": "native",
                    "difficulty_group": "IAM-test",
                    "iam_split": args.split,
                    "iam_index": idx,
                    "reference_source": f"{args.dataset_id} {args.split} split",
                },
            }
        )
        manifest.append({"sample_id": sample_id, "image": f"images/{image_name}", "reference": text})

    json_path = qwen_dir / f"iam_{args.split}_{len(records)}.json"
    manifest_path = output_root / f"iam_{args.split}_{len(records)}_manifest.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "limit": args.limit,
        "samples": len(records),
        "json_path": str(json_path),
        "images_dir": str(images_dir),
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
