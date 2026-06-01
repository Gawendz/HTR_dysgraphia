from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "FineTuning" / "data"


class LineOCRDataset(Dataset):
    def __init__(self, csv_path: Path, data_root: Path, processor: Any, max_target_length: int = 128, limit: int | None = None):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_target_length = max_target_length
        df = pd.read_csv(self.csv_path)
        self.df = df.head(limit).copy() if limit else df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_relpath"])
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            str(row["text"]),
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [token if token != self.processor.tokenizer.pad_token_id else -100 for token in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels), "sample_id": row["sample_id"]}


@dataclass
class OCRCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/trocr-large-handwritten")
    parser.add_argument("--train-csv", default=str(DATA_ROOT / "trocr" / "train.csv"))
    parser.add_argument("--val-csv", default=str(DATA_ROOT / "trocr" / "val.csv"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "FineTuning" / "TrOCR" / "outputs" / "trocr_large_standard"))
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load processor and one batch, then exit without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import TrOCRProcessor

    processor = TrOCRProcessor.from_pretrained(args.model_id)
    train_dataset = LineOCRDataset(Path(args.train_csv), Path(args.data_root), processor, args.max_target_length, args.train_limit)
    val_dataset = LineOCRDataset(Path(args.val_csv), Path(args.data_root), processor, args.max_target_length, args.val_limit)
    if args.dry_run:
        batch = OCRCollator()([train_dataset[i] for i in range(min(2, len(train_dataset)))])
        print({key: tuple(value.shape) for key, value in batch.items()})
        print(f"train={len(train_dataset)} val={len(val_dataset)}")
        return

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, VisionEncoderDecoderModel

    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = args.max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=OCRCollator(),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
