from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "FineTuning" / "data"
DEFAULT_PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


class QwenOCRDataset(Dataset):
    def __init__(self, json_path: Path, data_root: Path, limit: int | None = None):
        self.json_path = Path(json_path)
        self.data_root = Path(data_root)
        records = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.records = records[:limit] if limit else records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        text = item["conversations"][1]["value"]
        return {
            "id": item["id"],
            "image": Image.open(image_path).convert("RGB"),
            "text": text,
            "image_path": str(image_path),
            "metadata": item.get("metadata", {}),
        }


@dataclass
class QwenOCRCollator:
    processor: Any
    prompt: str
    max_pixels: int
    min_pixels: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        images = []
        for item in batch:
            image = item["image"]
            images.append(image)
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": [{"type": "text", "text": item["text"]}]},
            ]
            prompt_texts.append(self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False))

        full_inputs = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        labels = full_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row_idx, prompt_len in enumerate(prompt_inputs["attention_mask"].sum(dim=1).tolist()):
            labels[row_idx, : int(prompt_len)] = -100
        full_inputs["labels"] = labels
        return full_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-json", default=str(DATA_ROOT / "qwen" / "train.json"))
    parser.add_argument("--val-json", default=str(DATA_ROOT / "qwen" / "val.json"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "FineTuning" / "Qwen3VL_LoRA" / "outputs" / "standard"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Load processor and one batch, then exit without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoProcessor

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    train_dataset = QwenOCRDataset(Path(args.train_json), Path(args.data_root), args.train_limit)
    val_dataset = QwenOCRDataset(Path(args.val_json), Path(args.data_root), args.val_limit)
    collator = QwenOCRCollator(processor=processor, prompt=args.prompt, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    if args.dry_run:
        batch = collator([train_dataset[i] for i in range(min(2, len(train_dataset)))])
        print({key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")})
        print(f"train={len(train_dataset)} val={len(val_dataset)}")
        return

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, dtype=dtype, device_map="auto")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    max_steps = -1
    if args.epochs <= 0:
        max_steps = math.ceil(len(train_dataset) / max(1, args.batch_size * args.grad_accum))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
