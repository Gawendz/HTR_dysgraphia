from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TrOCRProcessor, VisionEncoderDecoderModel


DRIVE = Path("/content/drive/MyDrive/Magisterka")
WORK = Path("/content/Chapter3_excluded_probe")
MANIFEST = WORK / "excluded_probe_manifest.json"
OUTPUT = DRIVE / "evaluation" / "excluded_probe_all_models"

QWEN_BASE = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_POLISH_ADAPTER = DRIVE / "outputs" / "qwen3vl_lora_dysgraphia_oversampled"
QWEN_MIXED_ADAPTER = DRIVE / "outputs" / "qwen3vl_lora_mixed_domain"
TROCR_MODEL = DRIVE / "outputs" / "trocr_large_dysgraphia_oversampled"

PROMPT = "Transcribe exactly the handwritten text in this image. Return only the transcription, with no explanation."


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def clean(text: str) -> str:
    text = str(text).strip()
    for prefix in ("Transcription:", "transcription:", "Text:", "text:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        text = text[1:-1].strip()
    return " ".join(text.split())


def predict_qwen(records: list[dict]) -> list[dict]:
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(QWEN_BASE, min_pixels=3136, max_pixels=524288)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_BASE,
        dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")

    rows: list[dict] = []
    adapters = {
        "Qwen3-VL LoRA + oversampling": QWEN_POLISH_ADAPTER,
        "Qwen3-VL mixed-domain LoRA": QWEN_MIXED_ADAPTER,
    }
    for model_name, adapter in adapters.items():
        model = PeftModel.from_pretrained(base, adapter).eval()
        for item in records:
            image = Image.open(WORK / item["image"]).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "min_pixels": 3136, "max_pixels": 524288},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            prediction = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            rows.append({
                **item,
                "model": model_name,
                "prediction": clean(prediction),
            })
        del model
        torch.cuda.empty_cache()
    del base
    torch.cuda.empty_cache()
    return rows


def predict_trocr(records: list[dict]) -> list[dict]:
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).eval().to("cuda")
    rows: list[dict] = []
    for item in records:
        image = Image.open(WORK / item["image"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.inference_mode():
            ids = model.generate(pixel_values, max_new_tokens=96, num_beams=4)
        prediction = processor.batch_decode(ids, skip_special_tokens=True)[0]
        rows.append({
            **item,
            "model": "TrOCR-large full FT + oversampling",
            "prediction": clean(prediction),
        })
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_manifest()
    rows = []
    rows.extend(predict_qwen(records))
    rows.extend(predict_trocr(records))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "excluded_probe_qwen_trocr_predictions.csv", index=False, encoding="utf-8")
    display(df[["sample_id", "source_dataset", "display_reference", "model", "prediction"]])
    print(OUTPUT / "excluded_probe_qwen_trocr_predictions.csv")


if __name__ == "__main__":
    main()
