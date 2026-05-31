# Qwen mixed-domain LoRA

This is the next Qwen experiment after Polish-only LoRA.

Goal:

- keep the strong Polish Forms result,
- improve cross-domain generalization on Malaysian,
- avoid the stronger specialization observed in the 45% dysgraphia oversampling run.

Training data:

- Polish Forms train: 1844 lines.
- Malaysian train split, split by `group + image_number` to avoid line leakage.
- Malaysian train uses both `raw` and `auto_invert` variants as image-domain augmentation.

Validation data:

- Polish Forms val.
- Malaysian val, both variants.

Test data:

- Polish Forms test.
- Malaysian held-out test raw.
- Malaysian held-out test auto-invert.

## Colab setup

Upload:

```text
qwen_mixed_domain_colab.zip
```

to:

```text
MyDrive/Magisterka/
```

Then in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!rm -rf /content/FineTuning /content/ocr_benchmark_utils.py
!unzip -q /content/drive/MyDrive/Magisterka/qwen_mixed_domain_colab.zip -d /content
%cd /content
!pip -q install "transformers>=5.0.0" accelerate datasets peft qwen-vl-utils jiwer openpyxl "pandas==2.2.2" "pillow<12"
```

Check data:

```python
import json
from pathlib import Path
root = Path('/content/FineTuning/qwen_mixed_domain/qwen')
for name in ['mixed_train.json', 'mixed_val.json', 'polish_test.json', 'malaysian_test_raw.json', 'malaysian_test_auto_invert.json']:
    data = json.loads((root / name).read_text(encoding='utf-8'))
    print(name, len(data), data[0]['image'])
```

Smoke test:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py \
  --train-json /content/FineTuning/qwen_mixed_domain/qwen/mixed_train.json \
  --val-json /content/FineTuning/qwen_mixed_domain/qwen/mixed_val.json \
  --data-root /content/FineTuning/qwen_mixed_domain \
  --train-limit 2 \
  --val-limit 2 \
  --dry-run
```

## Training

Recommended first run:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py \
  --train-json /content/FineTuning/qwen_mixed_domain/qwen/mixed_train.json \
  --val-json /content/FineTuning/qwen_mixed_domain/qwen/mixed_val.json \
  --data-root /content/FineTuning/qwen_mixed_domain \
  --output-dir /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_mixed_domain \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 1e-4 \
  --lora-dropout 0.1 \
  --gradient-checkpointing
```

Why this configuration:

- `2` epochs instead of `3` reduces over-specialization.
- `1e-4` learning rate is more conservative than the Polish-only run.
- `lora-dropout 0.1` should help generalization.

## Evaluation

Polish test:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_mixed_domain \
  --test-json /content/FineTuning/qwen_mixed_domain/qwen/polish_test.json \
  --data-root /content/FineTuning/qwen_mixed_domain \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_mixed_domain_polish \
  --run-name qwen3vl_lora_mixed_domain_polish
```

Malaysian held-out raw:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_mixed_domain \
  --test-json /content/FineTuning/qwen_mixed_domain/qwen/malaysian_test_raw.json \
  --data-root /content/FineTuning/qwen_mixed_domain \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_mixed_domain_malaysian_raw \
  --run-name qwen3vl_lora_mixed_domain_malaysian_raw
```

Malaysian held-out auto-invert:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_mixed_domain \
  --test-json /content/FineTuning/qwen_mixed_domain/qwen/malaysian_test_auto_invert.json \
  --data-root /content/FineTuning/qwen_mixed_domain \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_mixed_domain_malaysian_auto_invert \
  --run-name qwen3vl_lora_mixed_domain_malaysian_auto_invert
```

Summary:

```python
import glob
import pandas as pd

rows = []
for path in glob.glob('/content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_mixed_domain*/*_summary.csv'):
    df = pd.read_csv(path)
    df['path'] = path
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary[['model_id', 'dataset', 'preprocessing_variant', 'samples', 'corpus_cer', 'corpus_wer', 'corpus_cla', 'corpus_crw']]
```
