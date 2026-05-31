# Malaysian external validation for Qwen LoRA

Use this after training Polish Forms Qwen3VL LoRA adapters.

The package contains corrected Malaysian manual line crops in Qwen evaluation format:

- `FineTuning/malaysian_external_qwen/qwen/malaysian_auto_invert.json`
- `FineTuning/malaysian_external_qwen/qwen/malaysian_raw.json`
- `FineTuning/malaysian_external_qwen/qwen/malaysian_both_variants.json`

The images are already saved in the selected preprocessing variant. Use the usual evaluator with `--test-json`.

## Colab setup

Upload `malaysian_qwen_external_validation_colab.zip` to:

```text
MyDrive/Magisterka/
```

Then:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!unzip -o -q /content/drive/MyDrive/Magisterka/malaysian_qwen_external_validation_colab.zip -d /content
%cd /content
!pip -q install "transformers>=5.0.0" accelerate datasets peft qwen-vl-utils jiwer openpyxl
```

## Evaluate Polish standard adapter on Malaysian

Auto-invert variant:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_standard \
  --test-json /content/FineTuning/malaysian_external_qwen/qwen/malaysian_auto_invert.json \
  --data-root /content/FineTuning/malaysian_external_qwen \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_standard_malaysian_auto_invert \
  --run-name qwen3vl_lora_standard_malaysian_auto_invert
```

Raw variant:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_standard \
  --test-json /content/FineTuning/malaysian_external_qwen/qwen/malaysian_raw.json \
  --data-root /content/FineTuning/malaysian_external_qwen \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_standard_malaysian_raw \
  --run-name qwen3vl_lora_standard_malaysian_raw
```

## Evaluate Polish dysgraphia-oversampled adapter on Malaysian

Auto-invert:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_dysgraphia_oversampled \
  --test-json /content/FineTuning/malaysian_external_qwen/qwen/malaysian_auto_invert.json \
  --data-root /content/FineTuning/malaysian_external_qwen \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_dysgraphia_oversampled_malaysian_auto_invert \
  --run-name qwen3vl_lora_dysgraphia_oversampled_malaysian_auto_invert
```

Raw:

```bash
!python /content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path /content/drive/MyDrive/Magisterka/outputs/qwen3vl_lora_dysgraphia_oversampled \
  --test-json /content/FineTuning/malaysian_external_qwen/qwen/malaysian_raw.json \
  --data-root /content/FineTuning/malaysian_external_qwen \
  --output-dir /content/drive/MyDrive/Magisterka/evaluation/qwen3vl_lora_dysgraphia_oversampled_malaysian_raw \
  --run-name qwen3vl_lora_dysgraphia_oversampled_malaysian_raw
```

## Summary table

```python
import glob
import pandas as pd

rows = []
for path in glob.glob('/content/drive/MyDrive/Magisterka/evaluation/*malaysian*/*_summary.csv'):
    df = pd.read_csv(path)
    df['path'] = path
    rows.append(df)

summary = pd.concat(rows, ignore_index=True)
summary[['model_id', 'dataset', 'samples', 'corpus_cer', 'corpus_wer', 'corpus_cla', 'corpus_crw', 'inference_seconds_mean']]
```
