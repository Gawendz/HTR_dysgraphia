# Qwen IAM final check

Purpose:

- final sanity check for Qwen3VL after LoRA fine-tuning,
- use one fixed IAM test subset for every Qwen variant,
- keep this as a supporting experiment, not the main thesis result.

Recommended subset:

- `Teklia/IAM-line`
- split: `test`
- samples: `200`

The dataset card reports an image-to-text handwritten dataset with train, validation, and test splits. The test split has about 2.92k rows, so a deterministic first-200 subset is enough for a compact sanity check.

## 1. Prepare IAM JSON in Colab

Run after unpacking `qwen_mixed_domain_colab.zip`, because it already contains the Qwen evaluation script.

```bash
!pip -q install datasets
```

```bash
!python /content/FineTuning/prepare_qwen_iam_eval.py \
  --dataset-id Teklia/IAM-line \
  --split test \
  --limit 200 \
  --output-root /content/FineTuning/qwen_iam_eval
```

If `prepare_qwen_iam_eval.py` is not present in `/content/FineTuning`, paste this local file into Colab or upload a refreshed zip.

Expected JSON:

```text
/content/FineTuning/qwen_iam_eval/qwen/iam_test_200.json
```

## 2. Evaluate Qwen variants

```python
from pathlib import Path
import subprocess
import sys

DRIVE = Path('/content/drive/MyDrive/Magisterka')
DATA_ROOT = Path('/content/FineTuning/qwen_iam_eval')
TEST_JSON = DATA_ROOT / 'qwen/iam_test_200.json'
EVAL_SCRIPT = Path('/content/FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py')

adapters = {
    'qwen3vl_base_iam_200': None,
    'qwen3vl_lora_standard_iam_200': DRIVE / 'outputs/qwen3vl_lora_standard',
    'qwen3vl_lora_dysgraphia_oversampled_iam_200': DRIVE / 'outputs/qwen3vl_lora_dysgraphia_oversampled',
    'qwen3vl_lora_mixed_domain_iam_200': DRIVE / 'outputs/qwen3vl_lora_mixed_domain',
}

for run_name, adapter_path in adapters.items():
    output_dir = DRIVE / 'evaluation' / run_name
    summary_path = output_dir / f'{run_name}_summary.csv'
    if summary_path.exists():
        print(f'Już istnieje, pomijam: {run_name}')
        continue

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        '--test-json', str(TEST_JSON),
        '--data-root', str(DATA_ROOT),
        '--output-dir', str(output_dir),
        '--run-name', run_name,
    ]
    if adapter_path is not None:
        if not adapter_path.exists():
            print(f'Pomijam, nie znaleziono adaptera: {adapter_path}')
            continue
        cmd.extend(['--adapter-path', str(adapter_path)])

    print('Uruchamiam:', run_name)
    subprocess.run(cmd, check=True)
```

## 3. Summary table and plot

```python
import pandas as pd
import matplotlib.pyplot as plt

summary_paths = [
    p for p in (DRIVE / 'evaluation').glob('qwen3vl*_iam_200/*_summary.csv')
    if not p.name.endswith('_group_summary.csv')
]

rows = []
for path in summary_paths:
    df = pd.read_csv(path)
    df['run_dir'] = path.parent.name
    rows.append(df)

iam = pd.concat(rows, ignore_index=True)
iam = iam[iam['dataset'].astype(str).str.startswith('iam_')].copy()
iam = iam.drop_duplicates(subset=['run_dir', 'dataset', 'corpus_cer', 'corpus_wer'])

cols = [
    'run_dir',
    'samples',
    'corpus_cer',
    'corpus_wer',
    'corpus_cla',
    'corpus_crw',
    'inference_seconds_mean',
    'cuda_peak_allocated_mb_max',
    'cuda_peak_reserved_mb_max',
]

display(iam[cols].sort_values('corpus_cer'))

plot_df = iam[cols].sort_values('corpus_cer').copy()
plot_df['label'] = plot_df['run_dir'].str.replace('qwen3vl_', '', regex=False).str.replace('_iam_200', '', regex=False)

ax = plot_df.plot(
    x='label',
    y=['corpus_cer', 'corpus_wer'],
    kind='bar',
    figsize=(10, 5),
    rot=25,
)
ax.set_title('Qwen3VL IAM Test Subset CER/WER')
ax.set_xlabel('')
ax.set_ylabel('Error rate')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out_png = DRIVE / 'evaluation/qwen3vl_iam_200_comparison.png'
plt.savefig(out_png, dpi=220)
plt.show()
print('Zapisano:', out_png)
```
