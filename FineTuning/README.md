# Fine-tuning OCR/HTR on Polish Forms

This folder contains the reproducible fine-tuning setup for the Polish Forms handwriting dataset.

## Experiment logic

The fixed evaluation protocol is:

1. Train on `train`.
2. Monitor on `val`.
3. Evaluate once on `test`.
4. Report metrics overall and by `difficulty_group`: `nie`, `dysgrafia`, `inne`.

Two training variants are prepared for every model family:

- `standard`: original train split.
- `dysgraphia_oversampled`: dysgraphia lines are sampled with replacement until they are about 45% of the training rows.

The important comparison is not only overall CER/WER. The key question is whether the `dysgrafia` group improves without damaging the `nie` group too much.

## Prepare data

Run once:

```powershell
& 'C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe' 'C:\Magisterka\FineTuning\prepare_finetuning_data.py'
```

Outputs:

- `FineTuning/data/manifests/*.csv`: shared manifests.
- `FineTuning/data/qwen/*.json`: Qwen-style image conversation data.
- `FineTuning/data/trocr/*.csv`: TrOCR train/eval CSV files.
- `FineTuning/data/kraken/*`: Kraken path-mode line images plus `.gt.txt`.
- `FineTuning/data/dataset_summary.json`: counts and oversampling summary.

For Colab transfer, create a zip:

```powershell
& 'C:\Magisterka\FineTuning\package_colab_dataset.ps1'
```

## Qwen3VL LoRA

Main model: `Qwen/Qwen3-VL-2B-Instruct`.

Standard LoRA:

```bash
python FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py \
  --train-json FineTuning/data/qwen/train.json \
  --val-json FineTuning/data/qwen/val.json \
  --output-dir FineTuning/Qwen3VL_LoRA/outputs/standard \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --gradient-checkpointing
```

Dysgraphia-aware LoRA:

```bash
python FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py \
  --train-json FineTuning/data/qwen/train_dysgraphia_oversampled.json \
  --val-json FineTuning/data/qwen/val.json \
  --output-dir FineTuning/Qwen3VL_LoRA/outputs/dysgraphia_oversampled \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --gradient-checkpointing
```

Evaluate adapter:

```bash
python FineTuning/Qwen3VL_LoRA/evaluate_qwen3vl_lora.py \
  --adapter-path FineTuning/Qwen3VL_LoRA/outputs/standard \
  --run-name qwen3vl_lora_standard
```

## TrOCR

Main classical OCR baseline: `microsoft/trocr-large-handwritten`.

Standard:

```bash
python FineTuning/TrOCR/train_trocr.py \
  --model-id microsoft/trocr-large-handwritten \
  --train-csv FineTuning/data/trocr/train.csv \
  --val-csv FineTuning/data/trocr/val.csv \
  --output-dir FineTuning/TrOCR/outputs/trocr_large_standard
```

Dysgraphia-aware:

```bash
python FineTuning/TrOCR/train_trocr.py \
  --model-id microsoft/trocr-large-handwritten \
  --train-csv FineTuning/data/trocr/train_dysgraphia_oversampled.csv \
  --val-csv FineTuning/data/trocr/val.csv \
  --output-dir FineTuning/TrOCR/outputs/trocr_large_dysgraphia_oversampled
```

Evaluate:

```bash
python FineTuning/TrOCR/evaluate_trocr.py \
  --model-path FineTuning/TrOCR/outputs/trocr_large_standard \
  --run-name trocr_large_standard
```

## Kraken

The Kraken data uses `path` mode: every image file has a matching `.gt.txt` file with the same prefix.

Standard:

```powershell
& 'C:\Magisterka\FineTuning\Kraken\train_kraken_standard.ps1'
```

Dysgraphia-aware:

```powershell
& 'C:\Magisterka\FineTuning\Kraken\train_kraken_dysgraphia_oversampled.ps1'
```

Evaluate:

```powershell
& 'C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe' 'C:\Magisterka\FineTuning\Kraken\evaluate_kraken.py' `
  --model-path 'C:\Magisterka\FineTuning\Kraken\outputs\standard\best.mlmodel' `
  --run-name kraken_standard
```

The exact checkpoint filename may differ; use the best validation checkpoint produced by `ketos`.

## Recommended first run order

1. Prepare data.
2. Run a smoke test with `--train-limit 16 --val-limit 8` for Qwen and TrOCR.
3. Train `Qwen3VL_LoRA/standard`.
4. Train `Qwen3VL_LoRA/dysgraphia_oversampled`.
5. Train `TrOCR/standard` and `TrOCR/dysgraphia_oversampled`.
6. Train Kraken variants.
7. Compare test metrics overall and by difficulty group.
