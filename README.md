# HTR Dysgraphia

The repository contains the datasets/manifests, model artifacts, notebooks, source code, and result files used in the experiments. The work compares three model families:

- **Qwen3-VL**: multimodal vision-language model evaluated zero-shot and fine-tuned with LoRA,
- **TrOCR**: transformer OCR/HTR encoder-decoder model,
- **Kraken/CRNN**: classical HTR pipeline based on line recognition and CTC decoding.

## Repository Structure

```text
datasets/
  polish_forms/       Polish handwritten forms dataset manifests and statistics
  malaysian/          Malaysian line-level annotations and manifests
  iam/                Notes for the IAM-200 control subset

notebooks/
  qwen3vl_experiments.ipynb
  trocr_experiments.ipynb
  kraken_experiments.ipynb

src/
  common/             Shared OCR evaluation utilities
  zero_shot/          Zero-shot inference scripts
  fine_tuning/        Fine-tuning and evaluation scripts

models/
  qwen3vl_lora_dysgraphia_oversampled/
  kraken_crnn_dysgraphia_oversampled/
  kraken_crnn_standard/
  trocr_large_dysgraphia_oversampled/

results/
  figures/            Final plots used in the thesis
  tables/             Final summary tables
  zero_shot/          Main zero-shot outputs
  fine_tuning/        Main fine-tuning outputs
```

## Datasets

The experiments used:

- a custom Polish handwriting dataset prepared from form scans,
- a Malaysian handwriting dataset with LPD/PD groups,
- the IAM line dataset as an external HTR control benchmark.

The repository includes manifests, annotations, and derived statistics needed to inspect the experimental splits and reproduce the evaluation pipeline. Large/private raw scan folders and temporary preprocessing workspaces are intentionally not tracked.

## Models

The repository includes model artifacts that are small enough for GitHub/LFS:

- Qwen3-VL LoRA adapter for the dysgraphia-oversampled experiment,
- Kraken/CRNN fine-tuned checkpoints for standard and dysgraphia-oversampled variants.

## Notebooks

The `notebooks/` directory contains the cleaned experimental notebooks for the three model families. They are kept as the main readable record of how the experiments were run.

## Metrics

The reported metrics are:

- corpus CER,
- corpus WER,
- corpus CLA,
- corpus CRW,
- inference time,
- memory usage.
