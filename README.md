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

## Quick Start

Clone the repository and install the lightweight Python dependencies:

```bash
git clone https://github.com/Gawendz/HTR_dysgraphia.git
cd HTR_dysgraphia
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The repository is organized as a reproducibility package for the thesis. The
notebooks can be opened locally in Jupyter/VS Code or uploaded to Google Colab.
Large model weights that are not tracked in Git should be downloaded separately
and unpacked into the matching folder under `models/`.

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

Large checkpoint files are not stored directly in the repository. In particular, the
fine-tuned TrOCR `model.safetensors` file is kept externally because it is a
multi-gigabyte artifact. If an external archive with weights is provided, it should
be unpacked so that its folders match the structure under `models/`.

The external model-weight archive used for the thesis is available on Hugging Face:

https://huggingface.co/Gawendz/HTR_dysgraphia_model_weights

It can be downloaded with:

```bash
pip install -U huggingface_hub hf_xet
hf download Gawendz/HTR_dysgraphia_model_weights --local-dir models_external
```

After downloading, copy the selected folders from `models_external/` into
`models/` if you want to reproduce the local layout used by the notebooks.

## Notebooks

The `notebooks/` directory contains re-runnable report notebooks for the three
model families. They include the key commands, saved output tables, and embedded
PNG figures, so they can be inspected without rerunning the experiments. In a
forked repository, the same notebooks can be rerun after replacing or adding
result files. Helper functions in the notebooks save custom outputs under
`results/custom_tables/` and `results/custom_figures/`.

The reusable training and evaluation scripts are stored under `src/`.

If GitHub's notebook preview is temporarily unavailable, download the `.ipynb`
file and open it in Jupyter, VS Code, or Google Colab.

Notebook overview:

- `qwen3vl_experiments.ipynb`: summarizes Qwen3-VL zero-shot inference, LoRA
  fine-tuning, dysgraphia oversampling, mixed-domain evaluation, and final
  comparison plots.
- `trocr_experiments.ipynb`: summarizes TrOCR-large zero-shot evaluation,
  standard fine-tuning, dysgraphia-oversampled fine-tuning, soft fine-tuning,
  and external-domain checks.
- `kraken_experiments.ipynb`: summarizes Kraken/CRNN zero-shot evaluation,
  fine-tuning with Kraken, dysgraphia oversampling, and result aggregation.

Typical workflow after cloning:

1. Open one of the notebooks from `notebooks/`.
2. Run the setup cells to load CSV summaries from `results/tables/`.
3. Inspect the saved result tables and figures.
4. For custom experiments, write new outputs to `results/custom_tables/` and
   `results/custom_figures/` instead of overwriting the thesis outputs.

## Metrics

The reported metrics are:

- corpus CER,
- corpus WER,
- corpus CLA,
- corpus CRW,
- inference time,
- memory usage.
