# Models

This directory contains selected model artifacts used in the thesis experiments.

Included artifacts:

- `qwen3vl_lora_dysgraphia_oversampled/`: Qwen3-VL LoRA adapter, tokenizer metadata, and chat template from the dysgraphia-oversampled fine-tuning run.
- `kraken_crnn_dysgraphia_oversampled/`: Kraken/CRNN fine-tuned checkpoint from the dysgraphia-oversampled run.
- `kraken_crnn_standard/`: Kraken/CRNN fine-tuned checkpoint from the standard run.
- `trocr_large_dysgraphia_oversampled/`: TrOCR checkpoint configuration, tokenizer metadata, trainer state, and a note about the external weight file.

The full TrOCR fine-tuned checkpoint is not included because its `model.safetensors` file is approximately 2.4 GB. This exceeds practical GitHub repository limits and should be stored externally, for example on Dropbox, Google Drive, institutional storage, Zenodo, or Hugging Face Hub. If an external weights archive is provided, unpack it with the same directory layout as `models/`.
