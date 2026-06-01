# TrOCR-Large Dysgraphia-Oversampled Checkpoint

The thesis experiments used a fine-tuned `microsoft/trocr-large-handwritten` checkpoint.

The model configuration, generation configuration, tokenizer metadata, and `trainer_state.json` are stored here. The final checkpoint file `model.safetensors` is approximately 2.4 GB, so it is intentionally not stored in this GitHub repository. The corresponding training and evaluation code is available in:

- `notebooks/trocr_experiments.ipynb`,
- `src/fine_tuning/trocr/train_trocr.py`,
- `src/fine_tuning/trocr/evaluate_trocr.py`.

The reported metrics and plots generated from this checkpoint are stored under `results/`.
