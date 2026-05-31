# Qwen3VL LoRA

This path is the primary fine-tuning experiment because Qwen3VL had the best zero-shot results.

Use `FineTuning/data/qwen/train.json` for the standard run and `train_dysgraphia_oversampled.json` for the dysgraphia-aware run.

On Colab, upload or mount the whole `FineTuning/data` folder, install:

```bash
pip install -r FineTuning/requirements_colab.txt
```

Then run `train_qwen3vl_lora.py`. For a large GPU, start with:

```bash
python FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --gradient-checkpointing
```

For dysgraphia-aware training switch `--train-json` to:

```text
FineTuning/data/qwen/train_dysgraphia_oversampled.json
```

The expected output is a PEFT LoRA adapter, not a full merged model.
