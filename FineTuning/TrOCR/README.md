# TrOCR Fine-tuning

Main model: `microsoft/trocr-large-handwritten`.

The script uses line-level images and exact text from:

- `FineTuning/data/trocr/train.csv`
- `FineTuning/data/trocr/val.csv`
- `FineTuning/data/trocr/test.csv`

For the dysgraphia-aware experiment, use:

```text
FineTuning/data/trocr/train_dysgraphia_oversampled.csv
```

Start with a smoke test:

```powershell
& 'C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe' 'C:\Magisterka\FineTuning\TrOCR\train_trocr.py' `
  --train-limit 16 `
  --val-limit 8 `
  --output-dir 'C:\Magisterka\FineTuning\TrOCR\outputs\smoke'
```

Then run the full experiment on Colab or another stronger GPU.
