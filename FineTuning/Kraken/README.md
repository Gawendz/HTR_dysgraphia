# Kraken Fine-tuning

Kraken uses path-mode training data:

```text
image.png
image.gt.txt
```

The preparation script creates this automatically under:

- `FineTuning/data/kraken/train`
- `FineTuning/data/kraken/val`
- `FineTuning/data/kraken/test`
- `FineTuning/data/kraken/train_dysgraphia_oversampled`

Run:

```powershell
& 'C:\Magisterka\FineTuning\Kraken\train_kraken_standard.ps1'
```

For dysgraphia-aware training:

```powershell
& 'C:\Magisterka\FineTuning\Kraken\train_kraken_dysgraphia_oversampled.ps1'
```

By default the scripts locate the same Kraken baseline used in zero-shot and continue from it. You can override it with `-BaseModel`.
