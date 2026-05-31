# Oddzielne paczki Colab

Masz dwie osobne paczki:

- `polish_forms_qwen3vl_lora_colab.zip` - Qwen3-VL LoRA.
- `polish_forms_trocr_colab.zip` - TrOCR fine-tuning.

Obie paczki zawierają folder:

```text
FineTuning/
ocr_benchmark_utils.py
```

W Colabie wrzuć wybraną paczkę do:

```text
MyDrive/Magisterka/
```

Potem w notebooku:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Qwen:

```bash
!rm -rf /content/FineTuning /content/ocr_benchmark_utils.py
!unzip -q /content/drive/MyDrive/Magisterka/polish_forms_qwen3vl_lora_colab.zip -d /content
%cd /content
!pip -q install -r /content/FineTuning/requirements_colab.txt
!python /content/FineTuning/Qwen3VL_LoRA/train_qwen3vl_lora.py --train-limit 2 --val-limit 2 --dry-run
```

TrOCR:

```bash
!rm -rf /content/FineTuning /content/ocr_benchmark_utils.py
!unzip -q /content/drive/MyDrive/Magisterka/polish_forms_trocr_colab.zip -d /content
%cd /content
!pip -q install -r /content/FineTuning/requirements_colab.txt
!python /content/FineTuning/TrOCR/train_trocr.py --train-limit 2 --val-limit 2 --dry-run
```

Najpierw uruchom smoke test, dopiero potem pełny trening.
