# Excluded Probe: Qwen + TrOCR

Ten folder zawiera małą próbę jakościową dla próbek wyłączonych z głównej ewaluacji.

Zawartość:

- `excluded_probe_manifest.json` - 4 próbki: 3 z własnego zbioru i 1 z Malaysian PD.
- `images/` - obrazy używane w próbie.
- `run_excluded_probe_colab.py` - skrypt do odpalenia w Google Colab na tych samych checkpointach Qwen i TrOCR, które były używane w pracy.

Kroki w Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Wrzuć folder `colab_excluded_probe` do `/content/Chapter3_excluded_probe`, np. przez upload zipa albo Drive.

Zainstaluj zależności jak przy wcześniejszym Qwen/TrOCR:

```bash
pip install -q "transformers>=5.0.0" peft qwen-vl-utils pillow pandas accelerate
```

Uruchom:

```bash
python /content/Chapter3_excluded_probe/run_excluded_probe_colab.py
```

Wynik zapisze się do:

```text
/content/drive/MyDrive/Magisterka/evaluation/excluded_probe_all_models/excluded_probe_qwen_trocr_predictions.csv
```

Potem wystarczy wyświetlić:

```python
import pandas as pd
path = "/content/drive/MyDrive/Magisterka/evaluation/excluded_probe_all_models/excluded_probe_qwen_trocr_predictions.csv"
pd.read_csv(path)[["sample_id", "source_dataset", "display_reference", "model", "prediction"]]
```
