# Artefakty nieprzechowywane w Git

Ten plik opisuje ważne artefakty, które zostały pozostawione lokalnie, ale nie są dodawane do repozytorium.

## Pominięte kategorie

- Lokalne środowiska Python i zależności Node.js: `.venv_ocr_gpu/`, `.venv/`, `node_modules/`.
- Surowe i pośrednie obrazy pisma: pełne cropy formularzy, katalogi robocze walidacji oraz obrazy treningowe.
- Paczki transferowe ZIP przygotowane dla Google Colab.
- Wagi modeli i checkpointy treningowe: `*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `pytorch_model.bin`.
- Katalogi stagingowe z danymi fine-tuningowymi.

## Najważniejsze lokalne artefakty

- `Chapter3_results/models/trocr_large_dysgraphia_oversampled_checkpoint_1600/model.safetensors` - lokalnie odtworzony checkpoint TrOCR użyty do dodatkowej próby jakościowej.
- `FineTuning/Kraken/outputs/` - checkpointy treningowe Kraken/CRNN.
- `FineTuning/*_colab*.zip` - paczki danych i skryptów wysyłane do Google Colab.
- `Formularze/processed_320_check/` - pełny katalog roboczy własnego zbioru formularzy; w repo zostawiono tylko `stats_thesis`.

Jeżeli repo ma zostać opublikowane z pełnymi danymi lub wagami, najlepiej użyć osobnego mechanizmu przechowywania artefaktów, np. Google Drive, Zenodo, Hugging Face Hub albo Git LFS.
