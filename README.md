# Eksperymenty HTR/OCR do pracy magisterskiej

Repozytorium zawiera kod, notebooki, skrypty ewaluacyjne oraz zagregowane wyniki wykorzystane w pracy magisterskiej dotyczącej rozpoznawania pisma odręcznego. Eksperymenty obejmowały trzy rozwiązania:

- `Qwen3-VL` - model wielomodalny testowany zero-shot oraz dostrajany metodą LoRA,
- `TrOCR` - transformerowy model OCR/HTR typu encoder-decoder,
- `Kraken/CRNN` - klasyczny system HTR oparty o rozpoznawanie linii tekstu.

## Najważniejsze katalogi

- `Qwen3VL/`, `TrOCR/`, `CRNN/` - skrypty inferencji zero-shot i analizy błędów.
- `FineTuning/` - skrypty przygotowania danych, treningu i ewaluacji Qwen3-VL, TrOCR oraz Kraken.
- `Chapter3_results/figures/` - wykresy i przykłady jakościowe użyte w rozdziale wynikowym.
- `Chapter3_results/tables/` - tabele CSV z najważniejszymi wynikami.
- `Qwen_final_report/`, `TrOCR_final_report/`, `Kraken_final_report/` - uporządkowane notebooki i podsumowania dla poszczególnych modeli.
- `Formularze/` - skrypty do przygotowania i walidacji własnego zbioru danych oraz materiały opisowe do części metodologicznej.
- `chapter3_wyniki.tex`, `chapter4_podsumowanie.tex` - fragmenty pracy z opisem wyników i podsumowaniem.
- `methodology_chatgpt_context.md` - skrócona notatka metodologiczna dla dalszego pisania pracy.

## Co nie jest wersjonowane

Repo celowo nie przechowuje lokalnych środowisk, zależności, surowych obrazów pisma, paczek ZIP, checkpointów i wag modeli. Te pliki są duże, często odtwarzalne i nie nadają się do zwykłego Gita. Szczegóły znajdują się w `ARTIFACTS_NOT_VERSIONED.md`.

## Odtworzenie wyników

Do pełnego odtworzenia treningu potrzebne są zewnętrzne artefakty: dane obrazowe, checkpointy modeli oraz paczki przygotowane do Google Colab. W repo pozostawiono skrypty przygotowania danych i ewaluacji oraz finalne tabele i wykresy, które były podstawą opisów w pracy.
