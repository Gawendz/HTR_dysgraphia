# Kontekst datasetu HTR do przekazania Codexowi

Projekt znajduje się w:

```text
C:\Magisterka
```

Główny folder formularzy:

```text
C:\Magisterka\Formularze
```

Aktualnie przetworzony i zwalidowany dataset jest tutaj:

```text
C:\Magisterka\Formularze\processed_320_check
```

## Cel datasetu

Dataset służy do porównania modeli rozpoznawania pisma odręcznego HTR/OCR, ze szczególnym uwzględnieniem pisma osób deklarujących dysgrafię.

Planowane modele:

- Kraken
- TrOCR
- Qwen / MLLM

Podstawowa jednostka danych to:

```text
obraz jednej linii pisma -> transkrypcja tej linii
```

Ewaluacja powinna być robiona przede wszystkim na poziomie:

- CER
- WER
- opcjonalnie normalized CER/WER, np. lowercase, bez interpunkcji

## Źródła danych

Surowe formularze:

```text
C:\Magisterka\Formularze\raw_examples
```

Pusty formularz wzorcowy:

```text
C:\Magisterka\Formularze\Formularz.pdf
```

Formularze mają markery narożne, które były używane do prostowania geometrii.

## Główne pliki po preprocessingu

```text
C:\Magisterka\Formularze\processed_320_check\forms.csv
C:\Magisterka\Formularze\processed_320_check\manifest_all.csv
C:\Magisterka\Formularze\processed_320_check\manifest.csv
C:\Magisterka\Formularze\processed_320_check\manifest.jsonl
```

Znaczenie plików:

- `forms.csv` - jeden wiersz na formularz/osobę.
- `manifest_all.csv` - wszystkie sloty linii, także `niepewne`, `wyklucz` i puste.
- `manifest.csv` - tylko linie eksportowalne, czyli finalny dataset do treningu/ewaluacji.
- `manifest.jsonl` - JSONL odpowiadający `manifest.csv`.

## Aktualne liczby datasetu

Stan po walidacji i przebudowie eksportów:

```text
formularze / osoby:             320
wszystkie sloty linii:          2560
wyeksportowane obrazy linii:    2313
linie niewyeksportowane:        247
```

Spójność eksportów:

```text
manifest.csv:       2313 wiersze
dataset/*.png:      2313 pliki PNG
trocr/*.csv:        2313 wiersze
qwen/*.jsonl:       2313 wiersze
kraken_gt/*.png:    2313 pliki PNG
kraken_gt/*.gt.txt: 2313 pliki GT
```

## Status GT

W `manifest_all.csv` każda linia ma `gt_status`.

Znaczenie:

- `ok` - linia jest używana w datasetach.
- `manual_reviewed` - historyczna wartość, traktować jak `ok`.
- `template_auto` - historyczna wartość, traktować jak `ok`, jeśli jest w `manifest.csv`.
- `uncertain` / `niepewne` - linia zostaje tylko w `manifest_all.csv`, nie jest eksportowana.
- `exclude` / `wyklucz` - linia zostaje tylko w `manifest_all.csv`, nie jest eksportowana.
- `empty_line` - pusta linia, nie eksportować.

Do treningu i głównej ewaluacji używać tylko `manifest.csv`.

Do analiz jakościowych można używać `manifest_all.csv`, szczególnie linii `uncertain`.

Aktualny rozkład statusów w `manifest_all.csv`:

```text
ok:        2313
niepewne:   199
wyklucz:     48
```

## Podział train/val/test

Split jest po osobie, czyli wszystkie linie jednej osoby trafiają do tego samego zbioru.

Nie wolno mieszać linii jednej osoby między `train`, `val` i `test`, bo grozi to przeciekiem stylu pisma.

Aktualny podział formularzy:

```text
train: 254 formularze
val:    33 formularze
test:   33 formularze
```

Aktualny podział obrazów linii:

```text
train: 1844 obrazy linii
val:    228 obrazów linii
test:   241 obrazów linii
```

Kolumna splitu:

```text
split
```

Występuje w:

- `forms.csv`
- `manifest_all.csv`
- `manifest.csv`

## Kategorie trudności

W `forms.csv` i manifestach są dwie kolumny:

```text
difficulty
difficulty_group
```

`difficulty` to bardziej szczegółowa etykieta z formularza lub walidacji.

Aktualny rozkład `difficulty` dla formularzy:

```text
brak trudności: 215
dysgrafia:       67
dysleksja:       31
dysortografia:    2
inne:             2
unknown:          3
```

Aktualny rozkład `difficulty` dla wyeksportowanych obrazów linii:

```text
brak trudności: 1613
dysgrafia:       433
dysleksja:       230
dysortografia:    13
inne:              7
unknown:          17
```

`difficulty_group` to uproszczona grupa do głównej analizy:

```text
nie
dysgrafia
inne
```

Aktualny rozkład `difficulty_group` dla formularzy:

```text
brak trudności: 212
dysgrafia:       71
inne:            37
```

Aktualny rozkład `difficulty_group` dla wyeksportowanych obrazów linii:

```text
brak trudności: 1594
dysgrafia:       458
inne:            261
```

W pracy najlepiej pisać ostrożnie: grupa osób deklarujących dysgrafię, a nie diagnozować dysgrafię samodzielnie na podstawie wyglądu pisma.

## Płeć i roczniki

Aktualny rozkład płci dla formularzy:

```text
mężczyzna: 202
kobieta:   116
unknown:     2
```

Aktualny rozkład roczników dla formularzy:

```text
2005:  3
2006: 17
2007: 52
2008: 52
2009: 60
2010: 95
2011: 14
2012:  8
2013: 10
2014:  8
unknown: 1
```

Wiek w raportach liczony był jako wiek rocznikowy:

```text
2026 - rok urodzenia
```

Nie ma dokładnych dat urodzenia.

## Foldery eksportowe

### Uniwersalny dataset obrazów

```text
C:\Magisterka\Formularze\processed_320_check\dataset
```

Struktura:

```text
dataset\<split>\<difficulty_group>\<sex>\year_<birth_year>\<sample_id>.png
```

Przykład:

```text
dataset\train\nie\mezczyzna\year_2010\001_p01_A_l01.png
```

### TrOCR

```text
C:\Magisterka\Formularze\processed_320_check\trocr
```

Pliki:

```text
train.csv
val.csv
test.csv
```

Każdy CSV ma kolumny:

```text
image_path,text
```

### Qwen

```text
C:\Magisterka\Formularze\processed_320_check\qwen
```

Pliki:

```text
train.jsonl
val.jsonl
test.jsonl
```

Każdy rekord ma m.in.:

```json
{
  "image": "...",
  "prompt": "Transcribe exactly the handwritten Polish text in this image. Return only the transcription.",
  "response": "...",
  "metadata": {
    "sample_id": "...",
    "writer_id": "...",
    "set_id": "...",
    "line_id": 1,
    "sex": "...",
    "birth_year": "...",
    "difficulty_group": "...",
    "difficulty": "..."
  }
}
```

### Kraken

```text
C:\Magisterka\Formularze\processed_320_check\kraken_gt
```

Struktura:

```text
kraken_gt\<split>\<sample_id>.png
kraken_gt\<split>\<sample_id>.gt.txt
```

Każdy `.gt.txt` zawiera transkrypcję dla odpowiadającego obrazu.

## Ważne zasady GT

GT powinno zawierać tekst widoczny w rękopisie, nie komentarze.

Nie wpisywać do GT tagów typu:

```text
[niepewne]
[wyklucz]
```

Niepewność oznaczać tylko w kolumnie:

```text
gt_status
```

Zasady przy ręcznej walidacji:

- jeśli tekst jest jednoznaczny, `gt_status = ok`;
- jeśli tekst jest ważny, ale niepewny, `gt_status = uncertain`;
- jeśli crop jest zły, tekst jest ucięty, przekreślony lub niemożliwy do rzetelnej transkrypcji, `gt_status = exclude`;
- do głównych wyników CER/WER brać tylko `manifest.csv`.

## Specjalne przypadki cropów

Dwa formularze mają specjalnie poszerzone cropy pionowe, bo uczestnicy pisali odpowiedzi w dwóch liniach:

```text
212_p01
227_p01
```

Dla nich w `process_forms.py` istnieje lista:

```python
DOUBLE_LINE_WRITERS = {"212_p01", "227_p01"}
```

Dla tych przypadków zastosowano:

- pełną szerokość strony,
- większą wysokość cropu,
- mocniejsze wybielanie tekstu drukowanego formularza.

## Skrypty

Główny preprocessing:

```text
C:\Magisterka\Formularze\process_forms.py
```

Panel ręcznej walidacji:

```text
C:\Magisterka\Formularze\review_forms_app.py
```

Odświeżanie cropów bez pełnego preprocessingu:

```text
C:\Magisterka\Formularze\refresh_line_crops.py
```

Generowanie statystyk:

```text
C:\Magisterka\Formularze\generate_dataset_statistics.py
```

Uruchomienie panelu:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\review_forms_app.py" --processed-dir "C:\Magisterka\Formularze\processed_320_check" --port 8765
```

Odświeżenie raportu statystycznego:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\generate_dataset_statistics.py" --processed-dir "C:\Magisterka\Formularze\processed_320_check" --reference-year 2026
```

## Raport statystyczny do pracy

Gotowe statystyki są w:

```text
C:\Magisterka\Formularze\processed_320_check\stats_thesis
```

Najważniejsze pliki:

```text
stats_thesis\summary.md
stats_thesis\summary.json
stats_thesis\dataset_statistics.xlsx
stats_thesis\tables\*.csv
stats_thesis\figures\*.png
```

Wykresy PNG nadają się do wstawienia do pracy magisterskiej.

## Backup

Przed ostatnią przebudową eksportów wykonano backup:

```text
C:\Magisterka\Formularze\processed_320_check\backup_before_rebuild_20260524_184940
```

Zawiera:

```text
forms.csv
manifest_all.csv
manifest.csv
manifest.jsonl
```

Najważniejszy plik do odzyskania ręcznej walidacji to:

```text
manifest_all.csv
```
