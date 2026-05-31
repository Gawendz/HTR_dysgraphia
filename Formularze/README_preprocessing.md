# Preprocessing formularzy pisma odręcznego

## Zalecany sposób pracy

1. Wrzuć skany albo zdjęcia podpisanych formularzy do folderu:

```powershell
C:\Magisterka\Formularze\raw
```

Najlepsze są skany. Zdjęcia też działają, jeśli widać wszystkie cztery czarne markery narożne i kartka nie jest ucięta.

2. Uruchom preprocessing:

Jeżeli uruchamiasz na świeżym środowisku, najpierw doinstaluj zależności:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" -m pip install -r "C:\Magisterka\Formularze\requirements_preprocessing.txt"
```

```powershell
cd C:\Magisterka\Formularze
..\.venv_ocr_gpu\Scripts\python.exe process_forms.py --input raw --output processed --template-pdf Formularz.pdf
```

Jeśli PowerShell nie lubi powyższej ścieżki, użyj pełnej:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\process_forms.py" --input "C:\Magisterka\Formularze\raw" --output "C:\Magisterka\Formularze\processed" --template-pdf "C:\Magisterka\Formularze\Formularz.pdf"
```

3. Wyniki pojawią się w:

```text
processed/
  aligned/          # wyprostowane formularze
  lines_flat/       # wycięte linie pisma
  metadata_crops/   # crop roku urodzenia i pola "inne"
  dataset/          # dane podzielone na train/val/test i kategorie
  trocr/            # CSV dla TrOCR
  kraken_gt/        # obrazy + .gt.txt dla Kraken
  qwen/             # JSONL dla Qwen/VLM
  forms.csv         # metadane formularzy
  manifest.csv      # linie eksportowane do modeli
  manifest_all.csv  # wszystkie pola linii, także puste/niepewne
```

Domyślnie puste albo niepewne linie nie trafiają do datasetu treningowego. Zostają jednak w `manifest_all.csv` z kolumnami:

```text
line_has_handwriting
line_status
line_exportable
line_ink_ratio
line_ink_bbox_ratio
```

Jeżeli chcesz mimo wszystko eksportować także puste/niepewne linie, dodaj:

```powershell
--include-empty-lines
```

## Półautomatyczny odczyt roku przez Qwen

Po preprocessingu możesz uruchomić Qwen tylko na małych cropach pola `Rok ur.`:

Jeżeli środowisko nie ma jeszcze zależności Qwena:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" -m pip install -r "C:\Magisterka\Formularze\requirements_qwen_birth_year.txt"
```

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\read_birth_years_qwen.py" --processed-dir "C:\Magisterka\Formularze\processed" --model "Qwen/Qwen3-VL-2B-Instruct"
```

Ten krok:

- czyta obrazy z `processed/metadata_crops/*_birth_year.png`,
- zapisuje surowe odpowiedzi w `processed/birth_year_qwen_predictions.csv`,
- aktualizuje `forms.csv`, `manifest.csv` i `manifest.jsonl`,
- przebudowuje katalog `dataset/`, więc próbki trafiają np. do `dataset/train/nie/kobieta/year_2010/`,
- przebudowuje eksporty `trocr/`, `kraken_gt/` i `qwen/`,
- generuje statystyki w `processed/stats/`.

Jeśli chcesz najpierw tylko zebrać predykcje Qwena bez aktualizowania manifestów:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\read_birth_years_qwen.py" --processed-dir "C:\Magisterka\Formularze\processed" --no-apply
```

Po ręcznej korekcie `birth_year_qwen_predictions.csv` możesz zastosować poprawione wyniki:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\read_birth_years_qwen.py" --processed-dir "C:\Magisterka\Formularze\processed" --predictions-csv "C:\Magisterka\Formularze\processed\birth_year_qwen_predictions.csv"
```

## Rok urodzenia

Jeżeli nie chcesz używać Qwena albo chcesz nadpisać część wyników, utwórz CSV:

```csv
writer_id,birth_year
img_6011,2008
img_6012,2010
```

i uruchom:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\process_forms.py" --input "C:\Magisterka\Formularze\raw" --output "C:\Magisterka\Formularze\processed" --template-pdf "C:\Magisterka\Formularze\Formularz.pdf" --metadata-overrides "C:\Magisterka\Formularze\metadata_overrides.csv"
```

## Split danych

Podział train/val/test jest robiony po `writer_id`, czyli wszystkie linie jednej osoby trafiają do tego samego splitu. Skrypt próbuje zachować proporcje w grupach `difficulty_group + sex`. Domyślnie:

```text
train = 80%
val   = 10%
test  = 10%
```

Możesz zmienić proporcje:

```powershell
--train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

## Statystyki

Po kroku z Qwenem w `processed/stats/` znajdziesz m.in.:

```text
forms_by_year_sex.csv                 # osoby według rocznika i płci
forms_by_year_difficulty_sex.csv      # osoby według rocznika, kategorii i płci
forms_by_difficulty_sex.csv           # osoby według kategorii i płci
forms_by_split_difficulty_sex.csv     # osoby w train/val/test
lines_by_split_difficulty_sex.csv     # liczba wyciętych linii w train/val/test
line_slots_by_status.csv              # liczba wypełnionych i pustych pól linii
line_slots_by_writer_status.csv       # puste/wypełnione pola per osoba
summary.json                          # szybkie podsumowanie
```

## Ręczna walidacja w panelu

Po preprocessingu możesz odpalić lokalny panel do szybkiego sprawdzania formularzy:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\review_forms_app.py" --processed-dir "C:\Magisterka\Formularze\processed_320_check" --port 8765
```

Otwórz w przeglądarce:

```text
http://127.0.0.1:8765
```

Panel pokazuje po lewej wyprostowany formularz, a po prawej najważniejsze pola z `forms.csv`.
Możesz poprawiać `sex`, `difficulty`, `difficulty_group`, `birth_year`, `split`
oraz `gt_text` dla każdej linii. Przy liniach jest też `GT status`:

- `ok` - linia może trafić do treningu/ewaluacji,
- `niepewne` - linia zostaje w `manifest_all.csv`, ale nie jest eksportowana,
- `wyklucz` - linia zostaje w `manifest_all.csv`, ale nie jest eksportowana.

Po kliknięciu `Zapisz` formularz jest automatycznie oznaczany jako ręcznie sprawdzony,
a zmiany trafiają do `forms.csv`, `manifest_all.csv`, `manifest.csv` i `manifest.jsonl`.

Przycisk `Przebuduj eksporty` odtwarza katalogi `dataset/`, `trocr/`, `qwen/`
i `kraken_gt/` na podstawie aktualnych CSV, więc używaj go po większej serii poprawek.

## Zbyt szerokie albo dwuliniowe odpowiedzi

Cropy linii są wycinane na pełną szerokość wyprostowanej strony, od lewej do prawej
krawędzi arkusza. Dzięki temu długie odpowiedzi wychodzące poza szarą linię nie są
ucinane po prawej stronie.

Jeżeli ktoś zapisał odpowiedź w dwóch liniach, a całość mieści się w pionowym obszarze
tej samej pozycji formularza, crop zwykle ją obejmie. Jeżeli jednak druga część weszła
w obszar kolejnej pozycji formularza, taka próbka nie jest czystą jedną linią HTR.
Wtedy najlepiej ustawić `GT status = wyklucz` albo `niepewne` i nie używać jej w głównej
ewaluacji.

Istniejące cropy można odświeżyć bez ponownego przetwarzania PDF-ów:

```powershell
& "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" "C:\Magisterka\Formularze\refresh_line_crops.py" --processed-dir "C:\Magisterka\Formularze\processed_320_check" --template-pdf "C:\Magisterka\Formularze\Formularz.pdf"
```
