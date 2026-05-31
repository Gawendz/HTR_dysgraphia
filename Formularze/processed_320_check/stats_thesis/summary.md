# Statystyki datasetu

Folder danych: `C:\Magisterka\Formularze\processed_320_check`
Rok referencyjny do wieku rocznikowego: `2026`.

## Podsumowanie

| metryka                     | wartość |
| --------------------------- | ------- |
| formularze / osoby          | 320     |
| wszystkie sloty linii       | 2560    |
| wyeksportowane obrazy linii | 2313    |
| linie niewyeksportowane     | 247     |
| pliki PNG w dataset/        | 2313    |
| wiersze TrOCR CSV           | 2313    |
| wiersze Qwen JSONL          | 2313    |
| pliki Kraken PNG            | 2313    |
| pliki Kraken GT             | 2313    |

## Formularze według deklarowanej trudności

| difficulty_raw_label | forms | percent |
| -------------------- | ----- | ------- |
| brak trudności       | 215   | 67.19   |
| dysgrafia            | 67    | 20.94   |
| dysleksja            | 31    | 9.69    |
| unknown              | 3     | 0.94    |
| dysortografia        | 2     | 0.62    |
| inne                 | 2     | 0.62    |

## Wyeksportowane obrazy linii według deklarowanej trudności

| difficulty_raw_label | images | percent |
| -------------------- | ------ | ------- |
| brak trudności       | 1613   | 69.74   |
| dysgrafia            | 433    | 18.72   |
| dysleksja            | 230    | 9.94    |
| unknown              | 17     | 0.73    |
| dysortografia        | 13     | 0.56    |
| inne                 | 7      | 0.3     |

## Formularze według grupy

| difficulty_group_label | forms | percent |
| ---------------------- | ----- | ------- |
| brak trudności         | 212   | 66.25   |
| dysgrafia              | 71    | 22.19   |
| inne                   | 37    | 11.56   |

## Wyeksportowane obrazy linii według grupy

| difficulty_group_label | images | percent |
| ---------------------- | ------ | ------- |
| brak trudności         | 1594   | 68.91   |
| dysgrafia              | 458    | 19.8    |
| inne                   | 261    | 11.28   |

## Formularze według płci

| sex_label | forms | percent |
| --------- | ----- | ------- |
| mężczyzna | 202   | 63.12   |
| kobieta   | 116   | 36.25   |
| unknown   | 2     | 0.62    |

## Formularze według roku urodzenia

| birth_year_label | forms | percent |
| ---------------- | ----- | ------- |
| 2005             | 3     | 0.94    |
| 2006             | 17    | 5.31    |
| 2007             | 52    | 16.25   |
| 2008             | 52    | 16.25   |
| 2009             | 60    | 18.75   |
| 2010             | 95    | 29.69   |
| 2011             | 14    | 4.38    |
| 2012             | 8     | 2.5     |
| 2013             | 10    | 3.12    |
| 2014             | 8     | 2.5     |
| unknown          | 1     | 0.31    |

## Podział formularzy na train/val/test

| split | forms | percent |
| ----- | ----- | ------- |
| train | 254   | 79.38   |
| test  | 33    | 10.31   |
| val   | 33    | 10.31   |

## Podział obrazów linii na train/val/test

| split | images | percent |
| ----- | ------ | ------- |
| train | 1844   | 79.72   |
| test  | 241    | 10.42   |
| val   | 228    | 9.86    |

## Status GT dla wszystkich slotów linii

| gt_status_label | line_slots | percent |
| --------------- | ---------- | ------- |
| ok              | 2313       | 90.35   |
| niepewne        | 199        | 7.77    |
| wyklucz         | 48         | 1.88    |

## Pliki

- Tabele CSV: `C:\Magisterka\Formularze\processed_320_check\stats_thesis\tables`
- Wykresy PNG: `C:\Magisterka\Formularze\processed_320_check\stats_thesis\figures`
- Arkusz XLSX: `C:\Magisterka\Formularze\processed_320_check\stats_thesis\dataset_statistics.xlsx`