# Datasets

This directory contains dataset manifests, annotations, and statistics used in the thesis experiments.

## `polish_forms/`

Custom Polish handwriting dataset prepared from form scans. The repository includes:

- participant/form manifests,
- line-level manifests,
- split information,
- dataset statistics.

Large raw scan folders and temporary preprocessing outputs are not included in the repository.

## `malaysian/`

Line-level annotations and evaluation manifests for the Malaysian handwriting dataset. The dataset contains LPD and PD groups. The repository includes both raw and inverted-color manifests used in the OCR/HTR experiments.

## `iam/`

IAM was used as an external HTR control benchmark. The actual IAM line images were loaded from Hugging Face during experiments, so only the description of the subset is stored here.
