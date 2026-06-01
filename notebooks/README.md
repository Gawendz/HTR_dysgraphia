# Notebooks

This directory contains re-runnable report notebooks for the three model families:

- `qwen3vl_experiments.ipynb`,
- `trocr_experiments.ipynb`,
- `kraken_experiments.ipynb`.

Each notebook has saved output tables and embedded PNG figures, so it can be read
as a record of the thesis experiments without rerunning the code. The same
notebook can also be rerun in a forked repository. If new CSV or PNG files are
written to `results/tables/` and `results/figures/`, rerunning the cells refreshes
the displayed tables and plots.

For new experiments, keep the original thesis outputs unchanged and save custom
artifacts under:

```text
results/custom_tables/
results/custom_figures/
```

Helper functions for saving custom tables and figures are defined near the top
of each notebook.

The notebooks are not intended to replace the reusable scripts. Training and
evaluation code is stored under `src/`.
