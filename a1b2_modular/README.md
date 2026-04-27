# a1b2_modular (paper runtime subset)

This folder contains the minimal runtime code used by the paper reproducibility flow.
The main paper path uses 10 conditions: task-routed modular size-25 plus
single-network baseline size-50, each across five initialization scales.

## What is inside

- `a1b2/`: package code (data, models, training, analysis)
- `scripts/`: runnable pipeline scripts
- `data/`: participant inputs and locally generated simulation outputs
- `notebooks/`: place `paper_figures_size25.ipynb` here
- `a1b2/models/experiments.json`: pruned to the paper condition matrix

## Script order

Run from repository root:

1. `python a1b2_modular/scripts/01_preprocess_data.py`
2. `python a1b2_modular/scripts/02_run_simulations.py <condition> --base-folder a1b2_modular`
3. `python a1b2_modular/scripts/03_fit_vonmises.py simulations --base-folder a1b2_modular --sim-name <run_id>`

Optional:

- `python a1b2_modular/scripts/03_fit_vonmises.py participants --base-folder a1b2_modular`
- `python a1b2_modular/scripts/03_functional_specialization.py <run_id> --base-folder a1b2_modular` (supplementary, not needed for main paper figures)

