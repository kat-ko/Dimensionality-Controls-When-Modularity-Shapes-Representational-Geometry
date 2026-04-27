

This repository is a focused reproducibility package for the paper in `overleaf/`.
It contains the exact experiment subset used for the main paper comparison:
modular recurrent networks versus a matched non-modular baseline.

## What this paper studies

The paper tests how network structure and task similarity affect:

- transfer to a new task, and
- interference with a previously learned task.

The included pipeline reproduces the main figure set from the paper after local training.

- `a1b2_modular/`: experiment code, scripts, and participant inputs
- `overleaf/`: paper source (`main.tex`, bibliography, expected figure files)

## Quickstart

### 1) Set up environment

```bash
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
pip install -e a1b2_modular
```

### 2) Preprocess participant data

```bash
python a1b2_modular/scripts/01_preprocess_data.py
```

### 3) Run the paper experiment matrix

To reproduce paper figures run each condition below:

```bash
python a1b2_modular/scripts/02_run_simulations.py <condition> --base-folder a1b2_modular
```

Conditions:

- `two_module_rnn_25_task_routed_no_comms_nb2_init0.001`
- `two_module_rnn_25_task_routed_no_comms_nb2_init0.01`
- `two_module_rnn_25_task_routed_no_comms_nb2_init0.1`
- `two_module_rnn_25_task_routed_no_comms_nb2`
- `two_module_rnn_25_task_routed_no_comms_nb2_init2`
- `single_module_rnn_50_nb2_init0.001`
- `single_module_rnn_50_nb2_init0.01`
- `single_module_rnn_50_nb2_init0.1`
- `single_module_rnn_50_nb2`
- `single_module_rnn_50_nb2_init2`

### 4) Fit paper metrics for each simulation run

```bash
python a1b2_modular/scripts/03_fit_vonmises.py simulations --base-folder a1b2_modular --sim-name <run_id>
```

To compute interference (participant-side fits):

```bash
python a1b2_modular/scripts/03_fit_vonmises.py participants --base-folder a1b2_modular
```

### 5) Generate paper figures

Run:

- `a1b2_modular/notebooks/paper_figures_size25.ipynb`

## Reproducibility note

Simulation outputs are intentionally not committed due to file size.
Run the pipeline locally to regenerate results and figures end to end.
