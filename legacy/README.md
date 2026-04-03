# Legacy Results from TopoCML/HEIMDALL

Historical experiment results from the original TopoCML project (2025–2026), preserved for provenance and reproducibility.

## Contents

### `heimdall_sweep/`
Full-model layer sweep results that determined the best attention layer for each model. These sweeps tested all 32 layers with paired t-tests on 100 samples per layer ("cannonball run").

- `heimdall_sweep_phi3.csv` — Phi-3 Mini: **Layer 28** (p=2.26e-08)
- `heimdall_sweep_mistral.csv` — Mistral-7B: **Layer 13** (p=2.30e-20)
- `heimdall_sweep_llama.csv` — Llama-3.1-8B: **Layer 15** (p=2.64e-15)
- `heimdall_full_model_sweep_results.csv` — Consolidated cross-model results

### `kappa_llm_original/`
Original per-head Kappa-LLM results from the published paper (DOI: 10.5281/zenodo.18883790). These use the **per-head** methodology (before the SIG inter-head correlation approach).

Key results:
- **Phi-3**: Kappa AUC = **0.942**, Accuracy = 85%
- **Mistral**: Kappa AUC = **0.871**, Accuracy = 70.4%
- **Llama**: Kappa AUC = **0.791**, Accuracy = 61.3%

Includes per-sample parquet files with headwise Kappa states, ROC curves, observable distributions, and feature importance plots.

### `kappa_neuro/`
EEG seizure prediction results from CHB-MIT Scalp EEG Database (21 pediatric patients) and EEGMMIDB healthy controls (93+ subjects). **Deferred to dedicated clinical paper.**

- `consolidated_summary.csv` — 120 rows: Oh_max, Oh_mean, phi_max, Xi_mean per subject
- `four_metrics.json` — 42,640 lines of detailed excursion data
- `ictal_analysis.csv` — Ictal vs interictal comparison
- `state_analysis/` — Classification reports, ROC curves, seizure trajectories, confusion matrices

Scripts: `katashi_eeg.py`, `launcher_v2.py`, `compute_four_metrics.py`, `state_classifier.py`

### `kappa_fin_original/`
Kappa-FIN v3/v4 results across **20+ historical financial scenarios** including GFC 2008, Dotcom 2000, COVID-19, Eurozone 2011, SVB 2023, Iran War 2026, and many more. Each scenario includes dashboard PNGs, state CSVs, viscosity CSVs, and analysis reports.

### `heimdall_paper/`
Original HEIMDALL paper (LaTeX + PDF) — the first publication of the coherence inversion finding in LLMs.

## Relationship to Current Work

These legacy results informed the design decisions in the current Kappa-SIG framework:
- HEIMDALL sweep → layer selection for SIG experiments
- Per-head Kappa-LLM → motivated transition to inter-head correlation (SIG)
- NEURO pipeline → ready for dedicated Kappa-NEURO paper
- Kappa-FIN scenarios → validated across engine versions v3→v5.8c
