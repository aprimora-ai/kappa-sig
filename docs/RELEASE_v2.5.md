# Kappa Sentinel Release Notes

## Release: v2.5 (2026-03-31)

### Pipeline Update (14 steps)
- Pipeline expanded from 13 to 14 steps
- Step 8 added: Impact Sensitivity Index computation (Section 8.12)
- Pipeline version label updated to v5.6

### New Scripts

**`compute_impact_sensitivity.py`** — Impact Sensitivity Index (Section 8.12)
- Computes I_S = Theta_A * Oh / (Oh_pre + delta) daily for all universes
- Classifies: LOW / ELEVATED / HIGH_VULNERABILITY / BARRIER_BREACHED
- Computes barrier gap (distance to breach)
- Saves to `data/v2_monitoring/impact_sensitivity.json`

**`run_experiment3.py`** — Beta Calibration (Section 8.13)
- Logistic regression on pooled daily observations from 19 universes
- Pre-event censoring (excludes post-t_S days)
- Leave-one-universe-out cross-validation
- Bootstrap (1000x) for coefficient confidence intervals
- Acceptance: no beta changes sign, ECE < 0.15
- Result: H=90 PASS (ECE=0.099), H=30 FAIL (ECE=0.182)

**`run_experiment3b.py`** — Robustness Validation (Section 8.14)
- Ablation: remove each covariate, measure AUC drop
- Shuffle test: 1000 permutations, empirical p-value
- Random features: inject 4 noise columns, verify beta stability
- Result: ALL TESTS PASS

### Key Results
- 21 universes monitored (expanded from 18)
- 6 EMERGENCY, 4 WARNING, 0 WATCH, 9 NOMINAL
- 3 prospective cases tracked: asia_pacific (RAMP, 85 steps), x_brazil_vuln (RAMP, 230 steps), x_us_systemic (RAMP, 555 steps)
- Impact Sensitivity: 4 BARRIER_BREACHED, 1 HIGH_VULNERABILITY
- Global Risk Score: 52.4/100

### Experiment Results Summary
- Theta_A is the dominant hazard predictor (beta=5.38, AUC drops 0.226 when removed)
- Capacity depletion rate is redundant when geometry included
- Permutation test: p < 0.001 (signal is real)
- Random features: all real betas stable, all noise betas zero

### Dependencies
- Python 3.8+
- scikit-learn (for experiment3/3b)
- numpy, pandas, yfinance

---
David Ohio | Independent Researcher | odavidohio@gmail.com
