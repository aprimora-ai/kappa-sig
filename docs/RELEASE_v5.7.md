# Kappa-FIN v5.7 Release Notes

## Release: v5.7 (2026-03-31)

### Engine Changes (engine_v5_fixed.py)

**FIX-11: Multi-episode geometric activation awareness**
- Previous versions classified based on FIRST activation only
- If Episode 1 was a spike that reverted, signal was labeled SPIKE even if Episode 2 was a sustained ramp
- Fix computes `current_run` from END, overrides SPIKE/BUILDING to RAMP when `current_run >= RAMP_MIN_PERSIST`
- Adds `effective_t_G` (start of current episode) vs `t_G` (first historical)
- Fixes `crystallization_duration` to reflect current episode length

**FIX-10: Multi-observable geometric spot check (Section 8.11)**
- Computes Theta_A independently for Oh(t), eta(t), mean_corr(t)
- Reports consensus: ALIGNED / PARTIAL / NONE
- Resolves circularity concern (L4)

**Calibrated Beta Coefficients (Experiment 3, Section 8.13)**
- H=90 coefficients calibrated via logistic regression on 19 universes (~4000 obs)
- ECE = 0.099 (well-calibrated), all signs stable in bootstrap(1000x)
- `beta_0 = -2.7049` (was -6.0)
- `beta_F = +1.1040` (was +1.0) — kappa_F
- `beta_rho = +0.0287` (was +0.5) — rho_bar_pos (nearly redundant)
- `beta_eta = +0.4146` (was +0.3) — eta_norm
- `beta_A = +5.3808` (was +0.5) — theta_A (dominant predictor)

### Key Finding
Geometric activation (Theta_A) is the dominant predictor of regime transition probability.
Removing Theta_A collapses model AUC from 0.753 to 0.527 (ablation test).
Theta_A alone achieves AUC = 0.706 (94% of full model).
Capacity depletion rate (rho_bar) is statistically redundant when geometry is included.

### Robustness (Experiment 3b)
- **Ablation**: Theta_A removal causes largest AUC drop (+0.226)
- **Permutation test**: p < 0.001 (1000 shuffles, real AUC 0.753 vs shuffled max 0.544)
- **Random features**: All real betas stable (<7% drift), all random betas ~zero

### Paper Update
- Paper v2.5 updated to 925 lines
- §8.12: Impact Sensitivity formalized (Damage Barrier Model)
- §8.13: Experiment 3 — Hazard Model Calibration results
- §8.14: Experiment 3b — Robustness Validation (ablation, permutation, random features)
- 7 empirical findings (was 5)

### Full Fix History
| Version | Fix | Issue |
|---------|-----|-------|
| v5.0 | — | Initial implementation |
| v5.1 | FIX-1 | phi_star=0 guard |
| v5.1 | FIX-2 | Theta_A cap(5.0) |
| v5.1 | FIX-3 | kappa_F cap(20.0) |
| v5.2 | FIX-4 | Multi-source alert logic |
| v5.2 | FIX-5 | Geometry reliability gate |
| v5.3 | FIX-6 | eta EMA smoothing |
| v5.4 | FIX-7 | z-score normalization for RQA |
| v5.5 | FIX-8 | Geometric activation classifier |
| v5.5 | FIX-9 | Activation tracking (t_G, duration) |
| v5.6 | FIX-10 | Multi-observable spot check |
| v5.7 | FIX-11 | Multi-episode awareness |
| v5.7 | — | Calibrated betas H=90 |

---
David Ohio | Independent Researcher | odavidohio@gmail.com
