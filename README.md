# Kappa-SIG: Structural Information Geometry, Latent Crystallization Detection, and CALM Layer-Conditional Normalization

**David Ohio** — Independent Researcher
odavidohio@gmail.com

**Extends:** [Kappa Method v2.5](https://doi.org/10.5281/zenodo.19339548)
**License:** CC BY 4.0

---

## Overview

This repository contains the complete Kappa framework: Kappa-FIN engine (v1 through v5.8c), Kappa Method v2 pipeline and experiments, and Kappa-SIG (this paper). It is self-contained for full reproducibility.

Kappa-SIG extends the Kappa Method v2.5 with three contributions:

1. **Structural Information Geometry (SIG):** Detection of structural crystallization from reduced windows using spectral (S1), compressive (S2), topological (S3), and learned (S5) methods.

2. **CALM Layer-Conditional Normalization:** Discovery that CALM attenuates +5.3% AUC of genuine signal. Engine v5.8c implements CALM_SATURATED detection.

3. **Latent Structural Crystallization Coordinate (LSCC):** Supervised latent detector with LOUO AUC 0.942 (vs 0.634 all analytics), seed-stable (CV=0.006), temporal generalization (0.722), CALM-blind-spot coverage (AUC 0.911).

## Key Results

| Test | Result |
|------|--------|
| Wheeler hypothesis (W=1) | Confirmed, p < 0.01, AUC 0.53 |
| CALM attenuation | +5.3% AUC, 100% positive deltas |
| LSCC LOUO AUC | 0.942 (fold-local PCA) |
| LSCC vs all analytics | +0.323 incremental value |
| LSCC non-collapse | max \|r\| = 0.11, R-squared = 0.032 |
| Seed stability (R1) | CV = 0.006 across 5 seeds |
| Dimension robustness (R2) | Spread = 0.035 (dim 4-32) |
| Encoder-only (R3) | 0.611 — classifier essential |
| Temporal split (R4) | AUC = 0.722 (train<2024, test 2024+) |
| Artifact check (R5) | Gap > 0.44 vs null baselines |
| Dual-horizon H=60 | AUC 0.772, ECE 0.082 |
| Dual-horizon H=90 | AUC 0.733, ECE 0.089 |
