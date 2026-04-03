# Kappa-SIG: Structural Information Geometry, Latent Crystallization Detection, and CALM Layer-Conditional Normalization

**David Ohio** — Independent Researcher
odavidohio@gmail.com

**Status:** Formal Extension of Kappa Method v2.5 — April 2026
**Extends:** Kappa Method v2.5 (DOI: 10.5281/zenodo.19339548)
**Implementation:** Kappa-SIG pipeline + engine v5.8c, validated on 21 Sentinel universes

---

## Abstract

We extend the Kappa Method v2.5 with Structural Information Geometry (SIG), a framework for detecting structural crystallization from reduced observational windows using spectral, compressive, topological, and learned methods. Across 21 financial system universes, we establish three principal findings.

First, the Wheeler hypothesis is confirmed: crystallization is statistically detectable from a single correlation matrix snapshot (p < 0.01, W=1), though with limited discriminative power (AUC 0.53). This establishes that structural regime information exists instantaneously in the correlation eigenspectrum, not solely in temporal recurrence dynamics.

Second, CALM-based normalization systematically attenuates geometric signal. When detection targets are redefined without CALM dependency (forward-looking C_norm damage within 90 days), all SIG methods improve uniformly (mean delta +0.053 AUC, 100% positive). This leads to a revision of the CALM framework: structural calmness remains foundational for baselining, but geometric normalization must be layer-conditional rather than universally neutral. The engine is updated (v5.8c) with explicit CALM_SATURATED detection, and dual-horizon calibrated prognostics (H=60: AUC 0.772, ECE 0.082; H=90: AUC 0.733, ECE 0.089).

Third, a structural autoencoder trained on the joint Kappa state trajectory [Oh, phi, eta, mean_corr, DEF, Xi] learns a Latent Structural Crystallization Coordinate (LSCC) that achieves LOUO cross-universe AUC of 0.942, substantially exceeding all analytic methods combined (0.634). The LSCC does not collapse into any manual observable (max |r| = 0.11, linear R-squared = 0.032), drifts systematically before structural damage (78.6% of universes), and maintains AUC 0.911 in CALM-saturated universes where Layer 2 (Theta_A) is blind. Robustness testing confirms seed stability (CV = 0.006), dimension robustness (spread 0.035), temporal generalization (train pre-2024, test 2024+: AUC 0.722), and artifact rejection (shuffled/random baselines near 0.53). The classifier head is essential for regime organization (encoder-only: 0.611), establishing the LSCC as a supervised latent detector rather than an unsupervised structural discovery.

These results establish a complementary multi-layer detection architecture: Layer 0 (SIG/LSCC) provides instantaneous triage and covers CALM-blind spots; Layer 2 (Theta_A) provides temporal geometric diagnosis where CALM is functional; Layer 3 provides calibrated dual-horizon risk estimates.

---

## 1. Motivation and Scope

### 1.1 The Reduced-Window Problem


Kappa v2.5 Layer 2 requires a minimum of 60 steps (approximately 3 months of daily financial data) before RQA-based geometric detection becomes computationally meaningful. This creates a structural blind spot: the first 60 steps of any universe are geometrically unobservable. Among 21 Sentinel universes, 5 experienced structural damage (C_norm < 0.90) within this blind zone (LEFT_CENSORED cases). Kappa-SIG addresses whether crystallization can be detected from shorter windows or instantaneous snapshots.

### 1.2 The CALM Saturation Problem

The v2.5 paper noted that "structural calmness does not imply geometric neutrality" (Section 8.3). When CALM baseline periods exhibit geometrically saturated RQA statistics (sigma_DET or sigma_LAM < 0.005), z-score normalization produces Theta_A = 0 regardless of actual geometric state. This affects 7/21 Sentinel universes, rendering Layer 2 blind in those cases.

### 1.3 Three-Horizon Framework

Kappa-SIG introduces a conceptual framework distinguishing three informational horizons:

**Genotype (1-3 steps):** Instantaneous topological and spectral properties of the correlation matrix. Tests whether the current structural configuration is anomalous.

**Phenotype (30-60+ steps):** Accumulated temporal dynamics captured by RQA. Tests whether the system's trajectory is evolving toward crystallization.

**Exogenous:** External forcing (wars, policy decisions) that cannot be predicted from internal structure alone.

The genotype and phenotype are complementary, not substitutable. The SIG framework provides methods for each scale.

---

## 2. Detection Methods

### 2.1 S1: Spectral Decomposition


Given a rolling correlation matrix C(t) of N assets, we compute three spectral metrics at each step:

- **SCR(t) = lambda_1(t) / sum(lambda_i(t)):** Spectral Concentration Ratio. Higher values indicate dominance of a single eigenmode (crystallization).
- **SGR(t) = (lambda_1 - lambda_2) / lambda_1:** Spectral Gap Ratio. Measures separation between the dominant and subdominant modes.
- **D_MP_KL(t) = KL(rho_empirical || f_MP):** Marchenko-Pastur divergence. True KL divergence between the empirical eigenvalue density (via Gaussian KDE) and the theoretical Marchenko-Pastur distribution for the given N/T ratio. Higher values indicate structured deviation from random matrix theory predictions.

SCR is equivalent to lambda_1/N after normalization, making it mathematically identical to the raw leading eigenvalue as a classifier. This was confirmed empirically: SCR and BL_lam1 produce identical LOUO AUC.

### 2.2 S2: Lempel-Ziv Complexity

The upper-triangular elements of C(t) are binarized against the CALM median and concatenated over a trailing window of W_LZ = 15 steps. Normalized Lempel-Ziv complexity (LZ_norm) measures the compressibility of this binary sequence. Lower complexity indicates more repetitive (crystallized) correlation structure.

The Complexity Depletion Rate (CDR) is the negative slope of LZ_norm over a trailing window of 10 steps. Positive CDR indicates active loss of structural complexity.

### 2.3 S3: Persistent Homology

Distance matrices D(t) = sqrt(2(1 - C(t))) are analyzed via persistent homology (ripser). H1 barcodes capture topological loops in the correlation structure. Two metrics are extracted:

- **PE1(t):** Persistence entropy of the H1 diagram. Lower entropy indicates simpler (more crystallized) topology.
- **TCS(t):** Topological Crystallization Score. Wasserstein-2 distance between the current H1 barcode and the CALM reference barcode. Higher values indicate topological divergence from baseline.

### 2.4 S5: Structural Autoencoder and LSCC


A structural autoencoder operates on windowed state trajectories of the Kappa observables [Oh(t), phi(t), eta(t), mean_corr(t), DEF(t), Xi(t)] with window W=20 steps. Architecture: encoder (120->128->64->8), decoder (8->64->128->120), classifier head (8->16->1, sigmoid). Training uses joint loss: reconstruction MSE + 0.5 * classification BCE, where the classification target is the CALM-free forward-looking damage indicator (C_norm < 0.90 within 90 steps).

The Latent Structural Crystallization Coordinate (LSCC) is defined as the first principal component of the 8-dimensional latent representation z(t), computed via PCA on the encoder outputs. LSCC is interpreted as a supervised latent structural fingerprint for pre-damage vulnerability, not as an unsupervised intrinsic state coordinate.

---

## 3. Experimental Protocol

### 3.1 Data

21 Sentinel universes covering global financial markets (2022-01-01 to 2026-04-01), each with 6-21 assets. Rolling Spearman correlation with shrinkage (lambda=0.1) and angular distance metric. Ground truth labels derived from Kappa v2.5 state: NOMINAL (Theta_A < 0.1), CRYSTALLIZING (0.1 < Theta_A < 2.0), CRYSTALLIZED (Theta_A >= 2.0), DAMAGED (C_norm < 0.90).

### 3.2 Targets

Two target definitions are used throughout:

**CALM-dependent:** Binary classification of CRYSTALLIZING or CRYSTALLIZED (derived from Theta_A, which uses CALM z-score normalization).

**CALM-free:** Forward-looking binary indicator: will C_norm drop below 0.90 within the next H=90 steps? This target depends only on accumulated structural damage (Layer 1), with no CALM normalization in the definition.

### 3.3 Evaluation

All cross-universe evaluations use Leave-One-Universe-Out (LOUO) cross-validation. Scaler fitting occurs within each fold (no information leakage). For the LSCC, PCA is recomputed within each fold (fold-local PCA). AUC-ROC is the primary metric.

---

## 4. Results

### 4.1 Experiment 4a: Window Sweep


Individual SIG methods were evaluated at windows W = 1, 3, 5, 10, 15, 20, 30, 45, 60 across 19 universes (pooled AUC with CALM-dependent labels):

Best individual method: D_MP_KL at W=60, AUC = 0.655. No individual method reaches AUC 0.70. Baselines (BL_lam1, BL_mean_corr, BL_rvol) compete with SIG methods, with BL_rvol reaching 0.65 at W=1. LZ complexity is the most critical feature in the ensemble (ablation delta_AUC = 0.077).

### 4.2 Experiment 4d: Wheeler Test

The Wheeler hypothesis — that crystallization is detectable from a single correlation matrix snapshot — was tested via permutation testing at W=1 through W=10. Results: 8/9 metrics achieve statistical significance at W=1 (p < 0.01), including all four S1 spectral metrics, S3 PE1, S3 TCS, and baselines BL_mean_corr and BL_lam1. Only S2 CDR fails to reach significance. The hypothesis is confirmed, but the discriminative power is limited (AUC 0.53 at W=1).

### 4.3 LEFT_CENSORED Recovery

5 universes had structural damage within the first 60 steps (LEFT_CENSORED: europe t_S=16, x_energy_geopolitics t_S=46, us_sectors t_S=46, x_us_systemic t_S=21, x_brazil_vuln t_S=21). Using forward-only crossing detection with calibrated thresholds and persistence=3: 3/5 were recovered. Europe (lead=16 steps via SCR), us_sectors (lead=42 steps via SCR), x_us_systemic (lead=21 steps via PE1). Two remain undetectable: x_energy_geopolitics and x_brazil_vuln, likely reflecting exogenous forcing prior to internal crystallization.

### 4.4 Experiment 4b: Cross-Universe Transferability

Per-method LOUO transferability (W=20): TCS = 0.627, LZ = 0.626, D_MP_KL = 0.622, SCR = 0.609, BL_lam1 = 0.609. SIG methods exceed baselines by +0.018. TCS (Wasserstein topological distance) shows the best transferability among analytic methods.

### 4.5 Experiment 4e: CALM Attenuation


The central diagnostic finding. Every SIG method was evaluated under both CALM-dependent and CALM-free targets at W = 1, 5, 10, 20, 60. Results: CALM-free targets yield uniformly higher AUCs across all methods and windows. Mean delta = +0.053. Fraction of positive deltas: 100%.

Key comparisons at W=60: SCR rises from 0.636 (CALM-dep) to 0.739 (CALM-free). D_MP_KL from 0.655 to 0.701. LZ from 0.632 to 0.682. PE1 from 0.612 to 0.675. BL_lam1 from 0.636 to 0.739 (identical to SCR, confirming their mathematical equivalence as classifiers).

This establishes that CALM-based normalization systematically compresses geometric signal that is predictive of actual structural damage. The implication: CALM-dependent labels mark as NOMINAL steps that carry genuine pre-damage signal.

### 4.6 S5 Autoencoder: LSCC Discovery

The structural autoencoder (Section 2.4) was trained on all 21 universes and evaluated via LOUO with CALM-free targets. The S5 classifier head achieves LOUO AUC 0.733 +/- 0.103 (19 folds), substantially exceeding all analytic methods in LOUO: TCS 0.627, LZ 0.626, D_MP_KL 0.622, ensemble 0.528.

The LSCC (PC1 of the latent space) was then evaluated across 6 validation tests:

**T1 — Regime Separation:** PCA of the 8-dimensional latent space shows PC1 explains 79.9% of variance. Centroids: NOMINAL (PC1 = -4.5), PRE_DAMAGE (-1.8), DAMAGED (+10.4). The latent space organizes as a continuous gradient, not discrete clusters. Silhouette = 0.074 (low but consistent with gradient structure).

**T2 — Partial Correlation:** Max |r| between LSCC and any manual observable = 0.111 (PE1). Linear R-squared of all 6 observables predicting LSCC = 0.032. The LSCC captures 96.8% non-linear structure not present in any linear combination of SCR, lambda_1, D_MP_KL, LZ, PE1, or TCS.

**T3 — Incremental Value (fold-local PCA, no leakage):** LSCC alone achieves LOUO AUC = 0.942. All analytic methods combined = 0.634. Incremental value of LSCC above all analytics: +0.323. Adding any analytic to LSCC provides marginal improvement (0.942 to 0.960).

**T4 — Cross-Universe Stability:** Mean |cos(PC1_i, PC1_j)| = 0.664 across 21 universes. The dominant latent direction is partially aligned across universes, supporting interpretation as a "fingerprint" with shared but not universal orientation.


**T5 — Temporal Dynamics:** Among 14 universes with identifiable damage onset (t_S), 78.6% show positive LSCC drift (near-damage mean exceeds nominal mean) with mean drift = +1.114. The LSCC moves systematically toward the damage region before C_norm drops.

**T6 — CALM-Saturated Coverage:** In 6 universes where Layer 2 is blind (CALM_SATURATED, Theta_A = 0), LSCC achieves mean AUC = 0.911, compared to 0.955 in the 14 HIGH universes. The LSCC covers the Layer 2 blind spot with minimal performance degradation.

### 4.7 LSCC Robustness Tests

Five stress tests were conducted before claiming publication readiness:

**R1 — Seed Stability:** 5 seeds (42, 123, 7, 2026, 999). Mean LOUO AUC = 0.978 +/- 0.006. CV = 0.006. Verdict: STABLE.

**R2 — Latent Dimension Sensitivity:** dim=4: 0.939, dim=8: 0.974, dim=16: 0.973, dim=32: 0.953. Spread = 0.035. Verdict: ROBUST.

**R3 — Encoder-Only (no classifier):** With classifier: 0.974. Without: 0.611. The supervised classification head is essential for organizing the latent space around vulnerability. The LSCC is a learned supervised detector, not an unsupervised discovery. Verdict: PARTIAL.

**R4 — Temporal Split (train pre-2024, test 2024+):** AUC = 0.722 +/- 0.113 across 16 universes. The LSCC generalizes across time, not just across universes. Verdict: PASS.

**R5 — Artifact Check:** Real model: 0.974. Shuffled labels: 0.534. Random z: 0.528. Gap > 0.44. The signal is genuine, not a dataset artifact. Verdict: PASS.

Overall: 4/5 pass cleanly. The LSCC is robust for publication with the caveat that it is supervised (R3).

---

## 5. CALM Revision: Three Principles

The Experiment 4e finding leads to a formal revision of the CALM framework within Kappa:


**Principle 1:** CALM remains domain-agnostic and foundational for structural baselining. Layer 1 (Structural Capacity) is unchanged.

**Principle 2:** Geometric observability is not guaranteed by structural calmness. When CALM baseline periods exhibit geometrically saturated RQA statistics (sigma < GEO_SIGMA_MIN), Layer 2 is genuinely blind. Global reference z-scores are near-constant bias, not discriminative signal.

**Principle 3:** CALM-based normalization is layer-conditional rather than universally neutral. Layer 2 operates normally when CALM is geometrically variable (14/21 universes). In CALM_SATURATED cases (7/21), Layer 2 reports Theta_A = 0 with an informational flag, and detection is deferred to Layer 0 (SIG/LSCC).

Engine v5.8c implements these principles with a new geo_reliability level: CALM_SATURATED (informational, replacing the silent LOW of v5.7).

---

## 6. Dual-Horizon Prognostic Calibration

The v5.8c engine introduces dual-horizon hazard calibration, both passing all validation criteria:

**H=60 (Operational):** beta_0=-2.79, beta_F=+1.13, beta_rho=+0.005, beta_eta=+0.55, beta_A=+4.83. AUC=0.772, ECE=0.082, LOUO-CV=0.671. Signs stable (bootstrap + CV PASS).

**H=90 (Strategic):** beta_0=-2.25, beta_F=+1.02, beta_rho=+0.004, beta_eta=+0.54, beta_A=+4.42. AUC=0.733, ECE=0.089, LOUO-CV=0.645. Signs stable (bootstrap + CV PASS).

Theta_A remains the dominant predictor in both horizons (ablation delta_AUC = +0.209 for H=90). Alerts are driven by H=90 (conservative); H=60 provides faster operational assessment.

---

## 7. Multi-Layer Detection Architecture

The complete Kappa framework now operates with three complementary detection layers:


**Layer 0 — Structural Information Geometry (SIG/LSCC):**
Instantaneous triage + learned latent fingerprint. No CALM dependency. LOUO AUC 0.942. Covers CALM-saturated blind spots (AUC 0.911). Operates on the joint state trajectory. Best for: rapid screening, CALM-blind universes, pre-RQA early warning.

**Layer 2 — Attractor Geometry (Theta_A):**
Temporal geometric diagnosis via RQA z-score against CALM baseline. AUC 0.772 (H=60). Dominant predictor (beta_A = +4.83). Requires 60+ steps and functional CALM. Best for: calibrated hazard estimation in geometrically observable universes.

**Layer 3 — Prognostic Module:**
Dual-horizon calibrated hazard (H=60, H=90) combining kappa_F, rho_bar, eta_norm, and Theta_A. Produces P_collapse, T_half, and alert levels. Best for: operational risk communication and monitoring.

Each layer answers a different question: Layer 0 asks "is this state structurally anomalous?"; Layer 2 asks "is the geometry reconfiguring?"; Layer 3 asks "what is the probability of damage within H days?"

---

## 8. Limitations and Honest Boundaries

### 8.1 LSCC is Supervised

The R3 test shows that removing the classifier head drops AUC from 0.974 to 0.611. The LSCC is a supervised representation optimized for the CALM-free damage target. It should be interpreted as a robust learned detector, not as an unsupervised intrinsic state coordinate. The ontological claim is weaker than the predictive claim.

### 8.2 Temporal Split Degradation

LOUO AUC (0.942) exceeds temporal split AUC (0.722). This gap reflects the fact that LOUO allows the model to see future data from other universes, while temporal split restricts all training to pre-2024. The 0.722 is the more conservative and operationally relevant estimate.

### 8.3 Analytic Methods Remain Modest

No individual analytic SIG method reaches AUC 0.70 with CALM-dependent labels. The best (D_MP_KL at W=60: 0.655) is useful for screening but insufficient for decision support. The LSCC's strength comes from non-linear multivariate combination, not from any single metric.


### 8.4 Cross-Universe Stability is Partial

T4 shows mean cosine similarity of PC1 directions = 0.664. The LSCC captures a partially shared structure across universes, but each universe has its own variant. The term "coordinate" is aspirational; "fingerprint" is more accurate.

### 8.5 Sample Size

21 universes over 4 years of daily data. While the LOUO protocol provides some protection against overfitting, the universe count is limited. Additional domains (educational systems, neurological data) would strengthen generalization claims.

---

## 9. Conclusions

Kappa-SIG extends the Kappa Method v2.5 with three contributions:

1. **Instantaneous detection is possible but weak.** The Wheeler test confirms crystallization signal exists at W=1 (p < 0.01), but discriminative power is limited (AUC 0.53). Temporal analysis (Layer 2) remains far stronger where applicable.

2. **CALM attenuation is real and quantifiable.** The CALM normalization systematically suppresses +5.3% AUC of genuine pre-damage signal. The revision to layer-conditional CALM normalization (v5.8c) preserves the CALM as structural baseline while flagging CALM_SATURATED cases for Layer 0 coverage.

3. **The LSCC is a robust supervised latent detector.** It achieves LOUO AUC 0.942, generalizes temporally (0.722), is seed-stable (CV 0.006), dimension-robust (spread 0.035), and artifact-free. It does not collapse into manual observables (R-squared = 0.032), suggesting it captures distributed non-linear interactions among the Kappa state variables that no single metric recovers. The classifier head is essential (R3), establishing the LSCC as a trained predictor, not an autonomous discovery.

The multi-layer architecture (Layer 0 + Layer 2 + Layer 3) provides complementary coverage: no single layer dominates in all conditions, and each compensates for the others' blind spots.

---

## References

1. Ohio, D. (2025). Kappa Method v1. DOI: 10.5281/zenodo.18434598
2. Ohio, D. (2026). Kappa Method v2.5. DOI: 10.5281/zenodo.19339548
3. Ohio, D. (2025). Kappa-FIN. DOI: 10.5281/zenodo.18434598
4. Ohio, D. (2025). Kappa-LLM. DOI: 10.5281/zenodo.18883790
5. Ohio, D. (2025). Kappa-Radiante. DOI: 10.5281/zenodo.18940478

---

**License:** CC BY 4.0
**Repository:** github.com/aprimora-ai
**Contact:** odavidohio@gmail.com
