# The Kappa Method v2: Structural Capacity, Attractor Geometry, and Probabilistic Regime Forecasting

**David Ohio** — Independent Researcher
odavidohio@gmail.com

**Status:** Formal Proposal v2.5 (Final with Historical Validation and Transition Taxonomy) — March 2026
**Extends:** Kappa Method v1 (DOI: 10.5281/zenodo.18434598)
**Implementation:** engine_v5 (v5.4) — 7 fixes across 5 iterations, validated on 21 Sentinel universes, historical lead-time analysis on 10 universes

---

## Abstract

We propose Kappa Method v2, a framework for decomposing structural vulnerability in complex systems through three layers of increasing inferential scope. Layer 1 introduces Structural Capacity C(t) = Φ* − Φ(t), a definitional measure of remaining system resilience derived directly from accumulated structural damage. Layer 2 extends the framework by analyzing attractor geometry of the damage trajectory via delay embeddings, testing the hypothesis that dynamical reconfiguration precedes observable structural degradation. Layer 3 introduces a probabilistic hazard model combining structural capacity, depletion rate, rigidity, and geometric activation.

A key empirical finding is that structural calmness does not imply geometric neutrality: baseline periods selected via CALM exhibit saturated recurrence structure, requiring z-score normalization for meaningful geometric detection. This leads to the identification of a degenerate baseline regime, in which geometric transitions are unobservable despite maximal rigidity.

Across 21 financial system universes, Layer 1 produces stable and interpretable structural ordering. Layer 2 yields parameter-robust geometric signals in 15/19 measurable cases. Historical analysis across 10 universes, after disambiguating shock-driven spikes from structural ramp transitions, shows that in all valid ramp-type cases (2/2), geometric activation preceded structural degradation, with lead times of 35 and 269 days. The sample size remains limited, and further validation — including three ongoing prospective cases — is required.

We interpret geometric activation not as a direct predictor of collapse, but as a modulation of impact sensitivity: systems in crystallized regimes exhibit reduced absorptive capacity, amplifying the effect of subsequent shocks. Baseline comparison (§8.10) confirms that geometric activation is the sole pre-damage signal in valid cases (24–185 steps lead time), while diagnostic signals like κ_F provide zero anticipation despite high concurrent AUC. Observable independence testing (§8.11) demonstrates that the geometric signal persists across embeddings of Oh(t), η(t), and mean_corr(t) — observables entirely independent of Φ(t) — substantially mitigating the circularity concern.

These results establish a falsifiable framework for early detection of structural vulnerability, while explicitly delineating the limits of observability and inference.

---

## 1. Epistemological Architecture

### 1.1 Layered Predictive Ambition

> **The predictive ambition of Kappa v2 is layered, not uniform.**
>
> Core quantities define structural exhaustion independently of estimator choice. Geometric extensions test whether loss of adaptive degrees of freedom is detectable through attractor deformation. Prognostic modules translate structural fragility into conditional risk estimates, but remain inferential rather than axiomatic.

This declaration governs the entire paper. Each result is tagged with its epistemological status:

| Status | Meaning | Falsifiability |
|---|---|---|
| **Definitional** | Follows from v1 axioms by algebraic construction | Cannot be "wrong" — only inapplicable |
| **Theoretical** | Proposition with strong motivation from established theory | Falsifiable via geometric measurements on S(t) |
| **Inferential** | Statistical model requiring calibration and validation | Falsifiable via calibration stability, out-of-sample performance |
| **Implementation-dependent** | Specific algorithm choice for a theoretical quantity | Replaceable without changing the theory |

### 1.2 What v2 Does Not Claim

Kappa v2 does **not** predict events (wars, policy decisions, market crashes). It estimates **structural vulnerability windows** — periods during which the system's remaining capacity to absorb shocks is critically low. The distinction is fundamental: v2 says "the probability that *any* shock of magnitude ≥ X will cause regime transition is now Y%," not "a shock will arrive at time T."

### 1.3 Relationship to v1

Kappa v2 is a **strict superset** of v1. All v1 observables, regime classifications, and metrics remain unchanged. v2 adds new outputs without modifying any existing output. The engine_v4 pipeline operates identically; v2 is an additional analysis layer (engine_v5) consuming the v4 state trajectory.

### 1.4 Irreversibility Baseline Requirement

**Definition 1.4 (Baseline Insufficiency).** When engine_v4 produces Φ* = 0 or Φ* < ε_{Φ*}, the universe has not accumulated sufficient structural damage to estimate the irreversibility threshold. In this case:

- Layer 1 normalized quantities (C/Φ*, κ_F) are **not interpretable**
- The universe is classified as **BASELINE_INSUFFICIENT**
- All alert outputs are suppressed

This is not an error condition but a formal epistemological class: the method cannot speak about structural capacity when irreversibility has never been approached. Two sub-cases exist:

- **no_damage_history**: Φ* = 0 and max(Φ) ≈ 0 — the universe has never experienced significant stress
- **estimated_fallback**: Φ* = 0 but max(Φ) > 0 — damage occurred but was insufficient to identify the irreversibility knee; a conservative estimate Φ* = 2·max(Φ) is used, flagged as estimated

---

## 2. Layer 1 — Structural Capacity (Core)

**Epistemological status: Definitional.**

This layer introduces no new estimators. Every quantity is derived algebraically from objects already defined in v1.

### 2.1 Definition: Structural Capacity C(t)

$$C(t) = \Phi^* - \Phi(t)$$

where Φ* is the irreversibility threshold (v1 Proposition B.15) and Φ(t) is the accumulated structural damage (v1 §IV.2).

**Interpretation:** C(t) measures the remaining structural buffer — the distance between the system's current accumulated damage and the point of irreversibility.

**Properties (derived from v1 axioms):**

**Property 2.1 (Boundedness).** C(t) ∈ (−∞, Φ*]. When C(t) > 0, capacity remains. When C(t) ≤ 0, the system is in post-irreversibility regime.

**Property 2.2 (Asymmetric dynamics).** Damage accumulates additively: dΦ/dt|_{accumulation} = D(t). Dissipation is multiplicative: dΦ/dt|_{dissipation} = −(1−γ)Φ(t). Since γ ∈ (0.97, 0.99), the dissipation rate is small relative to accumulation during stress. This asymmetry is the structural origin of hysteresis and the formal reason why false recovery exists.

**Property 2.3 (Path dependence).** C(t) depends on the entire history {Oh(s) : s ≤ t}, not only on the current state. Two systems with identical Oh(t) at time *t* may have radically different C(t) if their histories differ.

### 2.2 Capacity Dynamics

$$\Delta C(t) = (1-\gamma)\Phi(t) - D(t)$$

This reveals C(t) as a competition between healing ((1−γ)Φ(t)) and damage (D(t) = max(0, Oh(t) − Oh_pre − δ)).

### 2.3 Capacity Depletion Rate

$$\rho(t) = D(t) - (1-\gamma)\Phi(t)$$

When ρ(t) > 0, capacity is being consumed. When ρ(t) < 0, capacity is regenerating.

**Smoothed depletion rate:** ρ̄(t) = α_ρ · ρ(t) + (1−α_ρ) · ρ̄(t−1), with α_ρ = 0.1 (fixed).

### 2.4 Time-to-Exhaustion Estimate

$$\hat{T}_{exhaust}(t) = \frac{C(t)}{\max(\bar{\rho}(t), \epsilon)}$$

defined only when ρ̄(t) > 0 (active depletion).

### 2.5 Log-Capacity Ratio (Fragility Index)

$$\kappa_F(t) = \ln\left(\frac{\Phi^*}{\max(C(t), \epsilon)}\right)$$

Capped at κ_{F,max} = 20.0 to prevent logit overflow in Layer 3.

**Properties:** κ_F = 0 when C(t) = Φ* (pristine). κ_F → κ_{F,max} as C(t) → 0.

### 2.6 False Recovery — Formal Definition

**Definition 2.6.** A system is in false recovery at time *t* if: (1) Oh(t) < 1.0, (2) C(t)/Φ* < θ_FR, and (3) the system was in Katashi within the preceding T_FR steps.

### 2.7 Layer 1 Summary

| Quantity | Formula | Parameters | Status |
|---|---|---|---|
| C(t) | Φ* − Φ(t) | None new | Definitional |
| ρ(t) | D(t) − (1−γ)Φ(t) | None new | Definitional |
| ρ̄(t) | EMA of ρ | α_ρ = 0.1 (fixed) | Convention-bound |
| T̂_exhaust(t) | C(t)/ρ̄(t) | None new | Convention-bound |
| κ_F(t) | ln(Φ*/C(t)), capped at 20.0 | None new | Definitional |
| False Recovery | Three conditions (§2.6) | θ_FR, T_FR (CALM-derived) | Convention-bound |

**Total new free parameters in Layer 1: zero.**

---

## 3. Layer 2 — Attractor Geometry (Geometric Extension)

**Epistemological status: Theoretical — testable proposition with strong motivation.**

### 3.1 The Central Proposition

**Proposition 3.1 (Attractor Deformation Precedes Transition).** The attractor of the structural damage trajectory Φ(t) undergoes measurable geometric reconfiguration — including gradual deformation or dynamical crystallization — before a regime transition manifests in the instantaneous observables Oh(t) or Φ(t).

This proposition is independent of any specific estimator. Whether we measure attractor deformation via RQA, correlation dimension, Lyapunov exponents, or persistent homology is an implementation choice.

### 3.2 Delay Embedding and Attractor Reconstruction

By Takens' Theorem (1981):

$$\mathbf{y}(t) = [\Phi(t), \Phi(t-\tau_d), \Phi(t-2\tau_d), \ldots, \Phi(t-(m-1)\tau_d)] \in \mathbb{R}^m$$

where τ_d is estimated via the first zero-crossing of autocorrelation and m = 3 (conservative default).

**On the choice of Φ(t) as embedding observable:** A potential objection is circularity — Layer 2 derives geometry from Φ(t), which is also the basis of Layer 1's capacity measure C(t). However, the two layers extract fundamentally different information from Φ(t). Layer 1 measures the *level* of Φ(t) relative to Φ* (how much damage has accumulated). Layer 2 measures the *temporal organization* of the Φ(t) trajectory (how the system visits and revisits states in phase space). A constant Φ = 0.5 and an oscillating Φ averaging 0.5 produce identical C(t) but radically different RQA signatures. The information is complementary, not redundant.

### 3.3 Geometric Indicators

On the reconstructed attractor, computed over rolling windows W_a:

| Measure | Definition | Transition Signal |
|---|---|---|
| **DET** — Determinism | Fraction of recurrence points in diagonal lines | ↑ before collapse (rigidity) |
| **LAM** — Laminarity | Fraction of recurrence points in vertical lines | ↑ before collapse (stasis) |
| **TT** — Trapping Time | Mean vertical line length | ↑ before collapse (trapped states) |

The expected RQA signature of approaching collapse includes increases in DET, LAM, and/or TT, with empirical results in the Sentinel corpus indicating that TT is the dominant contributor in crystallized regimes. This is the dynamical-systems formalization of the Law of Katashi.

Secondary indicators D₂ (correlation dimension) and λ₁ (maximum Lyapunov exponent) are applicable when n ≥ 180 with convergence tests.

### 3.4 Data Sufficiency Protocol

| Condition | Available Indicators |
|---|---|
| n < 60 | Layer 2 **not applied** |
| 60 ≤ n < 180 | RQA only |
| n ≥ 180 | RQA + D₂ + λ₁ (with convergence tests) |

### 3.5 Attractor Transition Score Θ_A(t)

$$\Theta_A(t) = w_D \cdot \widetilde{DET}(t) + w_L \cdot \widetilde{LAM}(t) + w_T \cdot \widetilde{TT}(t)$$

where w_D = 0.40, w_L = 0.35, w_T = 0.25 (fixed), and each tilde denotes CALM-normalized values (see §3.6). Only positive deviations from CALM are counted (max(·, 0)). Θ_A is capped at Θ_{A,max} = 5.0.

### 3.6 CALM Normalization for Geometric Observables

**Critical finding (empirical, §8.3):** Structural calmness does not imply geometric neutrality. CALM selects regions of minimal structural variation where the reconstructed attractor is maximally regular — producing near-saturated RQA measures (e.g., LAM_CALM → 1.0). This makes CALM optimal as a level baseline (Layer 1) but inadequate as a direct geometric baseline (Layer 2).

**Resolution: z-score normalization for all RQA measures.**

For each RQA measure X ∈ {DET, LAM, TT}:

$$\widetilde{X}(t) = \frac{X(t) - \mu_{X}^{CALM}}{\max(\sigma_{X}^{CALM}, \sigma_{floor})}$$

where σ_{floor} = 0.01 prevents division by near-zero variance.

This replaces the original ratio normalization (X − X_CALM)/(1 − X_CALM), which produced explosive values when X_CALM was near saturation.

**Geometry reliability gate:** If σ of DET or LAM during CALM < σ_{geo,min} = 0.005, the CALM period lacks sufficient variability for meaningful z-scores, and Θ_A is suppressed (set to zero). This gate is based on distributional adequacy, not on the level of the measure.

### 3.7 Dual Operating Mode of Layer 2

**Empirical observation (§8.5):** Layer 2 operates in two distinct modes:

**Gradual mode:** In universes with intermediate dynamics (moderate CALM variability, σ_CALM > σ_{geo,min}), Θ_A produces proportional, parameter-robust values. Sensitivity analysis (§8.5) shows Θ_A varies by ±0.05 across ε_q ∈ {0.10, 0.20, 0.30} — strong evidence that the signal reflects an intrinsic property of the dynamics, not an artifact of the recurrence threshold.

**Crystallized mode:** In certain cross-layer universes, TT saturates at the window maximum (20.0) while DET and LAM remain near-neutral. The z-score of TT becomes extremely large (> 100σ), driving Θ_A to the cap. This indicates a regime consistent with dynamic crystallization: the system's recurrence structure has collapsed to a near-fixed point, and the attractor has lost transitional dynamics. The signal is binary (crystallized vs. not) rather than gradual, and is robust to ε_q variation.

The distinction between these modes is a property of the system, not the method.

### 3.8 Layer 2 Summary

| Quantity | Status | Requirement |
|---|---|---|
| Proposition 3.1 | Theoretical | Testable |
| RQA (DET, LAM, TT) | Implementation-dependent | n ≥ 60 |
| z-score normalization | Implementation-dependent | σ_CALM > σ_{geo,min} |
| Θ_A(t) | Theoretical + implementation | Capped at 5.0 |
| Geometry reliability gate | Implementation-dependent | σ-based |

### 3.9 Observability Limits and Degenerate Baselines

**Critical distinction (empirical, §8.9):** Θ_A measures *change in geometric organization*, not geometric organization itself. A system that is already maximally rigid at baseline produces Θ_A ≈ 0 — not because it is geometrically neutral, but because there is no transition to detect.

This leads to two distinct types of geometric rigidity:

**Dynamic rigidity (detectable):** The system transitions from a fluid to a rigid geometric regime during the observation period. Θ_A rises as the z-score captures departure from the CALM distribution. This is the signal Layer 2 is designed to detect.

**Static rigidity (undetectable):** The system is already in a maximally rigid geometric regime at baseline. The delay embedding of Φ(t) collapses to a near-constant trajectory, producing fully saturated recurrence matrices (DET ≈ 1, LAM = 1, TT = max) with σ_CALM = 0. No departure from baseline can be measured because the baseline itself is already at the geometric ceiling.

**Formal definition (Degenerate Baseline).** When σ_{X}^{CALM} < σ_{geo,min} for any RQA measure X ∈ {DET, LAM}, the CALM period is classified as geometrically degenerate. In such regimes, geometric observables are uninformative, and the absence of geometric precursor cannot be interpreted as evidence against Proposition 3.1.

**Implication:** The absence of geometric precursor is not evidence of absence of geometric transition — it may indicate that the transition occurred outside the observable window or that the system was already in a saturated geometric regime at baseline.

### 3.10 Disambiguating Shock-Driven vs Structural Geometric Transitions

**Empirical finding (§8.8):** Not all geometric activations carry the same informational content. Historical analysis reveals two distinct activation modes with fundamentally different interpretive status:

**Type 1 — Shock-driven spike.** Θ_A appears abruptly (often at the first computable step after warm-up) and dissipates within days. The geometric signal and structural damage are concurrent responses to the same exogenous shock. The spike does not represent anticipation; it represents the system's immediate geometric response to forcing. Such signals should be classified as AMBIGUOUS for lead-time analysis.

**Formal criterion:** A geometric activation is classified as shock-driven if:
1. Θ_A crosses the detection threshold within the first 5 steps of its computational availability, OR
2. Θ_A reverts below 0.1 within 20 steps of crossing the threshold

**Type 2 — Structural ramp.** Θ_A rises gradually over multiple weeks, reflecting progressive dynamical reorganization. The signal builds through sustained change in recurrence structure — not from a single event but from cumulative geometric transformation. Such signals persist for months and are consistent with genuine structural pre-instability.

**Formal criterion:** A geometric activation is classified as structural if:
1. Θ_A crosses the detection threshold via monotonic or near-monotonic growth over ≥ 15 steps, AND
2. Θ_A remains above the threshold for ≥ 30 steps after crossing

**Empirical basis:** Asia-Pacific Episode 1 (May 2022) exhibited Type 1 behavior: Θ_A spiked to 0.96 at step 60 and dropped below 0.1 by step 77, concurrent with C/Φ* crashing to −0.03 from the same Ukraine shock. Asia-Pacific Episode 2 (November 2025) exhibited Type 2 behavior: Θ_A rose from 0.0 → 0.11 → 0.32 → 0.50 → 0.84 → 1.18 → 1.84 over 27 days and has remained stable at 1.844 for over 3 months.

**Significance:** Only Type 2 (ramp) activations should be used as evidence for Proposition 3.1 and for lead-time measurement. Type 1 (spike) activations, while real geometric events, do not demonstrate anticipatory capacity because the geometric signal and structural damage share a common cause. This distinction provides a principled filter against false attribution of predictive power.

**Qualitative law:** Geometric transitions that carry structural information exhibit temporal continuity — they are built, not triggered. This may generalize beyond the present domain: in any system where Kappa v2 is applied, only ramp-like geometric activations should be interpreted as evidence of dynamical reorganization preceding structural change.

---

## 4. Layer 3 — Structural Hazard Function (Prognostic Module)

**Epistemological status: Inferential — requires calibration and empirical validation.**

### 4.1 Hazard Model

The instantaneous structural hazard via discrete logit link:

$$\text{logit}(h(t)) = \beta_0 + \beta_F \cdot \kappa_F(t) + \beta_\rho \cdot \bar{\rho}^+(t) + \beta_\eta \cdot \tilde{\eta}(t) + \beta_A \cdot \Theta_A(t)$$

$$h(t) = \text{sigmoid}(\text{logit}(h(t)))$$

where logit is clipped to [−20, 20] for numerical stability.

**Nested model architecture:**

| Model | Covariates | Purpose |
|---|---|---|
| A (Core-only) | β_0, β_F·κ_F, β_ρ·ρ̄⁺ | Baseline: does capacity predict? |
| B (Core+Rigidity) | A + β_η·η̃ | Does rigidity add value? |
| C (Full) | B + β_A·Θ_A | Does geometry add value? |

Placeholder β coefficients: β_0 = −6.0, β_F = 1.0, β_ρ = 0.5, β_η = 0.3, β_A = 0.5. These require calibration from historical corpus (§7.3). The current parameterization is illustrative and intended for structural testing of the framework, not for quantitative forecasting.

### 4.2 Rigidity Observable — Temporal Smoothing

**Empirical finding (§8.4):** Raw rigidity η(t) may spike to extreme values (e.g., η = 20.0) in a single step when Forman-Ricci curvature hits zero (η = 1/(|curv| + η_floor)). These spikes are estimator artifacts, not structural rigidity.

**Resolution:** Smooth η before normalization:

$$\bar{\eta}(t) = \alpha_\eta \cdot \eta(t) + (1-\alpha_\eta) \cdot \bar{\eta}(t-1)$$

with α_η = 0.1, then normalize: η̃(t) = η̄(t) / η̄_CALM, capped at 5.0.

This parallels the treatment of ρ via ρ̄ — observables entering the prognostic module should reflect structural tendencies, not estimator noise.

### 4.3 Collapse Probability and Half-Life

$$P_{collapse}(t, \Delta) \approx 1 - (1 - h(t))^\Delta$$

$$T_{1/2}(t) = \frac{\ln(0.5)}{\ln(1 - h(t))}$$

### 4.4 Alert Logic — Multi-Source with Capacity Confirmation

| Level | Condition | Rationale |
|---|---|---|
| **BASELINE_INSUFFICIENT** | Φ* not estimable | Method cannot speak |
| **NOMINAL** | C/Φ* ≥ 0.50 AND h < h_warn | Healthy |
| **WATCH** | C/Φ* < 0.50 OR κ_F > κ_{F,warn} | Capacity-driven |
| **WARNING** | (h ≥ h_warn AND C/Φ* < 0.70) OR T½ < T_warn | Hazard + capacity confirmation |
| **EMERGENCY** | C < 0 OR (h ≥ h_emerg AND C/Φ* < 0.50) | Post-irreversibility or confirmed crisis |

**Design principle:** Hazard alone cannot escalate to EMERGENCY without capacity confirmation. This prevents geometric or rigidity signals from generating false alarms in universes with intact capacity. The alert system is capacity-first, with hazard refining severity.

### 4.5 Honest Limitations of Layer 3

1. Cannot predict exogenous shocks — only structural vulnerability to them
2. β coefficients are placeholder values requiring calibration
3. Probabilities are conditional on model specification — calibrated estimates, not physical constants
4. In conservative reliability mode (geo_reliability = LOW), Layer 3 reverts to a capacity-driven prognostic model (Model A)

---

## 5. Predictive Time Windows and Lead-Time Structure

### 5.1 Motivation

The introduction of geometric observables (Layer 2) suggests the existence of an earlier stage of instability, preceding structural degradation (Layer 1). This motivates formal definitions of predictive time windows.

### 5.2 Layered Temporal Structure

Three temporal regimes:

**Geometric regime (pre-instability):** Θ_A(t) ≫ 0 with C(t) ≈ Φ*. Detectable deformation of the reconstructed attractor while structural capacity remains intact.

**Structural regime (fragility accumulation):** C(t) decreasing and/or κ_F(t) increasing. Accumulation of structural damage and loss of resilience.

**Critical regime (irreversibility):** C(t) ≤ 0. Post-irreversibility dynamics.

### 5.3 Lead Time Definitions

**Geometric activation time t_G:** Earliest time such that Θ_A(t) > θ_G for at least p_G consecutive steps, with geo_reliability = HIGH.

**Structural activation time t_S:** Earliest time such that C(t) < θ_C · Φ* or κ_F(t) > θ_F.

**Critical transition time t_C:** First time C(t_C) ≤ 0.

**Geometric-to-structural lead time:** ΔT_{G→S} = t_S − t_G

**Geometric-to-critical lead time:** ΔT_{G→C} = t_C − t_G

If consistently ΔT_{G→S} > 0 and ΔT_{G→C} > 0, Layer 2 provides predictive advantage over structural observables alone. This does not imply prediction of the triggering event, but early detection of loss of dynamical stability preceding structural degradation.

### 5.4 Epistemological Status

Lead times are **empirical quantities**, not axiomatic. t_S and t_C derive from definitional observables (Layer 1). t_G depends on geometric estimation and reliability gating (Layer 2). Predictive windows are measured properties of the system, not imposed constructs.

In the current version, these predictive windows are formally defined but only partially instantiated empirically. Their statistical properties (mean lead time, variance, and false positive rate) remain to be established through the validation protocol described in §7.

---

## 6. Theoretical Foundations

| Tradition | Key Reference | Layer |
|---|---|---|
| **Reliability Engineering** — cumulative damage | Miner (1945) | 1 |
| **Survival Analysis** — time-varying hazard | Cox (1972) | 3 |
| **Dynamical Systems** — attractor reconstruction | Takens (1981) | 2 |
| **Critical Transitions** — early warning signals | Scheffer et al. (2009) | 2, 3 |
| **Recurrence Analysis** — regime detection | Marwan et al. (2007) | 2 |

---

## 7. Experimental Validation Protocol

### 7.1 Experiment 1 — False Recovery Stress Test

**Question:** Does C(t) distinguish real recovery from false recovery?

Classify historical episodes into Class R (real recovery: Φ dissipates) and Class F (false recovery: system re-enters Katashi). Measure separation of C(t)/Φ* distributions.

### 7.2 Experiment 2 — Incremental Value Chain

**Question:** Does each layer add predictive value over simpler alternatives?

Four competing models: Model 0 (v1 baseline: Oh > threshold AND Φ > Φ_c), Model A (Core only), Model B (Core + Geometry), Model C (Full). Additionally, three naive baselines must be tested: (i) Φ slope over trailing window, (ii) Oh moving average threshold, (iii) random classifier. Metrics: Brier score, log-loss, dynamic AUC. **Required result:** Model A must outperform all naive baselines. If it does not, the v2 premise fails regardless of Layer 2/3 performance.

### 7.3 Experiment 3 — Calibration and Stability of β's

Leave-one-episode-out, bootstrap (1000×). Required: no β changes sign, ECE < 0.15.

### 7.4 Experiment 4 — Surrogate and Ablation

Remove one covariate at a time. Compare against null models (Oh-only, Φ-only, random).

### 7.5 Experiment 5 — Prospective Backtesting

Forward-only simulation. Measure correct alert rate, lead time distribution, false alarm rate.

---

## 8. Preliminary Empirical Findings

**Corpus:** 21 Sentinel financial universes (§A.1), spanning global macro, regional, sectoral, thematic, and cross-layer composites, from February 2022 to March 2026 (~1040 trading days each).

**Implementation:** engine_v5 (v5.4), iteratively refined across 5 versions with 7 bug fixes. Full fix history in §A.2.

### 8.1 Layer 1 — Strongly Corroborated

Layer 1 produces structurally plausible ordering across all 21 universes:

| Class | Universes | C/Φ* Range | Alert |
|---|---|---|---|
| Post-irreversibility | commodities, energy, europe, x_energy_geopolitics, x_europe_vuln | −101.3 to −0.47 | EMERGENCY |
| Partially consumed | financials (0.245), us_sectors (0.426) | 0.24 to 0.43 | WARNING |
| Intact | 12 universes (mena through ai_ecosystem) | 0.86 to 1.00 | NOMINAL |
| Baseline insufficient | latam, x_commodity_chain | N/A | BASELINE_INSUFFICIENT |

This ordering is stable across all engine iterations (v5.0 through v5.4) — implementation fixes to Layers 2 and 3 did not alter Layer 1 results. The Core resisted interventions without losing discriminative power.

### 8.2 Baseline Insufficiency — Formal Handling

Two universes (latam, x_commodity_chain) produced Φ* = 0 from engine_v4. These are separated from the measured universe set as BASELINE_INSUFFICIENT, with Layer 1 normalized outputs suppressed. This treatment was validated as epistemologically superior to assigning default values.

### 8.3 Layer 2 — CALM Normalization Incompatibility and Resolution

**Discovery:** The original ratio normalization for DET and LAM — (X − X_CALM)/(1 − X_CALM) — produced universally LOW geometry reliability (0/19 HIGH) because CALM selects maximally stable periods where LAM_CALM → 1.0, making the denominator (1 − LAM_CALM) → 0.

**Resolution (FIX-7):** Replacing ratio normalization with z-score normalization for all RQA measures produced 15/19 HIGH geometry reliability. The key insight: CALM is optimal as a structural filter (identifying stable periods) but inadequate as a direct normalizer for geometric measurement. The method now uses CALM to identify the baseline period and extract distributional statistics (μ, σ), but normalizes via z-score rather than ratio against ceiling.

**Implication for other Kappa domains:** This finding likely applies wherever CALM is used as geometric baseline — including Kappa-LLM attention analysis and Kappa-Education dropout prediction. This distinction suggests that CALM functions as a structural baseline rather than a geometric neutral reference, a separation that may generalize to other Kappa domains.

### 8.4 Rigidity Spike Resolution

**Discovery:** x_energy_tech showed P30 = 92.3% with C/Φ* = 0.998 (capacity nearly pristine). Hazard decomposition revealed that a single-day η spike to 20.0 (Forman-Ricci curvature = 0 edge case) accounted for 100% of the hazard elevation.

**Resolution (FIX-6):** Temporal smoothing of η via EMA (α = 0.1) before normalization, consistent with the treatment of ρ. After fix: x_energy_tech P30 = 11.1%, NOMINAL. All 21 universes clean.

**Methodological note for paper:** Raw rigidity estimates may exhibit one-step spikes induced by near-zero discrete curvature. Prognostic inference uses temporally smoothed η̄(t), not instantaneous η(t).

### 8.5 Epsilon Sensitivity Analysis — Layer 2 Parameter Robustness

**Design:** All 21 universes tested across ε_q ∈ {0.10, 0.20, 0.30}.

**Result:** For intermediate-dynamics universes, Θ_A varies by ±0.05 across epsilon values:

| Universe | ε=0.10 | ε=0.20 | ε=0.30 |
|---|---|---|---|
| china_property | 0.42 | 0.39 | 0.41 |
| global_macro | 0.41 | 0.35 | 0.45 |
| x_energy_tech | 0.33 | 0.35 | 0.36 |
| asia_pacific | 1.84 | 1.79 | 1.86 |
| ai_ecosystem | 0.43 | 0.41 | 0.45 |

For crystallized universes (x_us_systemic, x_brazil_vuln), Θ_A = 5.00 (cap) at all epsilon values. This confirms that the saturation is a system property, not a threshold artifact.

Higher epsilon does not improve geometry reliability (15 HIGH → 13 HIGH). Default ε_q = 0.10 is retained.

**Significance:** The geometric signal cannot be dismissed as threshold-dependent. Layer 2 is parameter-robust. This robustness indicates that the geometric signal is not a consequence of threshold selection but reflects an intrinsic property of the underlying dynamics.

### 8.6 Geometric Instability without Structural Exhaustion

Three universes exhibit a novel pattern:

| Universe | C/Φ* | κ_F | Θ_A | P30 | Onset |
|---|---|---|---|---|---|
| x_us_systemic | 1.000 | 0.000 | 5.00 | 88.8% | Jan 2024 |
| x_brazil_vuln | 1.000 | 0.000 | 5.00 | 62.8% | May 2025 |
| asia_pacific | 1.000 | 0.000 | 1.84 | 23.6% | Dec 2025 |

**Formal characterization:** C/Φ* ≈ 1, κ_F ≈ 0, Θ_A ≫ 0.

**Temporal analysis:** The geometric signal is driven by TT (Trapping Time) saturation — the recurrence structure suggests that the attractor has transitioned from gradual deformation into a collapsed or near-fixed-point regime. In x_us_systemic, TT jumped from 2.9 to 20.0 in January 2024 and has remained saturated for 26 months. In x_brazil_vuln, onset was May 2025 (10 months). In asia_pacific, onset was December 2025 (3 months).

**Interpretation:** These universes have intact structural capacity but frozen dynamics. The recurrence structure suggests that the attractor has transitioned into a collapsed or near-fixed-point regime rather than undergoing gradual deformation. If the proposed ordering between geometric instability and structural degradation holds, these universes would be expected to eventually show rising Oh, accumulating Φ, and declining C.

**Conceptual refinement:** Geometric activation (Θ_A) does not directly induce structural damage; rather, it modulates the system's response to subsequent forcing, effectively increasing its impact sensitivity. In crystallized regimes, the system behaves as if its absorptive capacity has been reduced, causing shocks of moderate intensity to produce disproportionately large structural effects. The variation in observed lead times (3 to 269 days in §8.8) is consistent with this interpretation: ΔT_{G→S} depends not only on the system's geometric state but on the intensity and timing of subsequent shocks.

x_brazil_vuln shows the earliest signs: Oh reached 0.786 with Katashi days appearing in March 2026, consistent with approaching structural activation. This would represent the first empirical measurement of ΔT_{G→S} (§5.3).

**Caution:** 26 months of crystallization without structural damage in x_us_systemic may indicate a false positive (resilient rigidity) rather than impending transition. Validation requires continued monitoring.

### 8.7 Current Operational Status

| Layer | Status | Evidence |
|---|---|---|
| Layer 1 | **Strongly corroborated** operationally | Stable ordering across 5 iterations, 21 universes |
| Layer 2 | **Operationally active**, parameter-robust in intermediate universes, crystallized in extreme cross-layer universes | 15/19 HIGH reliability, ε_q sensitivity confirms |
| Layer 3 | **Operationally useful in Core-driven mode** (Model A); full mode (Model C) produces novel signals pending validation | Zero false positives after all fixes |

Probability outputs from Layer 3 should not yet be interpreted as calibrated risk estimates, as β parameters remain uncalibrated.

In conservative reliability mode (geo_reliability = LOW), Layer 3 reverts to a capacity-driven prognostic model, which already produces operationally useful output.

### 8.8 Historical Lead-Time Validation

**Design:** For each universe that experienced structural damage (t_S identifiable), measure whether geometric activation (t_G) preceded it. 10 universes analyzed, spanning EMERGENCY, WARNING, and anomalous cases.

**Definitions used:**
- t_G: First time Θ_A > 0.5 sustained for ≥ 5 consecutive steps, with geo_reliability = HIGH
- t_S: First time C/Φ* < 0.90
- t_C: First time C/Φ* < 0 (post-irreversibility)

**Methodological constraint:** The RQA computation requires a minimum warm-up period of 60 steps (the rolling window). Θ_A is structurally undefined before step 60. Cases where t_S < 60 are **left-censored** — geometric detection was impossible by construction.

**Results:**

| Class | N | Universes | ΔT_{G→S} |
|---|---|---|---|
| GEO_PRECEDED | 2 | commodities (269d), energy (35d) | 35–269 days |
| AMBIGUOUS (shock-driven spike) | 1 | asia_pacific Ep.1 (3d, concurrent shock — §3.10 Type 1) | Not interpretable as lead time |
| LEFT_CENSORED | 5 | europe, x_energy_geopolitics, us_sectors, x_us_systemic, x_brazil_vuln | t_S within warm-up |
| ALREADY_CRYSTALLIZED | 2 | x_europe_vuln, financials | Degenerate baseline (§3.9) |
| Genuinely NO_PRECURSOR | 0 | — | — |

**Key finding:** In all cases where structural activation occurred after the minimum data sufficiency period for Layer 2, and where the geometric signal exhibited ramp-like (Type 2) rather than spike-driven (Type 1) behavior, geometric activation preceded structural degradation (2/2 valid cases). Lead times were 35 and 269 calendar days.

Asia-Pacific Episode 1 (3 days) was reclassified as AMBIGUOUS following audit (§3.10): the geometric spike at step 60 and structural damage at step 63 were concurrent responses to the Ukraine invasion shock, not a case of geometric anticipation. Critically, the same universe later exhibited a clean Type 2 ramp (Episode 2, November 2025), which constitutes a genuine prospective test case.

The 5 left-censored cases all experienced structural damage during the February–May 2022 period (Ukraine invasion shock), within the first 60 steps of the data window. This is not a failure of Layer 2 but an impossibility of measurement — analogous to attempting to detect a signal before the detector is turned on.

The 2 ALREADY_CRYSTALLIZED cases (§8.9) represent a distinct epistemological class where geometric measurement was uninformative, not absent.

**Interpretation:** After applying the spike/ramp disambiguation criterion (§3.10), these results are consistent with Claim E (t_G < t_S) but the valid sample is small (n=2). No confidence intervals or false positive rates can be computed from this sample. The finding should be treated as directionally supportive rather than statistically established. Three prospective cases (asia_pacific Episode 2, x_brazil_vuln, x_us_systemic) are under active monitoring and represent the strongest near-term opportunity for out-of-sample validation. Extended data windows or application to additional crisis episodes could resolve the left-censored cases and increase statistical power in future work.

### 8.9 Degenerate Baselines — The ALREADY_CRYSTALLIZED Class

**Discovery:** x_europe_vuln and financials showed no geometric precursor (Θ_A = 0 throughout). Investigation revealed that both universes had **fully saturated recurrence matrices during the entire CALM period**:

| Universe | LAM_CALM | σ_LAM_CALM | TT_CALM | σ_TT_CALM | % days LAM=1 |
|---|---|---|---|---|---|
| x_europe_vuln | 1.000 | 0.000 | 20.0 | 0.0 | 100% |
| financials | 1.000 | ~0.018 | 20.0 | ~5.0 | 95% |

The structural damage trajectory Φ(t) was near-constant during CALM (range ≈ 0), producing delay embeddings that collapsed to a single point. The resulting recurrence matrices were completely filled, leaving no room for z-score departure.

**Counterintuitive finding:** When structural damage began and Φ(t) gained variance, TT *decreased* — the system became geometrically *less* rigid during crisis:

| Universe | TT during CALM | TT during damage |
|---|---|---|
| x_europe_vuln | 20.0 (100% saturated) | 13.1 (58% saturated) |
| financials | 20.0 (88% saturated) | 6.9 (20% saturated) |

**Interpretation:** Structural damage introduces dynamical variation into the Φ(t) trajectory, breaking the recurrence saturation. The system de-crystallizes when disturbed. This is consistent with the conceptual distinction between static rigidity (maximally ordered, no variation to detect) and dynamic rigidity (increasing order, detectable as transition).

This finding refines Proposition 3.1: Layer 2 detects *transitions toward* geometric rigidity, not rigidity itself. A system that has never left the maximally rigid state provides no geometric transition to detect. This is an observability limit of the method, not a failure of the theoretical proposition.

### 8.10 Baseline Comparison — Event-Proximity vs Temporal Precedence

**Experiment 2** tests whether Θ_A adds value over simpler signals. Two distinct questions are separated by design, because early warning signals cannot be evaluated via concurrent classification metrics.

**Part A (Event-proximity discrimination under pre-event censoring):** AUC-ROC with target "will C/Φ* cross below 0.90 for the first time within H days?" All timesteps after the first structural activation are excluded (post-event censoring) to prevent tautological dominance of diagnostic signals. Only GEO_PRECEDED universes included. Taxonomy labels are audit-derived from prior episode analysis, not inferred by the experiment.

**Part A results (pooled, pre-event censored, GEO_PRECEDED only):**

| Signal | AUC H=30 | AUC H=60 | AUC H=90 |
|---|---|---|---|
| Θ_A | 0.722 | 0.643 | 0.605 |
| Oh_EMA | 0.674 | 0.759 | 0.682 |
| κ_F | 0.278 | 0.357 | 0.395 |
| Random | 0.559 | 0.555 | 0.531 |

Under pre-event censoring, κ_F falls *below random* (AUC 0.28–0.40). This confirms that κ_F's apparent dominance in uncensored tests is tautological — it rises when C has already fallen, making it a concurrent diagnostic rather than an anticipatory signal. Θ_A shows moderate discriminative ability (AUC 0.60–0.72), indicating partial correlation with event proximity. Oh_EMA is competitive at longer horizons (AUC 0.76 at H=60), consistent with Oh being a direct pressure indicator.

**Part B (Temporal precedence):** For each universe, which signal activated first before structural damage? Activation thresholds: Θ_A > 0.5 sustained 5 days, κ_F > 0.5, Oh_EMA > 0.8 sustained 5 days.

| Universe | Θ_A lead | κ_F lead | Oh_EMA lead |
|---|---|---|---|
| commodities | **185 steps** | 0 | 0 |
| energy | **24 steps** | 0 | 0 |

Θ_A is the **only** signal that activated before structural damage in both valid cases. κ_F, Oh_EMA, and Φ_slope provided zero lead time — they activated at or after the moment of damage.

**Threshold sensitivity (Part D):** The ordering (Θ_A before κ_F) was tested across 9 configurations of threshold (0.3, 0.5, 0.7) and persistence (3, 5, 7 days). Result: 6/9 configurations preserve the ordering. The 3 failures occur at Θ_A = 0.7 (commodities does not reach this threshold), indicating the default 0.5 is within the robust range.

**Interpretation:** Θ_A and κ_F answer fundamentally different questions. κ_F measures "are we in danger now?" — excellent concurrent diagnosis (but AUC < 0.40 under censoring, zero lead time). Θ_A measures "is the system becoming vulnerable?" — moderate concurrent AUC (0.60–0.72) but the sole pre-damage signal in valid cases (24–185 steps). The low AUC of Θ_A relative to Oh_EMA is not a failure — it is a structural consequence of measuring a *condition* (impact sensitivity) rather than an *event* (pressure).

> Θ_A opens the window. κ_F confirms when damage arrives.

### 8.11 Observable Independence — Resolving the Circularity Concern

**Problem (Limitation L4):** Both Layer 1 (C = Φ* − Φ) and Layer 2 (attractor geometry of Φ(t)) derive from the same observable. A reviewer could object that the geometric signal is an artifact of embedding the damage accumulator into itself.

**Method:** Recompute the full Layer 2 pipeline (delay embedding, RQA, z-score normalization, Θ_A) using three observables **completely independent of Φ(t)**:

- **Oh(t):** The Ohio Number — structural coherence of the correlation topology
- **η(t):** Viscosity — curvature dynamics of the topological landscape
- **mean_corr(t):** Mean pairwise correlation — aggregate correlation structure

If the geometric precedence signal persists across these independent embeddings, it cannot be attributed to Φ-specific dynamics.

**Results (GEO_PRECEDED universes):**

**Commodities (t_S = step 245):**

| Embedding | t_G | Lead time | Independent of Φ? |
|---|---|---|---|
| Φ(t) | 60 | 185 steps | — |
| Oh(t) | 233 | **12 steps** | Yes |
| η(t) | 63 | **182 steps** | Yes |
| mean_corr(t) | 60 | **185 steps** | Yes |

**Energy (t_S = step 84):**

| Embedding | t_G | Lead time | Independent of Φ? |
|---|---|---|---|
| Φ(t) | 60 | 24 steps | — |
| Oh(t) | 60 | **24 steps** | Yes |
| η(t) | 60 | **24 steps** | Yes |
| mean_corr(t) | 69 | **15 steps** | Yes |

**Asia-Pacific (t_S = step 63, AMBIGUOUS):**

| Embedding | t_G | Lead time | Independent of Φ? |
|---|---|---|---|
| Φ(t) | 60 | 3 steps | — |
| Oh(t) | 103 | no lead | Yes |
| η(t) | 125 | no lead | Yes |
| mean_corr(t) | 60 | **3 steps** | Yes |

**Summary:** 7 out of 9 independent-observable embeddings show geometric activation before structural damage. In energy, all four observables activate simultaneously (t_G = step 60), suggesting the entire system reorganized at once. In commodities, η and mean_corr activate early (steps 60–63) while Oh activates later (step 233), suggesting a temporal cascade: viscosity and correlation structure change first, then topological coherence responds.

The circularity concern is substantially mitigated: the geometric signal is not tied to a specific observable but emerges consistently across independent projections of system dynamics. This is consistent with the interpretation that attractor deformation reflects a system-level reorganization rather than a variable-specific artifact.

**Projection-dependent visibility.** The asia_pacific result reveals an important subtlety: not all observables carry the geometric signal in all cases. In asia_pacific (AMBIGUOUS, spike), Φ and mean_corr showed 3-step lead while Oh and η showed none. This implies that geometric reorganization, while system-wide in valid cases, may be *projection-dependent* — visible from some coordinates of the state space but not others. This has three practical consequences:

1. **No single observable is sufficient.** A negative result from one embedding does not negate a positive result from another. The multi-observable spot check must evaluate consensus across projections, not rely on any single one.
2. **Consensus strengthens the signal.** When all observables agree (ALIGNED), confidence in the geometric diagnosis is highest. Partial agreement (PARTIAL) warrants caution; the signal may be real but observable-specific, or the system may be in a transitional state.
3. **Operational guideline:** Layer 2 geometry should be reported alongside its multi-observable consensus. When consensus is ALIGNED, the geometric signal can be treated as high-confidence system-level reorganization. When consensus is PARTIAL or NONE, the signal should be flagged as projection-dependent and interpreted with reduced confidence. This consensus field is computed daily by the engine (§9.1, FIX-10).

### 8.12 Formalizing Impact Sensitivity — The Θ_A Interpretation

Experiment 2 (§8.10) demonstrated that Θ_A does not predict *when* damage will occur, but detects *when the system has become vulnerable*. Empirical analysis of the two GEO_PRECEDED cases reveals the precise mechanism: during the entire pre-damage phase, ΔΦ = 0 — no damage accumulates despite geometric crystallization being active. This rules out the "amplification" interpretation (Θ_A making existing damage worse) and confirms the "barrier-lowering" interpretation formalized below.

#### 8.12.1 The Damage Barrier Model

**Definition 8.1 (Damage threshold).** Define Oh_crit(t) as the instantaneous critical Ohio Number — the minimum value of Oh required to initiate damage accumulation at time t:

    ΔΦ(t) > 0  if and only if  Oh(t) > Oh_crit(t)

In v1, Oh_crit is constant: Oh_crit = Oh_pre + δ (the calibration baseline plus margin). This is the "barrier" — the system absorbs all shocks below it without structural consequence.

**Proposition 8.1 (Θ_A lowers the damage barrier).** When geometric crystallization is active (Θ_A > 0), the effective damage barrier decreases:

    Oh_crit(t) = (Oh_pre + δ) · g(Θ_A(t))

where g: [0, ∞) → (0, 1] is a monotonically decreasing function with g(0) = 1 (barrier intact at CALM geometry) and g(Θ) → 0 as Θ → ∞ (barrier vanishes at maximal crystallization).

**Interpretation:** A crystallized system requires a *smaller* perturbation to begin accumulating damage. The degrees of freedom that normally absorb minor shocks have been frozen by the geometric reorganization. The barrier has not been breached — it has been lowered.

**Candidate functional form:**

    g(Θ_A) = exp(-α · Θ_A)

where α > 0 is the sensitivity parameter (to be calibrated from data). This gives:

    Oh_crit(t) = (Oh_pre + δ) · exp(-α · Θ_A(t))

For Θ_A = 0: Oh_crit = Oh_pre + δ (full barrier). For Θ_A = 5.0 and α = 0.5: Oh_crit ≈ 0.08 · (Oh_pre + δ) (barrier reduced to 8%).

#### 8.12.2 Impact Sensitivity as a Computable Quantity

**Definition 8.2 (Impact Sensitivity Index).** Define the impact sensitivity I_S(t) as the ratio of the current system pressure to the current effective barrier:

    I_S(t) = Oh(t) / Oh_crit(t) = Oh(t) / [(Oh_pre + δ) · g(Θ_A(t))]

When I_S(t) < 1, the system is absorbing shocks within its barrier — no damage accumulates. When I_S(t) ≥ 1, the barrier has been breached and damage is actively accumulating.

**Properties:**

- I_S increases with Oh (more pressure) and with Θ_A (lower barrier)
- I_S = Oh/(Oh_pre + δ) when Θ_A = 0 (v1 behavior, barrier at full height)
- I_S can exceed 1.0 even for moderate Oh if Θ_A is sufficiently elevated
- The margin between I_S and 1.0 quantifies how much additional pressure the system can absorb before damage begins

**Simplified operational proxy (for daily monitoring):**

    I_S_proxy(t) = Θ_A(t) · Oh(t) / (Oh_pre + δ)

This linear approximation captures the essential behavior: high Θ_A and high Oh together produce high impact sensitivity. When Θ_A = 0, I_S_proxy = 0 regardless of Oh (CALM geometry absorbs). When Oh = 0, I_S_proxy = 0 regardless of geometry (no pressure to amplify).

#### 8.12.3 Empirical Consistency

**Key empirical observation:** In both GEO_PRECEDED cases, ΔΦ = 0 throughout the entire crystallization phase (t_G to t_S). No damage accumulated despite Θ_A being elevated. Damage began only when Oh exceeded the (lowered) barrier:

| Universe | Θ_A(t_G) | Oh mean (pre-damage) | Oh at t_S | ΔΦ pre-damage | Lead time |
|---|---|---|---|---|---|
| commodities | 0.52 | 0.310 | ~1.10 | 0.000 | 185 steps |
| energy | 1.21 | 0.632 | ~1.14 | 0.000 | 24 steps |

Commodities had lower Θ_A (0.52) and lower average Oh (0.310) — the barrier was only mildly lowered, and Oh was well below it. The system waited 185 steps until a sufficiently strong perturbation arrived. Energy had higher Θ_A (1.21) and higher Oh (0.632) — the barrier was more substantially lowered, and Oh was already closer to it. Only 24 steps elapsed before the barrier was breached.

This is consistent with the barrier model: ΔT depends on the *gap* between current Oh and the lowered barrier Oh_crit(Θ_A). A narrow gap (energy) produces short lead times. A wide gap (commodities) produces long lead times.

#### 8.12.4 The Lead-Time Relationship

From the barrier model, the expected lead time is:

    ΔT ≈ time until Oh(t) first exceeds Oh_crit(Θ_A)

This depends on three factors:

1. **Θ_A magnitude** — determines how low the barrier is
2. **Current Oh level** — determines how far Oh is from the barrier
3. **Oh dynamics** — how quickly Oh fluctuates (exogenous, unpredictable)

The first two are measurable. The third is not. This is why Θ_A provides *vulnerability windows* (factors 1–2) but cannot predict *timing* (factor 3). The system can sit in a vulnerable state indefinitely if Oh never reaches the lowered barrier — as the three prospective cases currently demonstrate.

#### 8.12.5 Limitations and Calibration Status

The barrier function g(Θ_A) = exp(-α·Θ_A) is a candidate, not a calibrated model. With n=2 valid cases, α cannot be estimated. The exponential form is chosen for mathematical convenience (monotone, bounded, single parameter) but other forms (logistic, power law) are equally defensible at this stage.

Calibration requires additional crisis episodes where both t_G and t_S are identifiable. The three active prospective cases — if any reaches t_S — would provide the first independent calibration data. Until then, the operational proxy I_S_proxy = Θ_A · Oh / (Oh_pre + δ) is used without the exponential barrier, as a dimensionless indicator of vulnerability pressure.

#### 8.12.6 Operational Implications

The Impact Sensitivity Index is reported daily by the engine (v5.6) for each universe. It enables a graded response protocol:

| I_S range | Interpretation | Recommended action |
|---|---|---|
| I_S < 0.2 | Low vulnerability | Standard monitoring |
| 0.2 ≤ I_S < 0.5 | Elevated vulnerability | Increase monitoring frequency |
| 0.5 ≤ I_S < 1.0 | High vulnerability, barrier narrowing | Reduce marginal exposure, prepare contingencies |
| I_S ≥ 1.0 | Barrier breached, damage expected | Activate crisis protocols, monitor C(t) closely |

These thresholds are provisional and require calibration (§7.3).

> The system does not break when geometry changes — but geometry changes before the system becomes breakable. Impact Sensitivity quantifies *how breakable* the system currently is.

### 8.13 Experiment 3 — Hazard Model Calibration

**Design.** Logistic regression on pooled daily observations from 19 Sentinel universes (excluding 2 BASELINE_INSUFFICIENT), totaling ~4,000 pre-event-censored observations per horizon. Target: binary indicator y(t) = 1 if C_norm crosses below 0.90 for the first time within H days. Covariates: κ_F(t), ρ̄⁺(t), η̃(t), Θ_A(t). Validation: leave-one-universe-out cross-validation (LOUO-CV) and bootstrap (1000×) for coefficient confidence intervals. Acceptance criteria: no β changes sign across bootstrap resamples, ECE < 0.15.

**Results (H = 90 days, the structurally consolidated regime):**

| Coefficient | Placeholder (v5.4) | Calibrated | 95% CI | Sign stable |
|---|---|---|---|---|
| β₀ | −6.00 | **−2.70** | [−2.88, −2.52] | Yes |
| β_F (κ_F) | 1.00 | **+1.10** | [+0.44, +1.49] | Yes |
| β_ρ (ρ̄⁺) | 0.50 | **+0.03** | [+0.005, +0.10] | Yes |
| β_η (η̃) | 0.30 | **+0.41** | [+0.31, +0.52] | Yes |
| β_A (Θ_A) | 0.50 | **+5.38** | [+5.04, +5.72] | Yes |

Full-fit performance: AUC = 0.753, Brier = 0.096, LogLoss = 0.336, ECE = 0.099. LOUO-CV (5 folds): AUC = 0.660 ± 0.116, Brier = 0.125. All signs stable in bootstrap and CV. **Calibration: PASS.**

H = 30 failed calibration (ECE = 0.182 > 0.15). H = 60 passed (ECE = 0.103) but exhibited sign instability for η̃ in one CV fold, consistent with the interpretation that rigidity has dual temporal behavior (locally stabilizing, globally destabilizing). H = 90 is the recommended operational horizon.

**Key findings:**

**F1. Geometric activation is the dominant predictor.** β_A = +5.38 is an order of magnitude larger than all other coefficients. This quantitatively confirms the qualitative finding of §8.10: Θ_A is not merely an early warning indicator — it is the primary statistical driver of regime transition probability. A unit increase in Θ_A increases log-odds by 5.38, corresponding to a ~217× increase in hazard odds.

**F2. Capacity depletion rate is statistically redundant.** β_ρ = +0.03 (CI nearly touching zero). When geometric activation is in the model, the rate of capacity consumption adds negligible explanatory power. This is consistent with the information hierarchy: ρ̄ is derived from Oh and Φ (Layer 1 surface dynamics), while Θ_A captures the deeper geometric reorganization that precedes surface-level damage. Formally: geometric information subsumes depletion dynamics.

**F3. Rigidity contributes but is secondary.** β_η = +0.41. Viscosity adds value above capacity alone, but is dwarfed by geometric activation. The sign reversal at H = 30 (β_η = −0.43) vs H = 90 (β_η = +0.41) reflects a dual temporal role: rigidity is locally stabilizing (short-term shock absorption) but globally destabilizing (long-term loss of adaptability).

**F4. The baseline hazard was overly conservative.** β₀ moved from −6.00 to −2.70, corresponding to a baseline probability of ~6.3% (vs ~0.25% previously). This reflects the non-trivial prevalence of structural damage in the Sentinel corpus (16.8% at H = 90).

**Limitations.** (1) Multiple universes experienced damage from the same exogenous shock (Ukraine 2022), creating pseudo-replication. LOUO-CV mitigates but does not eliminate this correlation. (2) The AUC drop from full-fit (0.753) to CV (0.660) indicates moderate overfitting, expected given the correlated episode structure. (3) No out-of-sample validation exists yet — the three prospective cases (asia_pacific, x_brazil_vuln, x_us_systemic) represent the first opportunity. (4) The model is calibrated on a single macro-regime (2022–2026, dominated by post-pandemic tightening and geopolitical shocks); generalization to other macro-regimes is untested.

**Interpretation.** These results support the central thesis of Kappa v2: **the system changes geometry before it loses capacity.** Surface-level degradation proxies (ρ̄⁺) lose explanatory power when geometric activation is included, confirming that Θ_A captures information not available in instantaneous structural observables. The calibrated model enables probability statements of the form: "given the current geometric and structural state, the probability that structural damage begins within 90 days is X%," with ECE = 0.099 indicating that these probabilities are well-calibrated in-sample.

### 8.14 Experiment 3b — Robustness Validation (Ablation, Permutation, Random Features)

Three robustness tests were applied to the calibrated H = 90 model to confirm that the signal is genuine and not an artifact of model structure or overfitting.

**Test 1: Ablation.** Each covariate was removed individually and the model was refit. Additionally, single-covariate models were tested.

| Removed covariate | AUC | ΔAUC from full | Interpretation |
|---|---|---|---|
| (none — full model) | 0.753 | — | Baseline |
| kappa_F | 0.750 | −0.003 | Minor contribution |
| rho_bar_pos | 0.753 | 0.000 | Redundant |
| eta_norm | 0.685 | −0.068 | Significant contribution |
| **theta_A** | **0.527** | **−0.226** | **Critical — model collapses** |

Single-covariate performance: Θ_A alone achieves AUC = 0.706 (94% of full model performance). κ_F alone achieves AUC = 0.509 (near random). This confirms the information hierarchy: geometric activation carries the vast majority of predictive information, while structural capacity diagnostics are largely redundant when geometry is available.

**Test 2: Permutation test (1000 shuffles).** Target labels were randomly permuted and the model refit 1000 times to establish the null distribution.

| Metric | Value |
|---|---|
| Real AUC | 0.753 |
| Shuffled mean | 0.513 ± 0.010 |
| Shuffled 99th percentile | 0.537 |
| Shuffled maximum | 0.544 |
| Empirical p-value | < 0.001 |

The real AUC exceeds the maximum of 1000 shuffled replicates. The model is not fitting noise. The signal-to-null separation is unambiguous.

**Test 3: Random features (4 noise columns, 100 trials).** Four columns of Gaussian random noise were added to the feature matrix and the model refit 100 times.

| Feature | Original β | β with noise (mean ± std) | Drift |
|---|---|---|---|
| kappa_F | +1.104 | +1.176 ± 0.016 | 6.5% |
| rho_bar_pos | +0.029 | +0.031 ± 0.002 | 6.6% |
| eta_norm | +0.415 | +0.414 ± 0.002 | 0.1% |
| theta_A | +5.381 | +5.374 ± 0.010 | 0.1% |
| random_0 | — | −0.000 ± 0.043 | — |
| random_1 | — | −0.007 ± 0.045 | — |
| random_2 | — | −0.003 ± 0.051 | — |
| random_3 | — | −0.006 ± 0.053 | — |

All real coefficients remain stable (drift < 7%). All random coefficients are indistinguishable from zero. The model does not absorb noise as signal.

**Combined verdict: all three tests PASS.** The dominance of Θ_A is confirmed by ablation (ΔAUC = 0.226 when removed), the signal is confirmed as non-spurious by permutation (p < 0.001), and the model is confirmed as non-overfitting by random feature injection (all drifts < 7%, all random β ≈ 0).

---

## 9. Implementation Notes

### 9.1 Engine Architecture

```
engine_v4 (unchanged — v1)
    → S(t) = (Oh, Φ, η, Ξ, DEF)
    → regimes, ν_s, PR, τ_Katashi, Φ*
        │
        ▼
engine_v5 (v5.4 — Kappa v2)
    │
    ├── Layer 1 (always computed):
    │   → C(t), ρ(t), ρ̄(t), T̂_exhaust, κ_F(t)
    │   → False Recovery detection
    │   → BASELINE_INSUFFICIENT handling
    │
    ├── Layer 2 (conditional on data sufficiency):
    │   → RQA (DET, LAM, TT) with z-score normalization
    │   → Geometry reliability gate (σ-based)
    │   → Θ_A(t) (capped at 5.0, suppressed when geo=LOW)
    │
    └── Layer 3 (requires calibrated β's):
        → η smoothed via EMA before normalization
        → h(t), P_collapse, T_{1/2}
        → Multi-source alert logic with capacity confirmation
        → Anomaly detection (auto-flagging)
```

### 9.2 Implementation Fix History (v5.0 → v5.4)

| Version | Fix | Issue | Resolution |
|---|---|---|---|
| v5.0 | — | Initial implementation | Ran on 21 universes; Layer 1 correct, Layer 2/3 unstable |
| v5.1 | FIX-1 | Φ*=0 → −inf | Guard: BASELINE_INSUFFICIENT status |
| v5.1 | FIX-2 | Θ_A explosive (467.99) | CALM norm floor (0.05) + cap (5.0) |
| v5.1 | FIX-3 | κ_F overflow in logit | Cap at 20.0 |
| v5.2 | FIX-4 | Alert logic inconsistent | Multi-source: capacity-first, hazard refines |
| v5.2 | FIX-5 | Unreliable geometry dominating hazard | Geometry reliability gate (ratio-based, later revised) |
| v5.3 | FIX-6 | η single-day spike hijacks hazard | EMA smoothing + cap (5.0) |
| v5.4 | FIX-7 | CALM norm incompatible with geometry | Z-score for all RQA; σ-based gate |

Each fix confirmed the epistemological hierarchy: Layer 1 remained stable across all iterations while Layers 2–3 required progressive containment.

---

## 10. Limitations

This section explicitly addresses the principal limitations of the current work, organized by severity.

**L1. Limited statistical validation of lead times.** The predictive lead-time claim rests on 3 methodologically valid cases. While all three show geometry preceding structure, no confidence intervals, p-values, or false positive rates can be computed from this sample. The claim is directionally supported, not statistically established.

**L2. Baseline comparison shows complementarity, not superiority.** Experiment 2 (§8.10) demonstrates that Θ_A does not dominate concurrent event-proximity discrimination (AUC 0.60–0.72 under pre-event censoring, vs Oh_EMA at 0.67–0.76). However, Θ_A is the only signal that provides lead time before structural damage in valid cases (24–185 steps), while κ_F and Oh_EMA provide zero lead time. The appropriate evaluation metric for an early warning signal is temporal precedence, not concurrent AUC.

**L3. Predictive vs. descriptive ambiguity.** Whether Θ_A is genuinely predictive (detects pre-instability before structural damage) or merely descriptive (captures concurrent dynamical change) cannot be fully resolved with the current corpus. The impact sensitivity interpretation (§8.6) is the most defensible framing: Θ_A measures a condition (reduced absorptive capacity), not an event.

**L4. Circularity substantially mitigated but not fully eliminated.** Layers 1 and 2 both derive from Φ(t), raising a potential circularity concern. However, the observable independence test (§8.11) demonstrates that 7/9 embeddings of independent observables (Oh, η, mean_corr) also show geometric precedence before structural damage. The signal is system-wide, not Φ-specific. Residual concern: all tested observables share the same underlying market data; truly independent validation would require a different domain entirely.

**L5. In-sample calibration only.** Layer 3 β coefficients are now calibrated (§8.13) with ECE = 0.099 at H = 90, but validation is in-sample with leave-one-universe-out CV. No out-of-sample validation exists. The three prospective cases are the first opportunity for independent validation. Probability outputs should be treated as calibrated estimates pending out-of-sample confirmation.

**L6. Single-domain validation.** All empirical results come from the financial domain (Sentinel). Generalization to other Kappa domains (education, LLMs, neurology) is hypothesized but untested.

---

## 11. Open Questions and Future Work

1. **RQA calibration for gradual detection.** TT saturates in crystallized universes (§8.6). Adaptive ε or alternative trapping measures may enable proportional rather than binary signals.

2. **Empirical validation of predictive windows.** ΔT_{G→S} must be measured across episodes where both t_G and t_S are identifiable. x_brazil_vuln is the current best candidate.

3. **β coefficient calibration.** All β values are placeholder. Historical corpus calibration (§7.3) is required before Layer 3 probabilities can be interpreted as calibrated estimates.

4. **Cross-domain applicability of CALM/geometry insight.** The finding that "structural calmness ≠ geometric neutrality" (§3.6) should be tested in Kappa-LLM and Kappa-Education implementations.

5. **Embedding parameter universality.** τ_d and m may need scenario-class adaptation.

6. **Connection to HUGO framework.** C(t) is formally analogous to the "structural capacity for protection" in REMIND's factorial dissection.

---

## Appendix A — Experimental Details

### A.1 Sentinel Universe Composition

21 universes organized in 5 levels: Global Macro (1), Regional (6), Sectoral (4), Thematic (3), Cross-Layer (7). Total unique tickers: ~130 US-listed ETFs covering the global financial system. Data period: February 2022 to March 2026.

### A.2 Configuration (v5.4 default)

| Parameter | Value | Layer |
|---|---|---|
| γ (damage persistence) | 0.97 | 1 (inherited from v1) |
| α_ρ (depletion smoothing) | 0.1 | 1 |
| α_η (rigidity smoothing) | 0.1 | 3 |
| rqa_window | 60 | 2 |
| rqa_epsilon_q | 0.10 | 2 |
| σ_floor | 0.01 | 2 |
| σ_{geo,min} | 0.005 | 2 |
| Θ_{A,max} | 5.0 | 2 |
| κ_{F,max} | 20.0 | 1 |
| η̃_max | 5.0 | 3 |
| β_0, β_F, β_ρ, β_η, β_A | −6.0, 1.0, 0.5, 0.3, 0.5 | 3 (placeholder) |
| h_warn, h_emerg | 0.02, 0.10 | 3 |

---

## Appendix B — Notation Consistency with v1

All v1 symbols (Oh, Φ, Φ*, η, Ξ, DEF, γ, δ, PR, ν_s, S(t)) retain their exact definitions. v2 quantities are constructed from v1 quantities without modification.

---

## Appendix C — Unit Convention

All temporal quantities are expressed in **time steps of the underlying process**. Financial domain: one trading day. Educational domain: one observation period. Neural domain: one token or attention snapshot.

---

**David Ohio** | Independent Researcher | odavidohio@gmail.com
March 2026

*This document proposes a formal mathematical extension to the Kappa Method. The epistemological status of each component is explicitly declared. Empirical findings (§8) demonstrate: (1) operational viability of Layer 1 across 21 universes; (2) parameter-robustness of Layer 2 geometric signals; (3) consistent geometric precedence before structural damage in all valid cases (2/2, lead times 24–185 steps); (4) complementarity between Θ_A (early vulnerability detection) and κ_F (concurrent diagnosis), confirmed via censored baseline comparison; (5) system-wide origin of the geometric signal, demonstrated via independent observable embeddings (7/9 cases); (6) calibrated hazard model (H=90, ECE=0.099) in which geometric activation is the dominant predictor (β_A=5.38), while capacity depletion rate is statistically redundant when geometry is included; and (7) robustness confirmed via ablation (Θ_A removal collapses AUC by 0.226), permutation test (p < 0.001), and random feature injection (all real coefficients stable, all noise coefficients zero). Three prospective cases are under active monitoring and represent the first opportunity for out-of-sample validation.*
