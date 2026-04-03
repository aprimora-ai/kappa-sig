# Obsessive Coherence — Development Roadmap
## David Ohio | Independent Researcher | April 2026

---

## THESIS

Across financial markets, language models, educational systems, and neural
signals, systemic instability emerges from the same topological mechanism:
excessive structural coherence rather than noise. The Kappa Method detects
this "obsessive coherence" as a universal precursor to regime failure.

---

## DOMAIN INVENTORY

### 1. FIN — Financial Markets [COMPLETE]
- Engine v5.8c, SIG/LSCC, 21 universes, dual-horizon
- Data: yfinance (public)
- Published: v2.5 (Zenodo 19339548), Kappa-SIG (pending)
- LSCC LOUO: 0.942 | CALM attenuation: +5.3%
- Obsessive pattern: Correlation crystallization (Oh > 1) precedes drawdowns
- TODO: [ ] Notebook  [ ] Standalone script

### 2. LLM — Hallucination Detection [PUBLISHED, NEEDS UPDATE]
- Code: TopoCML/scripts/kappa_llms/
- Data: HaluEval (HuggingFace, public)
- Published: Zenodo 18883790 | 85% acc, 94.2% AUC
- Obsessive pattern: Hallucinations = hyper-coherence (premature attention collapse)
- TODO: Update with v5.8c, apply SIG, test LSCC on attention dynamics

### 3. EDU — Educational Dropout [PUBLISHED, NEEDS UPDATE]
- Code: TopoCML/ (Kappa-Education)
- Data: OULAD (public, 32,593 students)
- Published: Zenodo 18940478 (Kappa-Radiante)
- Obsessive pattern: Withdrawn = chronic engagement rigidity (high Oh, low Xi)
- TODO: Reapply with v5.8c, apply SIG, test LSCC

### 4. NEURO — Epileptic Seizure Detection [PRELIMINARY, NEEDS UPDATE]
- Code: TopoCML/scripts/kappa_eegs/process_eeg_v2_corrected.py
- Data: CHB-MIT (PhysioNet, public, 23 patients)
- Preliminary: Pre-ictal Oh growth + Phi accumulation 45-60 min before seizure
- Obsessive pattern: Seizure = neural Katashi (hyper-coherent channel state)
- TODO: Update pipeline with v5.8c, reprocess, apply SIG, test LSCC

---

## UNIFYING EVIDENCE TABLE (target for paper)

| Domain | System | Obsessive Pattern | Lead Time | LSCC AUC |
|--------|--------|-------------------|-----------|----------|
| FIN | Correlation network | Eigenvalue crystallization | 35-269 days | 0.942 |
| LLM | Attention matrices | Premature collapse | Token-level | TBD |
| EDU | Engagement network | Chronic rigidity | 63 days | TBD |
| NEURO | Channel coherence | Hyper-synchronization | 30-90 min | TBD |

---

## PHASE PLAN

### Phase 1: Domain Updates
For each domain: map to S(t), run v5.8c, extract SIG, Wheeler test,
train LSCC autoencoder, validate (T1-T6), robustness (R1-R5)

### Phase 2: Reproducible Code
Per domain: Jupyter notebook + standalone .py, public datasets, end-to-end

### Phase 3: Paper "Obsessive Coherence"
Unifying paper: 4 domains, cross-domain LSCC, intervention taxonomy
