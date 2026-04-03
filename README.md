# Obsessive Coherence

**A Domain-Agnostic Structural Signature of Fragility and a Layered Architecture for Predictive Monitoring**

[![DOI Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19393239.svg)](https://doi.org/10.5281/zenodo.19393239)
[![DOI Code](https://zenodo.org/badge/DOI/10.5281/zenodo.15627983.svg)](https://doi.org/10.5281/zenodo.15627983)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## The Law of Katashi

> *Systemic instability emerges from excessive structural coherence rather than noise.*

Complex systems don't always fail because something goes wrong. Sometimes they fail because everything goes too right — too correlated, too aligned, too coherent. Until they shatter.

**Obsessive coherence** is the hypothesis that when a system's components become excessively coupled and spectrally concentrated, the system enters a crystallized state that is paradoxically the most ordered and the most fragile configuration it can achieve.

## The Finding

**24 out of 25 predicted directional changes confirmed across 5 domains, ~8 orders of magnitude in temporal scale, zero domain-specific tuning.**

| Domain | Substrate | Network Type | Directions | OCI Delta | Scale |
|--------|-----------|-------------|------------|-----------|-------|
| **FIN** | Financial markets | Asset correlation (21 universes) | 5/5 | +1.203 | Months |
| **LLM** | Transformer attention | Head correlation (3 architectures) | 14/15 | +0.020 | Milliseconds |
| **EDU** | Student engagement | Channel correlation (OULAD) | 5/5 | +0.003 | Weeks |
| **POL** | Political blogs | Adjacency eigenstructure (1491 nodes) | 5/5 | +0.141 | Static |
| **MET** | Atmospheric dynamics | Temperature correlation (30 stations) | 5/5 | +0.143 | Days |

---

## The Pipeline

The same mathematical pipeline is applied identically across all domains:

1. Construct a symmetric coupling matrix **C** (correlation or adjacency)
2. Compute eigenvalues of **C**
3. Extract five observables: **Oh** (concentration), **η** (rigidity), **DEF** (dominance), **Ξ** (diversity), **ρ̄** (coupling)
4. Compare observables between nominal and stressed conditions
5. Test five predicted directional changes

No domain-specific tuning, threshold adjustment, or observable weighting required.

## Key Findings

### Softmax Observation (LLM)
Post-softmax attention matrices are row-stochastic, degenerating 3/5 per-head observables (φ→0, ξ→∞, δ→NaN). The resolution: inter-head correlation analysis, consistent with the framework's network-level methodology across all domains.

### Crystallization-Collapse Cycle (EDU)
Withdrawn students start with the *highest* structural coherence (Oh=4.50) then experience the steepest collapse (drift=−1.55). The dropout pattern mirrors the financial crisis cycle: Katashi → rupture.

### Structural vs Dynamic Failure (MET)
The framework discriminates atmospheric blocking (5/5 confirmed) from dynamic storms (1–2/5). Kappa detects structural failure modes — where the system becomes too rigid — not all extreme events.

### Synthetic Networks
Standard random graph models (Barabási-Albert, Erdős-Rényi, Watts-Strogatz) do not consistently reproduce the obsessive coherence pattern, suggesting it is a property of naturally evolved complex systems.

### The 2008 GFC — Ten Months Early
The Katashi regime onset was detected on November 13, 2007 — approximately ten months before the Lehman Brothers bankruptcy. Oh exceeded 1.0 and remained elevated for 386 consecutive days. LSCC vulnerability metric achieves AUC = 0.942.

---

## Repository Structure

```
kappa-sig/
├── run_obsessive_coherence.py              # Cross-domain unified analysis
│
├── engine/                                 # Kappa engine lineage (v1 → v5.8c)
│   ├── engine.py                           # Original engine
│   └── engine_v5_fixed_v58c.py             # Current production engine
│
├── sig/                                    # SIG pipeline (analytic + learned)
│   ├── run_sig_experiment.py               # S1-S3 analytic precursors
│   ├── run_sig_s4_s5.py                    # S4 Neural ODE + S5 Autoencoder
│   ├── run_lscc_tests.py                   # LSCC tests T1-T6
│   └── run_lscc_robustness.py              # Robustness R1-R5
│
├── sentinel/                               # Production monitoring system
│   ├── pipeline/                           # Daily pipeline (downloader, analyzer)
│   ├── config/                             # 21 universe definitions
│   └── experiments/                        # v2 experiments and calibration
│
├── domains/
│   ├── llm/                                # LLM hallucination detection
│   │   ├── run_halueval_experiment.py       # Full HaluEval rerun (GPU, ~16 min)
│   │   ├── run_llm_sig.py                  # OC analysis with verified data
│   │   ├── test_aggregation.py             # Mean vs Max pooling comparison
│   │   ├── kappa_llm/                      # Core LLM library
│   │   └── results/halueval_rerun/         # Raw CSVs from 3 models
│   ├── edu/                                # Educational dropout
│   │   ├── run_edu_sig.py                  # OULAD correlation-based Kappa
│   │   ├── src/                            # KatashiAnalyzer + topological metrics
│   │   └── data/                           # OULAD cohort CSVs included
│   ├── pol/                                # Political polarization
│   │   ├── run_pol_sig.py                  # Blogosphere eigenstructure
│   │   └── data/polblogs.gml              # Dataset included
│   ├── met/                                # Atmospheric dynamics
│   │   ├── run_met_sig.py                  # Open-Meteo 30 stations
│   │   └── data/temperatures.csv           # Cached temperature data
│   └── networks/                           # Synthetic vs real comparison
│       └── run_multi_network_sig.py
│
├── results/                                # All experiment results (JSON)
│   ├── obsessive_coherence/                # Cross-domain unified
│   ├── lscc/                               # LSCC tests
│   ├── lscc_robustness/                    # Robustness R1-R5
│   ├── sig_experiments/                    # SIG S1-S5
│   └── v2_calibration/                     # Engine v2 calibration
│
├── data/sig_features/                      # Pre-computed SIG features (21 universes)
├── scripts/                                # Historical validation scripts
├── paper/                                  # Synthesis paper (md + pdf)
└── docs/                                   # Technical documentation
```

## Running

```bash
# Cross-domain unified analysis (loads pre-computed results)
python run_obsessive_coherence.py

# Individual domains
python domains/pol/run_pol_sig.py              # ~0.2s, data included
python domains/edu/run_edu_sig.py              # ~0.2s, data included
python domains/met/run_met_sig.py              # ~0.5s, cached data
python domains/llm/run_llm_sig.py              # ~0.0s, verified data hardcoded
python domains/networks/run_multi_network_sig.py  # ~5s, uses NetworkX built-in

# Full LLM rerun (requires GPU + ~16 min)
python domains/llm/run_halueval_experiment.py
```

## Requirements

```
numpy
pandas
scipy
scikit-learn
networkx
```

For LLM domain additionally:
```
torch
transformers
bitsandbytes
datasets
```

## Related Publications

| Paper | DOI |
|-------|-----|
| **Obsessive Coherence** (this work) | [10.5281/zenodo.19393239](https://doi.org/10.5281/zenodo.19393239) |
| Kappa Method v2.5 | [10.5281/zenodo.19339548](https://doi.org/10.5281/zenodo.19339548) |
| Kappa-FIN | [10.5281/zenodo.18917558](https://doi.org/10.5281/zenodo.18917558) |
| Kappa-LLM | [10.5281/zenodo.18883790](https://doi.org/10.5281/zenodo.18883790) |
| Kappa-Radiante | [10.5281/zenodo.18940478](https://doi.org/10.5281/zenodo.18940478) |

## Citation

```bibtex
@article{ohio2026obsessive,
  author  = {Ohio, David},
  title   = {Obsessive Coherence: A Domain-Agnostic Structural Signature
             of Fragility and a Layered Architecture for Predictive Monitoring},
  year    = {2026},
  doi     = {10.5281/zenodo.19393239},
  license = {CC BY 4.0}
}
```

## Author

**David Ohio** — Independent Researcher
odavidohio@gmail.com
GitHub: [aprimora-ai](https://github.com/aprimora-ai)

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
