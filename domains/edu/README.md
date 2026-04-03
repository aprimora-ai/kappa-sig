# Kappa-Education: Regime Detection in Educational Systems

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Application of the [Kappa Method](https://github.com/odavidohio/Kappa-Method) for detecting structural regime transitions in educational systems.**

---

## 🎯 Overview

This repository contains the **complete implementation and validation** of the Katashi method (カタシ, "shape/structure") applied to student dropout detection in higher education. Using topological data analysis and dissipative structures theory, we detect regime transitions **7-77 days before final outcomes** based exclusively on engagement patterns.

### Key Results

| Cohort | Lead Time | Pattern Type | Interpretation |
|--------|-----------|--------------|----------------|
| **Fail** | 63 days | Gradual deterioration | Early warning signals |
| **Pass** | 77 days | Prolonged stability | Resilient regime |
| **Distinction** | 7 days | **Explosive coherence** | Rapid reorganization |
| **Withdrawn** | 68 days | Chronic rigidity | Pathological regime |

### What Makes This Different?

Traditional learning analytics ask: **"Who will drop out?"**  
We ask: **"When does the educational system enter a structural risk regime?"**

This paradigm shift enables:
- ✅ **Regime-based interventions** instead of individual predictions
- ✅ **No grade data required** – works with engagement patterns only
- ✅ **Mechanistic interpretation** via dissipative structures theory
- ✅ **Actionable lead times** for pedagogical interventions

---

## 📚 Documentation

Three complementary documents form the scientific foundation:

### 1. [Institutional White Paper](docs/papers/Katashi_White_Paper_Institucional.pdf) (2 pages)
Strategic positioning and conceptual bridge connecting science to practice.

### 2. [Scientific Protocol](docs/papers/artigo_katashi_protocolo_cientifico.pdf) (19 pages)
Complete mathematical formalization with:
- Rigorous definitions of Ω, Ξ, Φ, η observables
- Topological analysis via persistent homology
- Replication protocol with O(n log n) complexity

### 3. [Educational Analysis](docs/papers/artigo_katashi_educacao_v2.pdf) (39 pages)
Empirical validation on OULAD dataset (32,593 students) including:
- Four distinct operational regimes
- Lead time quantification
- Statistical validation (Kruskal-Wallis, Mann-Whitney)
- Ethical considerations and limitations

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/odavidohio/Kappa-Education.git
cd Kappa-Education
pip install -r requirements.txt
```

### Reproduce Paper Results

```python
from src.katashi_analyzer import KatashiAnalyzer
import pandas as pd

# Load OULAD data (see data/README.md for download instructions)
df = pd.read_csv('data/oulad_processed.csv', parse_dates=['date'], index_col='date')

# Initialize analyzer
analyzer = KatashiAnalyzer(window=10, calm_length=12)

# Detect regimes
state = analyzer.analyze(df)
calm_info = analyzer.detect_calm(state)
thresholds = analyzer.calibrate_thresholds(state, calm_info)
crossings = analyzer.detect_crossings(state, thresholds)

# Visualize
analyzer.plot_regime_evolution(state, calm_info, thresholds, crossings)
```

### Or Use Jupyter Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/odavidohio/Kappa-Education/blob/main/notebooks/01_reproduce_paper_results.ipynb)

1. `notebooks/01_reproduce_paper_results.ipynb` – Reproduces Figures 1-4 and Tables 1-5
2. `notebooks/02_regime_analysis.ipynb` – Deep dive into regime characteristics
3. `notebooks/03_lead_time_validation.ipynb` – Statistical validation of lead times
4. `notebooks/04_cohort_comparison.ipynb` – Comparative analysis across cohorts

---

## 📊 Core Method

### The Five Structural Observables

The Kappa Method detects regime transitions through five independent but coupled structural observables:

**1. Oh (Regime Pressure) - Normalized Regime Number:**
```
Oh = Ξ / Ξ_c
```
Measures distance from structural baseline (CALM). Oh > 1 indicates critical regime.

**2. Φ (Structural Memory) - Accumulated Damage:**
```
Φ(t+1) = γ × Φ(t) + δ × max(0, Oh(t) - 1.0)
```
- γ = 0.97 (forgetting factor)
- δ = 0.08 (damage sensitivity)
- Accumulates only when Oh > 1.0 (critical regime)

**3. η (Dynamic Rigidity) - Structural Friction:**
```
η = 1 + log(1 + ||C||_Frobenius)
```
Measures resistance to state changes via correlation matrix norm.

**4. Ξ (Structural Diversity) - Intensity:**
```
Ξ = (1 + |ρ_mean|) × (1 + λ_dominant) × k
```
Captures richness of independent structural paths.

**5. DEF (State-Phase Divergence) - Coherence:**
```
DEF(t) = |x(t) - P(ẋ(t))|
```
**NEW**: Measures incoherence between where the system is (state x) and where it's going (phase ẋ).
- High DEF: System in wrong place for its trajectory → precedes observable failures
- DEF detects instabilities invisible to other metrics

### Why DEF Matters

Traditional metrics measure what's happening NOW. DEF measures the **mismatch between current state and future trajectory**, detecting crises before they manifest:

- **Stable but misaligned**: System appears stable (low Oh) but moving toward instability (high DEF)
- **Early warning**: DEF rises before Oh crosses critical threshold
- **Predictive power**: Identifies systems "in the wrong place" for their dynamics

### CALM Period Detection

**Collectively Aligned Low-variance Mode** – temporal window with:
1. Low variance in Ω: std(Ω) < threshold
2. No systematic drift: linear trend ≈ 0
3. Collectively synchronized behavior

### Crossing Detection

**Three alert levels:**
1. **Ω > 1.0** – Absolute rigid regime
2. **Ω > θ** – Deviation from CALM reference
3. **Φ > Φ_c** – Critical accumulated memory

**Two temporal stages:**
- **Sensitization:** First sustained violation (2 weeks)
- **Confirmation:** Prolonged violation (5 weeks)

---

## 🔬 Dataset

**Open University Learning Analytics Dataset (OULAD)**
- 32,593 students across 22 modules
- Analyzed course: AAA_2014J (Social Sciences, Oct 2014)
- Duration: 40 weeks
- Granularity: Individual clicks with timestamps

**Cohorts:**
- Pass: 1,234 students (55%)
- Fail: 456 students (15%)
- Distinction: 789 students (8%)
- Withdrawn: 678 students (22%)

Download instructions: [data/README.md](data/README.md)

---

## 📈 Results Summary

### Temporal Chronology of Threshold Crossings

| Cohort | φ_sens | φ_confirm | Ω>θ_confirm | Ω>1_confirm |
|--------|--------|-----------|-------------|-------------|
| Pass | 02/07 | 09/07 | 16/07 | 17/09 |
| Fail | 04/06 | 11/06 | 06/08 | 06/08 |
| Distinction | 13/08 | 20/08 | 20/08 | 20/08 |
| Withdrawn | 09/07 | 16/07 | 16/07 | (truncated) |

### Lead Time Metrics

| Cohort | Δt_total (days) | Δt_cascade (days) | Sync Ratio |
|--------|----------------|-------------------|------------|
| Pass | 77 | 77 | 1.00 |
| Fail | 63 | 63 | 1.00 |
| **Distinction** | **7** | **0** | **0.00** |
| Withdrawn | ~68 | ~68 | 1.00 |

**Key Discovery:** Distinction exhibits **explosive coherence** – all thresholds crossed simultaneously within 7 days, indicating rapid structural reorganization rather than gradual deterioration.

### Statistical Validation

**Kruskal-Wallis Test:**
- Δt_total: H = 38.7, p < 0.001
- max(Ω): H = 31.2, p < 0.001
- std(Ξ): H = 24.8, p < 0.001

**Effect sizes (Cohen's d):**
- Distinction vs Pass: d = 2.8
- Distinction vs Fail: d = 2.4

---

## 🎓 Theoretical Foundation

### Connection to Dissipative Structures (Prigogine, 1977)

Educational trajectories as **informational dissipative systems**:

| Physical Concept | Educational Analog |
|-----------------|-------------------|
| Energy dissipation | Information processing via engagement |
| Non-linearity | Feedback between motivation ↔ performance |
| Self-organization | Emergent collective study patterns |
| Bifurcations | Learning crises as regime transitions |

### Early Warning Signals

Convergence with complex systems theory:
- **Critical slowing down:** Extended lead times in Fail/Pass
- **Increased variance:** High std(Ξ) pre-transition
- **Absence of slowing:** Distinction's first-order transition

---

## 🛠️ Repository Structure

```
Kappa-Education/
├── src/                              # Python modules
│   ├── katashi_analyzer.py          # Main class
│   ├── topological_metrics.py       # Ξ, Φ, Ω, η computation
│   ├── calm_detection.py            # CALM period identification
│   ├── regime_classifier.py         # Regime classification
│   └── visualization.py             # Plotting functions
├── notebooks/                        # Jupyter notebooks
│   ├── 01_reproduce_paper_results.ipynb
│   ├── 02_regime_analysis.ipynb
│   ├── 03_lead_time_validation.ipynb
│   └── 04_cohort_comparison.ipynb
├── scripts/                          # Executable scripts
│   ├── run_analysis.py              # Full pipeline
│   └── generate_figures.py          # Create all figures
├── tests/                            # Unit tests
├── docs/papers/                      # Scientific papers (PDFs)
├── data/                             # Data processing
└── results/                          # Figures, tables, metrics
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@software{ohio2026katashi_education,
  author = {Ohio, David Ferreira},
  title = {Kappa-Education: Regime Detection in Educational Systems},
  year = {2026},
  url = {https://github.com/odavidohio/Kappa-Education},
  doi = {10.5281/zenodo.XXXXXX}
}
```

And the core method:

```bibtex
@software{ohio2026kappa_method,
  author = {Ohio, David Ferreira},
  title = {Kappa: A Method for Informational Regime Detection via Geometry and Dynamics},
  year = {2026},
  url = {https://github.com/odavidohio/Kappa-Method},
  doi = {10.5281/zenodo.18434598}
}
```

---

## ⚖️ Ethical Considerations

### Limitations
- ✋ Single-course validation (AAA_2014J)
- ✋ Observational study (no causal inference)
- ✋ Group-level patterns (individual uncertainty remains)
- ✋ Parameter sensitivity requires institutional calibration

### Responsible Implementation
- ✅ Use for **support offering**, never punishment
- ✅ Full transparency about methods and metrics
- ✅ Right to contest and opt-out
- ✅ Continuous equity auditing across demographics
- ✅ Human oversight of intervention decisions

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Multi-institutional validation
- Causal inference studies
- Real-time monitoring systems
- Integration with LMS platforms

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

Code is additionally available under MIT License for practical implementations.

---

## 🔗 Related Projects

- [Kappa-Method](https://github.com/odavidohio/Kappa-Method) – Core methodological framework
- [Kappa-Finance](https://github.com/odavidohio/Kappa-Finance) – Financial crisis detection (Heimdall)
- [Kappa-LLM](https://github.com/odavidohio/Kappa-LLM) – AI hallucination detection

---

## 📧 Contact

**David Ferreira Ohio Junior**  
Senior IT Manager, Instituto Avança SP  
Independent Researcher in AI Safety & Topological Data Analysis

- GitHub: [@odavidohio](https://github.com/odavidohio)
- Email: [contact information]

---

<div align="center">

**When complex systems forget how to remain stable**

*Educational crises are not punctual events, but persistent structural states detectable through topological invariants.*

</div>
