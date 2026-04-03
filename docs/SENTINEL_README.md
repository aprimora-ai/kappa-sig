# Kappa Sentinel — Global Structural Intelligence Platform

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **"Not knowing what will happen, but measuring whether the system can handle what happens."**

Kappa Sentinel monitors the structural health of the global financial system using topological analysis on ~130 ETFs across 18 universes, augmented with OSINT data for contextual interpretation.

It does **not** predict events. It measures **structural absorption capacity** — the system's ability to handle shock.

## Architecture

```
DATA LAYER          ENGINE LAYER         INTELLIGENCE LAYER
─────────────       ─────────────        ──────────────────
Yahoo Finance  ──>  Kappa-FIN v4   ──>   OSINT Context Engine
134 ETFs             5 observables        GDELT + RSS feeds
18 universes         regime classify      signal → event correlation
5 cross-layers       Oh, Φ, η, Ξ, DEF    automated narrative
```

## Key Concept: Cross-Layer Analysis

Sectoral analysis masks inter-dependency coupling. Our first scan revealed:

| Universe | Sectoral ν_s | Cross-Layer ν_s | Amplification |
|---|---|---|---|
| Europe | 3.1 (healthy) | 210.5 (pressurized) | **68×** |
| Tech/AI | 1.9 (healthy) | 886.8 (pressurized) | **467×** |

**The risk lives between layers, not within them.**

## Universes (18)

**Level 1 — Global Macro** (21 tickers): Cross-region, cross-asset systemic monitor
**Level 2 — Regional**: US Sectors (14), Europe (12), Asia-Pacific (12), Latin America (7), MENA (8)
**Level 3 — Sectoral**: Energy (14), Tech/AI (14), Financials (12), Commodities (14)
**Level 4 — Thematic**: Iran War (14), AI Ecosystem (15), China Property (9)
**Level 5 — Cross-Layer**: US Systemic (13), Energy-Geopolitics (14), Europe Vulnerability (12), Global Contagion (13), Commodity Chain (13)

## Quick Start

```bash
# Download all price data
python src/kappa/downloader.py --full --start 2024-01-01

# Run global scan (all non-thematic universes)
python src/kappa/pipeline.py --start 2024-01-01

# Run cross-layer analysis
python src/kappa/pipeline.py --universes x_global_contagion x_us_systemic x_energy_geopolitics x_europe_vuln x_commodity_chain

# Test OSINT context engine
python test_osint.py
```

## First Scan Results (March 17, 2026)

**CRITICAL:** US Sectors (ν_s=440.3), Commodities (Oh=1.945 active spike)
**PRESSURIZED:** Europe Cross-Layer (ν_s=210.5, 227d frozen), Energy-Geopolitics (ν_s=175.8), Global Energy (ν_s=165.2, 176d frozen)
**HEALTHY:** Asia-Pacific, Latin America, Tech/AI (sectoral), Global Contagion

## Dependencies

```
pip install yfinance numpy scipy networkx gudhi pandas feedparser requests
```

## Related Work

- [Kappa-FIN v4](https://github.com/aprimora-ai/Kappa-FIN) — Engine and paper (DOI: 10.5281/zenodo.19068079)
- [Kappa Method](https://github.com/aprimora-ai/Kappa) — Theoretical framework

## Author

David Ohio — Independent Researcher
odavidohio@gmail.com | [GitHub](https://github.com/aprimora-ai)

## License

CC BY 4.0
