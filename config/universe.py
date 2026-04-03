# -*- coding: utf-8 -*-
"""
Kappa Sentinel — ETF Universe Definition
==========================================
~130 ETFs covering the entire global financial system.

Organized hierarchically:
  Level 1: Global Macro (cross-region, cross-asset)
  Level 2: Regional (US, Europe, Asia, EM, LatAm, MENA)
  Level 3: Sectoral (Energy, Tech/AI, Financials, Commodities)
  Level 4: Thematic (on-demand, constructed when L1/L2 flags)

David Ohio | odavidohio@gmail.com | March 2026
"""

# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1: GLOBAL MACRO — systemic fragility detection
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_MACRO = {
    "name": "Global Macro",
    "description": "Cross-region, cross-asset systemic monitor",
    "k": 5,
    "tickers": [
        # Equity regions
        "SPY",    # US S&P 500
        "QQQ",    # US Nasdaq 100
        "IWM",    # US Small caps
        "EFA",    # Developed ex-US (Europe, Japan, Aus)
        "EEM",    # Emerging Markets
        "FXI",    # China
        "EWJ",    # Japan
        "EWZ",    # Brazil
        # Fixed income
        "TLT",    # US Long Treasury 20y+
        "IEF",    # US 7-10y Treasury
        "SHY",    # US 1-3y Treasury
        "LQD",    # US Investment Grade Corp
        "HYG",    # US High Yield Corp
        "EMB",    # EM Sovereign Debt
        # Commodities
        "GLD",    # Gold
        "SLV",    # Silver
        "USO",    # Crude Oil
        "UNG",    # Natural Gas
        "DBA",    # Agriculture
        # Currency / Dollar
        "UUP",    # US Dollar Index (long)
        # Volatility
        "VIXY",   # VIX Short-Term Futures
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2: REGIONAL UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════

REGIONAL_US = {
    "name": "US Sectors",
    "description": "US economy by sector — detects internal rotation and concentration",
    "k": 5,
    "tickers": [
        "XLF",    # Financials
        "XLE",    # Energy
        "XLK",    # Technology
        "XLV",    # Healthcare
        "XLI",    # Industrials
        "XLC",    # Communication
        "XLY",    # Consumer Discretionary
        "XLP",    # Consumer Staples
        "XLU",    # Utilities
        "XLB",    # Materials
        "XLRE",   # Real Estate
        "IYT",    # Transportation
        "KBE",    # Banks (regional)
        "XHB",    # Homebuilders
    ],
}

REGIONAL_EUROPE = {
    "name": "Europe",
    "description": "European markets by country — detects fragmentation vs integration",
    "k": 5,
    "tickers": [
        "EZU",    # Eurozone
        "EWG",    # Germany
        "EWQ",    # France
        "EWI",    # Italy
        "EWP",    # Spain
        "EWU",    # United Kingdom
        "EWL",    # Switzerland
        "EWD",    # Sweden
        "ENOR",   # Norway
        "GREK",   # Greece
        "EPOL",   # Poland
        "TUR",    # Turkey
    ],
}

REGIONAL_ASIA = {
    "name": "Asia-Pacific",
    "description": "Asia-Pacific markets — detects China contagion and Japan dynamics",
    "k": 5,
    "tickers": [
        "EWJ",    # Japan
        "FXI",    # China Large Cap
        "KWEB",   # China Internet
        "EWT",    # Taiwan
        "EWY",    # South Korea
        "INDA",   # India
        "EWA",    # Australia
        "EWS",    # Singapore
        "THD",    # Thailand
        "VNM",    # Vietnam
        "EPHE",   # Philippines
        "IDX",    # Indonesia
    ],
}

REGIONAL_LATAM = {
    "name": "Latin America",
    "description": "Latin American markets — commodity exposure + political risk",
    "k": 4,
    "tickers": [
        "EWZ",    # Brazil
        "EWW",    # Mexico
        "ECH",    # Chile
        "ARGT",   # Argentina
        "GXG",    # Colombia
        "EPU",    # Peru
        "ILF",    # Latin America 40
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# BRAZIL-SPECIFIC UNIVERSES (US-listed Brazilian assets)
# ══════════════════════════════════════════════════════════════════════════════

REGIONAL_BRAZIL = {
    "name": "Brazil — Sectoral Decomposition",
    "description": "Brazilian economy by sector via US-listed ADRs/ETFs. "
                   "Detects internal rotation, commodity dependency, and "
                   "financial sector concentration risk.",
    "k": 5,
    "tickers": [
        # Broad market
        "EWZ",    # iShares MSCI Brazil ETF
        "EWZS",   # iShares MSCI Brazil Small-Cap ETF
        # Energy / Resources (commodity channel)
        "PBR",    # Petrobras (oil, state-controlled)
        "VALE",   # Vale (iron ore, nickel, copper)
        "GGB",    # Gerdau (steel)
        "SID",    # CSN - Companhia Siderúrgica Nacional
        "SBS",    # SABESP (water/sanitation, recently privatized)
        "CIG",    # CEMIG (electricity, Minas Gerais)
        # Financials (concentration risk)
        "ITUB",   # Itaú Unibanco (largest private bank)
        "BBD",    # Bradesco
        "BSBR",   # Banco Santander Brasil
        "NU",     # Nubank (fintech disruptor)
        "XP",     # XP Inc (brokerage/wealth)
        # Consumer / Tech
        "ABEV",   # Ambev (beverages, BRL consumer proxy)
        "STNE",   # StoneCo (payments)
        "PAGS",   # PagSeguro (payments)
    ],
}

CROSS_LAYER_BRAZIL_VULNERABILITY = {
    "name": "Brazil Vulnerability — Cross-Layer",
    "description": "Tests structural coupling between Brazilian equities, "
                   "commodities (iron ore, oil), USD/BRL proxy, US rates, "
                   "and EM debt. Detects whether Brazil-specific stress "
                   "propagates across channels or remains contained.",
    "k": 5,
    "tickers": [
        # Brazilian equities
        "EWZ",    # Broad Brazil
        "ITUB",   # Financial bellwether
        "PBR",    # Energy bellwether
        "VALE",   # Commodity bellwether
        # Commodity dependency channel
        "USO",    # Oil (Petrobras revenue)
        "GLD",    # Gold (safe haven / real depreciation hedge)
        "SLV",    # Silver (industrial metals proxy)
        "DBA",    # Agriculture (agribusiness export channel)
        # Currency / rates channel
        "UUP",    # Dollar strength (inverse proxy for BRL)
        "EMB",    # EM sovereign debt (Brazilian bond proxy)
        "TLT",    # US long treasury (Selic differential driver)
        # Contagion / risk appetite
        "EEM",    # EM equities (risk appetite proxy)
        "HYG",    # US high yield (global risk appetite)
    ],
}

REGIONAL_MENA = {
    "name": "Middle East & Africa",
    "description": "MENA + Africa — energy geopolitics + frontier markets",
    "k": 4,
    "tickers": [
        "KSA",    # Saudi Arabia
        "UAE",    # UAE
        "QAT",    # Qatar
        "EGPT",   # Egypt
        "NGE",    # Nigeria
        "EZA",    # South Africa
        "AFK",    # Africa total
        "TUR",    # Turkey (also in Europe)
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3: SECTORAL UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════

SECTOR_ENERGY = {
    "name": "Global Energy",
    "description": "Energy value chain — upstream, midstream, downstream, renewables",
    "k": 5,
    "tickers": [
        "XLE",    # US Energy sector
        "XOP",    # US Oil & Gas E&P
        "OIH",    # Oil Services
        "AMLP",   # MLPs (midstream)
        "USO",    # Crude Oil
        "UNG",    # Natural Gas
        "ICLN",   # Clean Energy
        "TAN",    # Solar
        "LIT",    # Lithium & Battery
        "URA",    # Uranium
        "XOM",    # Exxon (bellwether)
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger
    ],
}

SECTOR_TECH_AI = {
    "name": "Technology & AI Ecosystem",
    "description": "AI value chain — compute, infrastructure, software, supply chain",
    "k": 5,
    "tickers": [
        "QQQ",    # Nasdaq 100
        "SMH",    # Semiconductors
        "SOXX",   # Semiconductor Index
        "IGV",    # Software
        "SKYY",   # Cloud Computing
        "BOTZ",   # Robotics & AI
        "ARKK",   # Innovation (Ark)
        "NVDA",   # NVIDIA
        "MSFT",   # Microsoft
        "GOOGL",  # Google
        "META",   # Meta
        "AMD",    # AMD
        "AVGO",   # Broadcom
        "TSM",    # TSMC
    ],
}

SECTOR_FINANCIALS = {
    "name": "Global Financials",
    "description": "Financial system — banks, insurance, fintech, crypto",
    "k": 5,
    "tickers": [
        "XLF",    # US Financials
        "KBE",    # US Banks
        "KRE",    # US Regional Banks
        "IAK",    # Insurance
        "EUFN",   # Europe Financials
        "FINX",   # Fintech
        "IBIT",   # Bitcoin ETF
        "ETHA",   # Ethereum ETF
        "JPM",    # JPMorgan (bellwether)
        "GS",     # Goldman Sachs
        "BAC",    # Bank of America
        "HSBC",   # HSBC (global)
    ],
}

SECTOR_COMMODITIES = {
    "name": "Commodities & Materials",
    "description": "Physical economy — metals, agriculture, mining",
    "k": 5,
    "tickers": [
        "GLD",    # Gold
        "SLV",    # Silver
        "PPLT",   # Platinum
        "CPER",   # Copper
        "DBA",    # Agriculture
        "WEAT",   # Wheat
        "CORN",   # Corn
        "SOYB",   # Soybeans
        "XME",    # Metals & Mining
        "PICK",   # Global Mining
        "WOOD",   # Timber
        "REMX",   # Rare Earth
        "USO",    # Oil
        "UNG",    # Natural Gas
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 4: THEMATIC UNIVERSES (constructed on-demand when L1/L2 flags)
# ══════════════════════════════════════════════════════════════════════════════

THEMATIC_IRAN_WAR = {
    "name": "Iran War — Energy & Defense",
    "description": "Energy/defense exposure to Iran conflict + Hormuz risk",
    "k": 5,
    "tickers": [
        "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "DVN", "FANG",
        "LMT", "RTX", "NOC", "GD",  # Defense
        "USO", "UNG",               # Commodities
    ],
}

THEMATIC_AI_ECOSYSTEM = {
    "name": "AI Full Ecosystem — Cross-Layer",
    "description": "Multi-layer AI coherence: M7 + speculative + infra + supply",
    "k": 5,
    "tickers": [
        "NVDA", "MSFT", "GOOGL", "META", "AMD", "AVGO", "TSM",
        "SMCI", "ARM", "PLTR", "SNOW", "AI",  # AI plays
        "VST", "CEG", "EQIX",                  # Infrastructure
    ],
}

THEMATIC_CHINA_PROPERTY = {
    "name": "China Property Crisis",
    "description": "China real estate + contagion channels",
    "k": 4,
    "tickers": [
        "FXI", "KWEB", "MCHI", "CHIQ",   # China broad
        "GXC",                             # China total market
        "EWH",                             # Hong Kong
        "EWA", "EWS",                      # Aus/Singapore (exposure)
        "EMLC",                            # EM Local Currency Debt
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE REGISTRY & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# All universes organized by level
# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 5: CROSS-LAYER COMPOSITES — the hidden risk layer
# ══════════════════════════════════════════════════════════════════════════════
#
# CRITICAL INSIGHT (March 17, 2026):
# Sectoral analysis masks cross-layer coupling.
# Tech/AI showed nu_s=1.9 (healthy) as a sector,
# but nu_s=886.8 (pressurized) as cross-layer ecosystem.
#
# The same masking effect likely applies to ALL sectors.
# These composites test for hidden inter-sector coupling.

CROSS_LAYER_US_SYSTEMIC = {
    "name": "US Systemic — Cross-Layer",
    "description": "US economy cross-layer: sectors + bonds + dollar + volatility. "
                   "Tests if US sectors are moving together 'because macro' the same way "
                   "AI tickers moved together 'because AI'.",
    "k": 5,
    "tickers": [
        # Equity sectors (representatives, not all 14)
        "XLF",    # Financials
        "XLE",    # Energy
        "XLK",    # Technology
        "XLV",    # Healthcare
        "XLI",    # Industrials
        # Fixed income
        "TLT",    # Long Treasury
        "HYG",    # High Yield
        "LQD",    # Investment Grade
        # Macro
        "UUP",    # Dollar
        "GLD",    # Gold
        "VIXY",   # Volatility
        "IWM",    # Small caps (risk appetite proxy)
        "SPY",    # Broad market
    ],
}

CROSS_LAYER_ENERGY_GEOPOLITICS = {
    "name": "Energy-Geopolitics Nexus — Cross-Layer",
    "description": "Energy + MENA + Defense + Commodities + EM Debt. "
                   "Tests structural coupling between energy prices, Gulf states, "
                   "defense spending, commodity routes, and EM vulnerability. "
                   "The Iran War channel.",
    "k": 5,
    "tickers": [
        # Energy producers
        "XOM", "CVX", "SLB",
        # Defense
        "LMT", "RTX", "NOC",
        # Gulf states
        "KSA", "UAE", "QAT",
        # Commodities affected by Hormuz
        "USO", "UNG", "GLD",
        # EM vulnerability channel
        "EMB",   # EM sovereign debt
        "EEM",   # EM equities
    ],
}

CROSS_LAYER_EUROPE_VULNERABILITY = {
    "name": "Europe Vulnerability — Cross-Layer",
    "description": "Europe + Energy imports + EM exposure + China trade. "
                   "Tests if Europe's apparent health masks dependency coupling.",
    "k": 5,
    "tickers": [
        # European equities
        "EZU",    # Eurozone
        "EWG",    # Germany (industry)
        "EWU",    # UK (finance)
        # Energy dependency
        "UNG",    # Natural Gas (EU depends on imports)
        "USO",    # Oil
        # Financial exposure
        "EUFN",   # European Financials
        "EMB",    # EM Debt (European banks hold EM debt)
        # Trade partners
        "FXI",    # China (EU's 2nd trade partner)
        "EWJ",    # Japan
        # Safe havens
        "GLD",    # Gold
        "TLT",    # US Treasury (flight to safety)
        "EWL",    # Switzerland (safe haven proxy)
    ],
}

CROSS_LAYER_GLOBAL_CONTAGION = {
    "name": "Global Contagion Map — Cross-Layer",
    "description": "One representative from each major asset class/region. "
                   "The ultimate test: is the global financial system moving as one? "
                   "High nu_s here = systemic crystallization across all channels.",
    "k": 5,
    "tickers": [
        # Major equity regions
        "SPY",    # US
        "EZU",    # Europe
        "FXI",    # China
        "EWJ",    # Japan
        "EEM",    # Emerging Markets
        "EWZ",    # Brazil (LatAm)
        # Fixed income
        "TLT",    # US Treasuries
        "EMB",    # EM Debt
        # Commodities
        "USO",    # Oil
        "GLD",    # Gold
        # Currency
        "UUP",    # Dollar
        # Volatility
        "VIXY",   # VIX
        # Crypto
        "IBIT",   # Bitcoin
    ],
}

CROSS_LAYER_ENERGY_TECH_TRANSMISSION = {
    "name": "Energy-Tech Transmission — Cross-Layer",
    "description": "Tests structural coupling between energy markets and semiconductor "
                   "supply chain. Hypothesis: Hormuz closure → LNG disruption → Taiwan "
                   "electricity stress → fab cost/availability → global chip supply. "
                   "All tickers are real, liquid, US-listed assets with deep history. "
                   "No proxies. Added March 28, 2026.",
    "k": 5,
    "tickers": [
        # Energy (the shock source)
        "USO",    # Crude Oil
        "UNG",    # Natural Gas (LNG proxy — Henry Hub tracks Asian spot with lag)
        "XLE",    # US Energy sector
        # Semiconductors (the transmission target)
        "TSM",    # TSMC — 60% global foundry, Taiwan electricity dependent
        "SMH",    # VanEck Semiconductor ETF
        "NVDA",   # NVIDIA — AI compute demand driving Taiwan fab expansion
        "AVGO",   # Broadcom — diversified semi, fab-dependent
        "AMD",    # AMD — TSMC customer
        # Semiconductor infrastructure
        "SOXX",   # iShares Semiconductor Index
        "AMAT",   # Applied Materials — fab equipment
        # Cross-channel bridges
        "EWT",    # iShares MSCI Taiwan — Taiwan equity market (energy + tech)
        "GLD",    # Gold — safe haven / risk-off signal
        "IYT",    # Transportation — shipping/logistics bottleneck
    ],
}

CROSS_LAYER_COMMODITY_CHAIN = {
    "name": "Commodity Supply Chain — Cross-Layer",
    "description": "Energy + Agriculture + Metals + Mining + Transportation. "
                   "Tests if commodity sectors are coupling (war/sanctions effect).",
    "k": 5,
    "tickers": [
        # Energy
        "USO", "UNG",
        # Agriculture
        "DBA", "WEAT", "CORN",
        # Metals
        "GLD", "SLV", "CPER",
        # Mining
        "XME", "PICK",
        # Transportation (supply chain bottleneck)
        "IYT",
        # Rare earths (strategic)
        "REMX",
        # Fertilizers/chemicals (food security)
        "XLB",
    ],
}

ALL_UNIVERSES = {
    # Level 1
    "global_macro":       GLOBAL_MACRO,
    # Level 2
    "us_sectors":         REGIONAL_US,
    "europe":             REGIONAL_EUROPE,
    "asia_pacific":       REGIONAL_ASIA,
    "latam":              REGIONAL_LATAM,
    "brazil_sectors":     REGIONAL_BRAZIL,
    "mena":               REGIONAL_MENA,
    # Level 3
    "energy":             SECTOR_ENERGY,
    "tech_ai":            SECTOR_TECH_AI,
    "financials":         SECTOR_FINANCIALS,
    "commodities":        SECTOR_COMMODITIES,
    # Level 4 (thematic)
    "iran_war":           THEMATIC_IRAN_WAR,
    "ai_ecosystem":       THEMATIC_AI_ECOSYSTEM,
    "china_property":     THEMATIC_CHINA_PROPERTY,
    # Level 5 (cross-layer composites)
    "x_us_systemic":      CROSS_LAYER_US_SYSTEMIC,
    "x_energy_geopolitics": CROSS_LAYER_ENERGY_GEOPOLITICS,
    "x_europe_vuln":      CROSS_LAYER_EUROPE_VULNERABILITY,
    "x_global_contagion": CROSS_LAYER_GLOBAL_CONTAGION,
    "x_commodity_chain":  CROSS_LAYER_COMMODITY_CHAIN,
    "x_energy_tech":      CROSS_LAYER_ENERGY_TECH_TRANSMISSION,
    # Level 5b (Brazil cross-layer)
    "x_brazil_vuln":      CROSS_LAYER_BRAZIL_VULNERABILITY,
}

# Priority levels for monitoring frequency
PRIORITY = {
    "critical": ["global_macro", "x_global_contagion", "ai_ecosystem",
                 "brazil_sectors", "x_brazil_vuln"],                      # Brazil added March 20
    "high":     ["us_sectors", "energy", "tech_ai", "financials",
                 "x_us_systemic", "x_energy_geopolitics",
                 "iran_war"],                                             # every day — Iran War added March 19
    "medium":   ["europe", "asia_pacific", "commodities",
                 "x_europe_vuln", "x_commodity_chain",
                 "x_energy_tech",
                 "china_property"],                                       # every day — China Property added March 19
    "low":      ["latam", "mena"],                                        # every day
}


def get_all_unique_tickers() -> list:
    """Return deduplicated list of all tickers across all universes."""
    seen = set()
    result = []
    for u in ALL_UNIVERSES.values():
        for t in u["tickers"]:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return sorted(result)


def summary():
    """Print universe summary statistics."""
    all_tickers = get_all_unique_tickers()
    print(f"{'='*60}")
    print(f"  KAPPA SENTINEL — Universe Summary")
    print(f"  David Ohio | odavidohio@gmail.com")
    print(f"{'='*60}")
    print(f"\n  Total unique tickers: {len(all_tickers)}")
    print(f"\n  Universes ({len(ALL_UNIVERSES)}):")
    for uid, u in ALL_UNIVERSES.items():
        n = len(u['tickers'])
        print(f"    {uid:<25s} {n:>3d} tickers  — {u['name']}")
    print(f"\n  Priority levels:")
    for level, uids in PRIORITY.items():
        total = sum(len(ALL_UNIVERSES[uid]['tickers']) for uid in uids)
        print(f"    {level:<10s} {len(uids)} universes, {total} tickers")
    print(f"\n{'='*60}")
    return all_tickers


if __name__ == "__main__":
    tickers = summary()
    print(f"\n  All tickers ({len(tickers)}):")
    for i in range(0, len(tickers), 10):
        row = ", ".join(tickers[i:i+10])
        print(f"    {row}")

