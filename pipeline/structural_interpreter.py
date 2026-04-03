"""
Kappa Sentinel — Structural Interpreter
=========================================
Generates human-readable structural intelligence from Kappa-FIN output.
Explains WHY metrics matter, not just WHAT they are.

Key insight: the value of Kappa is not nu_s or Oh — it's understanding
what structural coupling MEANS for specific assets and portfolios.

David Ohio | odavidohio@gmail.com | March 2026
"""
import json, math, os, sys, io
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from config.universe import ALL_UNIVERSES

REPORTS_DIR = _ROOT / "data" / "reports"

# ══════════════════════════════════════════════════════════════════════════════
# ASSET ROLE TAXONOMY
# Each ticker classified by structural function within the network.
# The SAME ticker can have DIFFERENT roles in different universes.
# ══════════════════════════════════════════════════════════════════════════════

ASSET_ROLES = {
    # Safe havens — normally LOW correlation with risk assets
    # When these correlate with risk → diversification is broken
    "GLD":  {"role": "safe_haven",  "label": "Ouro",
             "normal": "Correlação baixa/negativa com ativos de risco",
             "anomaly": "Ouro correlacionando com risco = flight-to-safety generalizado ou fator macro dominante eliminando diversificação"},
    "SLV":  {"role": "safe_haven",  "label": "Prata",
             "normal": "Híbrido: industrial + refúgio",
             "anomaly": "Prata acompanhando risco = demanda industrial, não refúgio"},
    "TLT":  {"role": "safe_haven",  "label": "US Treasury 20y+",
             "normal": "Inversamente correlacionado com equities em stress",
             "anomaly": "Treasuries caindo junto com equities = mercado precificando inflação ou crise fiscal, não recessão"},
    "UUP":  {"role": "macro_driver", "label": "Dólar (DXY proxy)",
             "normal": "Dólar forte = pressão sobre EM e commodities",
             "anomaly": "Dólar forte + commodities fortes = choque de oferta (guerra/sanções), não demanda"},
    "VIXY": {"role": "fear_gauge",  "label": "VIX Futures",
             "normal": "Sobe em pânico, cai em complacência",
             "anomaly": "VIX baixo + Oh alto = complacência perigosa — mercado rígido sem precificar risco"},

    # Commodity bellwethers
    "USO":  {"role": "commodity",   "label": "Petróleo",
             "normal": "Ciclo de demanda global + geopolítica",
             "anomaly": "Petróleo acoplado com equities = macro override; acoplado com ouro = crise de oferta"},
    "UNG":  {"role": "commodity",   "label": "Gás Natural",
             "normal": "Sazonal + supply chain",
             "anomaly": "Gás correlacionando com equities europeus = dependência energética exposta"},
    "DBA":  {"role": "commodity",   "label": "Agricultura",
             "normal": "Sazonal + clima",
             "anomaly": "Agricultura acoplada com energia = crise de fertilizantes/logística"},

    # EM/Brazil bellwethers
    "EWZ":  {"role": "bellwether",  "label": "Brasil (iShares MSCI)",
             "normal": "Proxy Brasil — commodities + juros + fiscal",
             "anomaly": "EWZ acoplado com commodities globais = Brasil como canal de transmissão, não economia isolada"},
    "EEM":  {"role": "bellwether",  "label": "Emergentes",
             "normal": "Risk appetite global para EM",
             "anomaly": "EEM acoplado com treasuries = flight from EM; com commodities = macro override"},
    "EMB":  {"role": "contagion",   "label": "Dívida Soberana EM",
             "normal": "Spread de crédito EM vs developed",
             "anomaly": "EMB caindo junto com EEM = fuga de capital de EM generalizada"},
    "HYG":  {"role": "contagion",   "label": "High Yield US",
             "normal": "Apetite por risco de crédito",
             "anomaly": "HYG acoplado com EM debt = contágio crédito global"},

    # Brazilian ADRs
    "PBR":  {"role": "bellwether",  "label": "Petrobras",
             "normal": "Proxy energia BR + risco político estatal",
             "anomaly": "PBR descolando de petróleo global = risco político doméstico; acoplado = macro override"},
    "VALE": {"role": "bellwether",  "label": "Vale",
             "normal": "Proxy minério de ferro + demanda China",
             "anomaly": "VALE acoplando com financeiras BR = macro Brasil dominando, não fundamentos de commodity"},
    "ITUB": {"role": "bellwether",  "label": "Itaú Unibanco",
             "normal": "Proxy saúde financeira BR + Selic",
             "anomaly": "ITUB correlacionando com commodities = macro BR dominando; com EM debt = contágio financeiro"},
    "NU":   {"role": "growth",      "label": "Nubank",
             "normal": "Fintech/growth — correlação baixa com bancos tradicionais",
             "anomaly": "NU acoplando com ITUB/BBD = mercado tratando fintech como banco, não como tech"},
    "ABEV": {"role": "defensive",   "label": "Ambev",
             "normal": "Consumo defensivo — proxy poder de compra do brasileiro",
             "anomaly": "ABEV caindo junto com risco = consumidor brasileiro sob pressão"},

    # Sector proxies
    "XLF":  {"role": "sector",      "label": "Financials US",
             "normal": "Proxy sistema bancário US",
             "anomaly": "XLF acoplando com EM = contágio cross-border; com energia = macro override"},
    "XLE":  {"role": "sector",      "label": "Energia US",
             "normal": "Proxy upstream/downstream energia",
             "anomaly": "XLE descolando de USO = rotação setorial; acoplando com defesa = geopolítica"},
    "XLK":  {"role": "sector",      "label": "Tecnologia US",
             "normal": "Growth/momentum",
             "anomaly": "XLK acoplando com TLT = duration trade dominante; com VIXY = risk-off tech"},

    # Defense
    "LMT":  {"role": "thematic",    "label": "Lockheed Martin",
             "normal": "Defesa — correlação baixa com mercado amplo",
             "anomaly": "Defesa acoplando com energia = precificação de conflito; com SPY = rally de guerra"},
}


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def load_state(universe_id: str) -> Optional[pd.DataFrame]:
    """Load Kappa state CSV for a universe."""
    path = REPORTS_DIR / f"sentinel_{universe_id}" / "kappa_v4_state.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def compute_recent_correlations(universe_id: str, window: int = 22) -> Optional[pd.DataFrame]:
    """Compute correlation matrix from recent returns."""
    universe = ALL_UNIVERSES.get(universe_id)
    if not universe:
        return None
    tickers = universe["tickers"]

    # Load viscosity CSV which has per-ticker data
    visc_path = REPORTS_DIR / f"sentinel_{universe_id}" / "kappa_v4_viscosity.csv"
    if not visc_path.exists():
        return None
    try:
        df = pd.read_csv(visc_path, parse_dates=["date"])
        # Get last `window` rows for recent correlation
        recent = df.tail(window)
        # Extract Oh and eta columns if available
        return recent
    except Exception:
        return None


def classify_coupling_meaning(universe_id: str, status: str, oh_max: float,
                                nu_s: float, regime: str, tickers: list) -> List[str]:
    """
    Generate structural insights based on universe composition + Kappa metrics.
    This is the core intellectual contribution: what does coupling MEAN?
    """
    insights = []
    roles_present = {}
    for t in tickers:
        if t in ASSET_ROLES:
            role_info = ASSET_ROLES[t]
            roles_present.setdefault(role_info["role"], []).append(
                {"ticker": t, **role_info}
            )

    # ── RULE 1: Safe havens in a CRITICAL/PRESSURIZED universe ──
    safe_havens = roles_present.get("safe_haven", [])
    if safe_havens and status in ("CRITICAL", "PRESSURIZED"):
        names = ", ".join(a["label"] for a in safe_havens)
        insights.append({
            "type": "anomaly",
            "severity": "high" if status == "CRITICAL" else "medium",
            "title": f"Ativos de refúgio acoplados ao risco",
            "body": (f"{names} estão dentro de uma rede com acoplamento anormal "
                     f"(Oh={oh_max:.3f}). Em condições normais, estes ativos têm "
                     f"correlação baixa ou negativa com ativos de risco. "
                     f"Quando se movem juntos, a diversificação está quebrada — "
                     f"um fator macro (dólar, juros, geopolítica) está dominando tudo."),
            "implication": ("Hedges tradicionais (ouro, treasuries) podem não proteger "
                           "durante a próxima transição de regime. Considerar "
                           "alternativas de proteção não-correlacionadas.")
        })

    # ── RULE 2: Bellwethers all coupled → macro override ──
    bellwethers = roles_present.get("bellwether", [])
    if len(bellwethers) >= 2 and status in ("CRITICAL", "PRESSURIZED"):
        names = ", ".join(a["label"] for a in bellwethers)
        insights.append({
            "type": "structural",
            "severity": "high",
            "title": "Bellwethers em acoplamento — macro override",
            "body": (f"{names} estão se movendo juntos de forma anormal. "
                     f"Quando bellwethers de setores diferentes se acoplam, "
                     f"significa que um fator macro está dominando e os "
                     f"fundamentos individuais não importam mais."),
            "implication": ("Análise fundamentalista perde poder preditivo "
                           "neste regime. O mercado está respondendo a "
                           "narrativa macro, não a resultados/earnings.")
        })

    # ── RULE 3: Contagion channels active ──
    contagion = roles_present.get("contagion", [])
    if contagion and status in ("CRITICAL", "PRESSURIZED"):
        names = ", ".join(a["label"] for a in contagion)
        insights.append({
            "type": "contagion",
            "severity": "high" if oh_max > 1.2 else "medium",
            "title": "Canais de contágio ativados",
            "body": (f"{names} estão acoplados à rede em regime rígido. "
                     f"Estes instrumentos são os canais pelos quais estresse "
                     f"de um setor/região se propaga para outros."),
            "implication": ("Monitorar a velocidade de transmissão: se estes "
                           "canais passarem de PRESSURIZED para CRITICAL em "
                           "outro universo, o risco se torna sistêmico.")
        })

    # ── RULE 4: Growth/defensive convergence ──
    growth = roles_present.get("growth", [])
    defensive = roles_present.get("defensive", [])
    if growth and defensive and status in ("CRITICAL", "PRESSURIZED"):
        g_names = ", ".join(a["label"] for a in growth)
        d_names = ", ".join(a["label"] for a in defensive)
        insights.append({
            "type": "anomaly",
            "severity": "medium",
            "title": "Growth e defensivos convergindo",
            "body": (f"Ativos de crescimento ({g_names}) e defensivos ({d_names}) "
                     f"estão acoplados. Normalmente estes setores divergem — "
                     f"growth sobe em risk-on, defensivos em risk-off. "
                     f"Convergência indica que o mercado perdeu capacidade de "
                     f"discriminar entre estilos."),
            "implication": ("Rotação setorial está travada. Quando destrava, "
                           "a divergência tende a ser brusca.")
        })

    # ── RULE 5: Katashi regime persistence ──
    if regime == "Katashi" and nu_s > 100:
        insights.append({
            "type": "regime",
            "severity": "high" if nu_s > 500 else "medium",
            "title": f"Regime Katashi persistente (ν_s = {nu_s:.0f})",
            "body": ("A rede está congelada — correlações maximamente rígidas. "
                     "Parece estável, mas é fragilidade estrutural (Princípio "
                     "de Katashi: instabilidade emerge de coerência excessiva, "
                     "não de ruído). Quanto mais tempo persiste, mais violenta "
                     "tende a ser a transição."),
            "implication": ("Não confundir rigidez com segurança. Volatilidade "
                           "baixa neste regime é ilusória — a energia estrutural "
                           "está se acumulando, não se dissipando.")
        })

    # ── RULE 6: Cross-layer HEALTHY while sectoral CRITICAL ──
    # (This is detected at the ecosystem level, not per-universe)

    # ── RULE 7: Fear gauge complacency ──
    fear = roles_present.get("fear_gauge", [])
    if fear and status == "HEALTHY" and oh_max > 0.7:
        insights.append({
            "type": "warning",
            "severity": "low",
            "title": "VIX contido mas estrutura enrijecendo",
            "body": ("O índice de medo (VIX) permanece contido enquanto a "
                     "estrutura topológica se rigidifica. O mercado não está "
                     "precificando o risco estrutural que o Kappa detecta."),
            "implication": ("Historicamente, divergência VIX vs Oh precede "
                           "correções abruptas. O mercado pode estar "
                           "subestimando risco.")
        })

    # ── RULE 8: Commodity + Currency coupling ──
    commodities = roles_present.get("commodity", [])
    macro_drivers = roles_present.get("macro_driver", [])
    if commodities and macro_drivers and status in ("CRITICAL", "PRESSURIZED"):
        c_names = ", ".join(a["label"] for a in commodities)
        m_names = ", ".join(a["label"] for a in macro_drivers)
        insights.append({
            "type": "structural",
            "severity": "medium",
            "title": "Commodities acopladas a drivers macro",
            "body": (f"{c_names} estão correlacionados com {m_names}. "
                     f"Quando commodities se movem por dinâmica de câmbio/juros "
                     f"ao invés de oferta/demanda física, os modelos tradicionais "
                     f"de precificação perdem validade."),
            "implication": ("Análise de fundamentos de commodity (estoque, "
                           "produção, clima) pode estar sendo sobrepujada "
                           "por fluxo financeiro.")
        })

    return insights


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-UNIVERSE STRUCTURAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def cross_universe_insights(reports: list) -> List[dict]:
    """Detect patterns ACROSS universes that individual analysis misses."""
    insights = []
    by_uid = {r["universe"]: r for r in reports if not r.get("error")}

    # ── Sectoral CRITICAL but cross-layer HEALTHY ──
    # Pattern: internal fragility not propagating
    pairs = [
        ("brazil_sectors", "x_brazil_vuln", "Brasil"),
        ("us_sectors", "x_us_systemic", "EUA"),
        ("europe", "x_europe_vuln", "Europa"),
        ("energy", "x_energy_geopolitics", "Energia"),
        ("commodities", "x_commodity_chain", "Commodities"),
    ]
    for sectoral, cross, label in pairs:
        s = by_uid.get(sectoral, {})
        c = by_uid.get(cross, {})
        if not s or not c:
            continue
        s_status = s.get("status", "HEALTHY")
        c_status = c.get("status", "HEALTHY")

        if s_status in ("CRITICAL", "PRESSURIZED") and c_status == "HEALTHY":
            insights.append({
                "type": "containment",
                "severity": "info",
                "title": f"{label}: estresse interno CONTIDO",
                "body": (f"O universo setorial {label} está {s_status} "
                         f"(ν_s={s.get('nu_s',0):.0f}), mas o cross-layer "
                         f"está HEALTHY (ν_s={c.get('nu_s',0):.0f}). "
                         f"A fragilidade é endógena — não está propagando "
                         f"para canais globais."),
                "implication": (f"Monitorar {cross}: se transitar para "
                               f"PRESSURIZED, o risco {label} deixa de ser "
                               f"doméstico e se torna sistêmico.")
            })

        elif s_status in ("CRITICAL", "PRESSURIZED") and c_status in ("CRITICAL", "PRESSURIZED"):
            insights.append({
                "type": "contagion",
                "severity": "critical",
                "title": f"{label}: PROPAGAÇÃO ATIVA",
                "body": (f"ATENÇÃO: tanto o setorial quanto o cross-layer {label} "
                         f"estão em estado de alerta ({s_status} / {c_status}). "
                         f"O estresse está propagando além das fronteiras "
                         f"setoriais/regionais."),
                "implication": (f"Risco sistêmico ativo. Verificar x_global_contagion "
                               f"para avaliar se o contágio é regional ou global.")
            })

    # ── Global contagion check ──
    gc = by_uid.get("x_global_contagion", {})
    if gc and gc.get("status") in ("CRITICAL", "PRESSURIZED"):
        n_critical = sum(1 for r in reports if r.get("status") == "CRITICAL")
        insights.append({
            "type": "systemic",
            "severity": "critical",
            "title": "CONTÁGIO GLOBAL DETECTADO",
            "body": (f"O mapa de contágio global está {gc.get('status')} "
                     f"(ν_s={gc.get('nu_s',0):.0f}, Oh={gc.get('oh_max',0):.3f}). "
                     f"{n_critical} universos em estado CRITICAL. "
                     f"O sistema financeiro global está se movendo como "
                     f"uma unidade — diversificação geográfica e setorial "
                     f"está efetivamente eliminada."),
            "implication": ("Cenário de correlação-1: apenas caixa e "
                           "posições não-correlacionadas protegem.")
        })

    return insights


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTERPRETATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def interpret_all(summary_path: str = None) -> dict:
    """
    Run full structural interpretation on all universes.
    Returns dict of {universe_id: [insights], "_cross": [cross-insights]}.
    """
    if summary_path is None:
        summary_path = str(REPORTS_DIR / "sentinel_summary.json")

    with open(summary_path, "r") as f:
        raw = f.read().replace("NaN", "null")
        summary = json.loads(raw)

    result = {}
    for report in summary["reports"]:
        if report.get("error"):
            continue
        uid = report["universe"]
        universe = ALL_UNIVERSES.get(uid, {})
        tickers = universe.get("tickers", [])

        insights = classify_coupling_meaning(
            universe_id=uid,
            status=report.get("status", "HEALTHY"),
            oh_max=report.get("oh_max", 0),
            nu_s=report.get("nu_s", 0),
            regime=report.get("regime", "Nagare"),
            tickers=tickers,
        )
        if insights:
            result[uid] = insights

    # Cross-universe analysis
    cross = cross_universe_insights(summary["reports"])
    if cross:
        result["_cross"] = cross

    return result


def enrich_cockpit_with_insights(cockpit_path: str = None):
    """Add structural insights to existing cockpit JSON."""
    if cockpit_path is None:
        cockpit_path = str(_ROOT / "dashboard" / "public" / "sentinel_cockpit.json")

    with open(cockpit_path, "r", encoding="utf-8") as f:
        cockpit = json.load(f)

    interpretations = interpret_all()

    # Add per-universe insights
    for uid, insights in interpretations.items():
        if uid == "_cross":
            continue
        if uid in cockpit.get("universes", {}):
            cockpit["universes"][uid]["structural_insights"] = insights

    # Add cross-universe insights at top level
    if "_cross" in interpretations:
        cockpit["cross_universe_insights"] = interpretations["_cross"]

    with open(cockpit_path, "w", encoding="utf-8") as f:
        json.dump(cockpit, f, indent=2, ensure_ascii=False)

    # Stats
    total = sum(len(v) for k, v in interpretations.items())
    print(f"  Structural insights: {total} total")
    for uid, insights in interpretations.items():
        label = uid if uid != "_cross" else "CROSS-UNIVERSE"
        for ins in insights:
            try:
                print(f"    [{ins['severity']:>8s}] {label}: {ins['title']}")
            except UnicodeEncodeError:
                safe_title = ins['title'].encode('ascii', 'replace').decode()
                print(f"    [{ins['severity']:>8s}] {label}: {safe_title}")

    return interpretations


if __name__ == "__main__":
    print("=" * 60)
    print("  Kappa Sentinel — Structural Interpreter")
    print("  David Ohio | odavidohio@gmail.com")
    print("=" * 60)
    enrich_cockpit_with_insights()
    print("\n  Done!")
