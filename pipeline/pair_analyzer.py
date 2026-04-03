"""
Kappa Sentinel — Pair-Level Correlation Analyzer
==================================================
Computes per-pair correlations and identifies anomalous couplings.
This is the data-driven layer that makes structural interpretation
move from generic rules to specific, actionable insights.

The key insight: Kappa-FIN tells you the NETWORK is rigid.
This module tells you WHICH PAIRS are driving that rigidity
and WHAT THAT MEANS for each specific pair.

David Ohio | odavidohio@gmail.com | March 2026
"""
import sys, os, io, json, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from config.universe import ALL_UNIVERSES

REPORTS_DIR = _ROOT / "data" / "reports"
CACHE_DIR = _ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_returns(tickers: list, start: str = "2022-01-01") -> Optional[pd.DataFrame]:
    """Load daily returns for a set of tickers (with caching)."""
    import yfinance as yf
    end = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"returns_{'_'.join(sorted(tickers)[:5])}_{len(tickers)}_{end}"
    cache_path = CACHE_DIR / f"{hash(cache_key) & 0xFFFFFFFF:08x}.parquet"

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if set(tickers).issubset(df.columns):
                return df[tickers]
        except Exception:
            pass

    prices = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if prices.empty:
        return None
    returns = prices.pct_change().dropna()
    try:
        returns.to_parquet(cache_path)
    except Exception:
        pass
    return returns


def compute_pair_correlations(returns: pd.DataFrame, window: int = 22) -> pd.DataFrame:
    """Compute rolling correlation for all pairs in the last window."""
    recent = returns.tail(window)
    corr = recent.corr()
    return corr


def compute_calm_correlations(returns: pd.DataFrame,
                               calm_start: str = "2022-06-01",
                               calm_end: str = "2023-06-01") -> pd.DataFrame:
    """Compute average correlation during CALM period."""
    mask = (returns.index >= calm_start) & (returns.index <= calm_end)
    calm = returns.loc[mask]
    if len(calm) < 22:
        calm = returns.head(252)  # fallback: first year
    return calm.corr()


def find_anomalous_pairs(returns: pd.DataFrame,
                          calm_start: str = "2022-06-01",
                          calm_end: str = "2023-06-01",
                          window: int = 22,
                          threshold: float = 0.3) -> List[dict]:
    """
    The core function: find pairs whose correlation changed most
    vs the CALM baseline.

    A pair going from corr=0.1 (CALM) to corr=0.7 (now) is anomalous.
    A pair staying at corr=0.8 in both periods is just normally coupled.

    threshold: minimum absolute change in correlation to flag.
    """
    corr_calm = compute_calm_correlations(returns, calm_start, calm_end)
    corr_now = compute_pair_correlations(returns, window)

    tickers = returns.columns.tolist()
    pairs = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            c_calm = corr_calm.loc[t1, t2] if t1 in corr_calm.index and t2 in corr_calm.columns else 0
            c_now = corr_now.loc[t1, t2] if t1 in corr_now.index and t2 in corr_now.columns else 0

            if np.isnan(c_calm) or np.isnan(c_now):
                continue

            delta = c_now - c_calm
            if abs(delta) >= threshold:
                pairs.append({
                    "ticker_1": t1,
                    "ticker_2": t2,
                    "corr_calm": round(float(c_calm), 3),
                    "corr_now": round(float(c_now), 3),
                    "delta": round(float(delta), 3),
                    "direction": "coupling" if delta > 0 else "decoupling",
                })

    # Sort by absolute delta descending
    pairs.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return pairs


# Import asset roles from structural interpreter
from src.kappa.structural_interpreter import ASSET_ROLES


def interpret_pair(t1: str, t2: str, corr_calm: float, corr_now: float, delta: float) -> Optional[str]:
    """
    Generate a human-readable interpretation of WHY this pair coupling matters.
    This is where domain knowledge meets data.
    """
    r1 = ASSET_ROLES.get(t1, {})
    r2 = ASSET_ROLES.get(t2, {})
    role1 = r1.get("role", "unknown")
    role2 = r2.get("role", "unknown")
    label1 = r1.get("label", t1)
    label2 = r2.get("label", t2)

    direction = "se acoplando" if delta > 0 else "se desacoplando"
    strength = "fortemente" if abs(delta) > 0.5 else "moderadamente"

    # ── Safe haven + risk asset coupling ──
    safe_roles = {"safe_haven"}
    risk_roles = {"bellwether", "sector", "commodity", "growth", "contagion"}
    if (role1 in safe_roles and role2 in risk_roles) or (role2 in safe_roles and role1 in risk_roles):
        safe = label1 if role1 in safe_roles else label2
        risk = label1 if role1 in risk_roles else label2
        if delta > 0:
            return (f"{safe} está {direction} {strength} com {risk} "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Ativo de refúgio perdendo propriedade defensiva — "
                    f"diversificação via {safe} está comprometida neste regime.")
        else:
            return (f"{safe} está {direction} de {risk} "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Refúgio recuperando independência — bom sinal de normalização.")

    # ── Bellwether cross-sector coupling ──
    if role1 == "bellwether" and role2 == "bellwether":
        if delta > 0:
            return (f"{label1} e {label2} estão convergindo {strength} "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Bellwethers de setores diferentes se acoplando indica "
                    f"fator macro dominando — fundamentos individuais perdem relevância.")
        else:
            return (f"{label1} e {label2} estão divergindo "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Setores recuperando dinâmica própria — mercado voltando a discriminar.")

    # ── Commodity + macro driver ──
    if (role1 == "commodity" and role2 == "macro_driver") or (role2 == "commodity" and role1 == "macro_driver"):
        comm = label1 if role1 == "commodity" else label2
        macro = label1 if role1 == "macro_driver" else label2
        if delta > 0:
            return (f"{comm} está respondendo a {macro} mais do que ao fundamento físico "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Precificação financeira dominando oferta/demanda.")

    # ── Contagion channel activation ──
    if "contagion" in (role1, role2):
        contagion_label = label1 if role1 == "contagion" else label2
        other_label = label2 if role1 == "contagion" else label1
        if delta > 0:
            return (f"Canal de contágio {contagion_label} está se acoplando com {other_label} "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Estresse pode estar propagando entre setores/regiões via este canal.")

    # ── Growth + Defensive convergence ──
    if ("growth" in (role1, role2)) and ("defensive" in (role1, role2)):
        if delta > 0:
            return (f"{label1} (growth) e {label2} (defensivo) convergindo "
                    f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}). "
                    f"Mercado perdendo capacidade de discriminar estilos — "
                    f"risk-on e risk-off se movendo juntos.")

    # ── Generic significant change ──
    if abs(delta) > 0.4:
        return (f"{label1} e {label2} mudaram {strength} "
                f"(CALM: {corr_calm:.2f} → agora: {corr_now:.2f}, Δ={delta:+.2f}). "
                f"Mudança estrutural significativa neste par.")

    return None


def analyze_universe_pairs(universe_id: str, top_n: int = 10) -> dict:
    """
    Full pair-level analysis for a universe.
    Returns top anomalous pairs with data-driven interpretations.
    """
    universe = ALL_UNIVERSES.get(universe_id)
    if not universe:
        return {"error": f"Universe {universe_id} not found"}

    tickers = universe["tickers"]
    name = universe["name"]

    print(f"  {universe_id} ({len(tickers)} tickers)...", end=" ", flush=True)

    returns = load_returns(tickers)
    if returns is None or returns.empty:
        print("NO DATA")
        return {"error": "No return data"}

    # Find anomalous pairs
    anomalous = find_anomalous_pairs(returns, threshold=0.25)
    top = anomalous[:top_n]

    # Generate interpretations
    interpreted = []
    for pair in top:
        interp = interpret_pair(
            pair["ticker_1"], pair["ticker_2"],
            pair["corr_calm"], pair["corr_now"], pair["delta"]
        )
        interpreted.append({**pair, "interpretation": interp})

    # Summary stats
    n_coupling = sum(1 for p in anomalous if p["direction"] == "coupling")
    n_decoupling = sum(1 for p in anomalous if p["direction"] == "decoupling")
    avg_delta = np.mean([abs(p["delta"]) for p in anomalous]) if anomalous else 0

    print(f"{len(anomalous)} anomalous pairs ({n_coupling} coupling, {n_decoupling} decoupling)")

    return {
        "universe": universe_id,
        "name": name,
        "tickers": tickers,
        "n_anomalous": len(anomalous),
        "n_coupling": n_coupling,
        "n_decoupling": n_decoupling,
        "avg_delta": round(float(avg_delta), 3),
        "top_pairs": interpreted,
    }


def enrich_cockpit_with_pairs(cockpit_path: str = None,
                               universes: list = None):
    """Add pair-level analysis to cockpit JSON."""
    if cockpit_path is None:
        cockpit_path = str(_ROOT / "dashboard" / "public" / "sentinel_cockpit.json")

    with open(cockpit_path, "r", encoding="utf-8") as f:
        cockpit = json.load(f)

    if universes is None:
        universes = list(cockpit.get("universes", {}).keys())

    for uid in universes:
        if uid not in cockpit.get("universes", {}):
            continue
        result = analyze_universe_pairs(uid, top_n=8)
        if "error" not in result:
            cockpit["universes"][uid]["pair_analysis"] = {
                "n_anomalous": result["n_anomalous"],
                "n_coupling": result["n_coupling"],
                "n_decoupling": result["n_decoupling"],
                "avg_delta": result["avg_delta"],
                "top_pairs": result["top_pairs"],
            }

    with open(cockpit_path, "w", encoding="utf-8") as f:
        json.dump(cockpit, f, indent=2, ensure_ascii=False)

    print(f"\n  Pair analysis saved to cockpit")


if __name__ == "__main__":
    print("=" * 60)
    print("  Kappa Sentinel — Pair-Level Correlation Analyzer")
    print("  David Ohio | odavidohio@gmail.com")
    print("=" * 60)
    # Analyze key universes
    for uid in ["brazil_sectors", "x_brazil_vuln", "global_macro", "commodities"]:
        result = analyze_universe_pairs(uid)
        if "error" in result:
            continue
        print(f"\n  Top anomalous pairs for {result['name']}:")
        for p in result["top_pairs"][:5]:
            interp = p.get("interpretation", "")
            short = interp[:100] + "..." if interp and len(interp) > 100 else (interp or "N/A")
            print(f"    {p['ticker_1']:5s} ↔ {p['ticker_2']:5s}  "
                  f"CALM={p['corr_calm']:+.2f} → NOW={p['corr_now']:+.2f} "
                  f"(Δ={p['delta']:+.3f})  {p['direction']}")
            if interp:
                print(f"      → {short}")
    print("\n  Done!")
