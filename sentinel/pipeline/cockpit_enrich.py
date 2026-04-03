"""
Kappa Sentinel — Cockpit Enrichment
Computes derived metrics for the fund manager cockpit view.

David Ohio | odavidohio@gmail.com | March 2026
"""
import json, math, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from config.universe import ALL_UNIVERSES

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reports"
SUMMARY_PATH = REPORTS_DIR / "sentinel_summary.json"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "dashboard" / "public" / "sentinel_cockpit.json"

def sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def load_state_csv(universe_id):
    path = REPORTS_DIR / f"sentinel_{universe_id}" / "kappa_v4_state.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def compute_oh_trend(df, days=7):
    """Compute Oh trend: direction + delta over last N days."""
    if df is None or len(df) < days + 1:
        return {"direction": "stable", "delta_7d": 0, "delta_30d": 0, "current": 0}
    recent = df.tail(days)
    oh_now = float(df["Oh"].iloc[-1])
    oh_7d_ago = float(df["Oh"].iloc[-days]) if len(df) >= days else oh_now
    oh_30d_ago = float(df["Oh"].iloc[-30]) if len(df) >= 30 else oh_now
    delta_7 = oh_now - oh_7d_ago
    delta_30 = oh_now - oh_30d_ago
    if delta_7 > 0.05:
        direction = "rising"
    elif delta_7 < -0.05:
        direction = "falling"
    else:
        direction = "stable"
    return {"direction": direction, "delta_7d": round(delta_7, 4),
            "delta_30d": round(delta_30, 4), "current": round(oh_now, 4)}


def compute_regime_duration(df):
    """How many consecutive days the current regime has been active."""
    if df is None or len(df) < 2:
        return {"current_regime": "?", "duration_days": 0, "avg_duration": 0, "max_duration": 0}
    current = df["regime"].iloc[-1]
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df["regime"].iloc[i] == current:
            count += 1
        else:
            break
    # Historical durations for this regime
    runs = []
    run_len = 0
    prev = None
    for r in df["regime"]:
        if r == current:
            run_len += 1
        else:
            if run_len > 0:
                runs.append(run_len)
            run_len = 0
        prev = r
    if run_len > 0:
        runs.append(run_len)
    avg_d = round(np.mean(runs), 1) if runs else 0
    max_d = max(runs) if runs else 0
    return {"current_regime": str(current), "duration_days": count,
            "avg_duration": avg_d, "max_duration": max_d,
            "above_average": count > avg_d}


def compute_oh_percentile(df):
    """Where does current Oh sit in the historical distribution?"""
    if df is None or len(df) < 10:
        return {"percentile": None, "mean": 0, "std": 0}
    oh_now = float(df["Oh"].iloc[-1])
    pct = float((df["Oh"] < oh_now).mean() * 100)
    return {"percentile": round(pct, 1), "mean": round(float(df["Oh"].mean()), 4),
            "std": round(float(df["Oh"].std()), 4)}


def find_historical_spikes(df, threshold=1.0):
    """Find past Oh spikes above threshold."""
    if df is None or len(df) < 10:
        return []
    spikes = []
    in_spike = False
    spike_start = None
    spike_peak = 0
    for _, row in df.iterrows():
        oh = row["Oh"]
        if oh > threshold and not in_spike:
            in_spike = True
            spike_start = row["date"]
            spike_peak = oh
        elif oh > threshold and in_spike:
            spike_peak = max(spike_peak, oh)
        elif oh <= threshold and in_spike:
            spikes.append({
                "start": spike_start.strftime("%Y-%m-%d"),
                "peak": round(float(spike_peak), 4),
                "duration_days": (row["date"] - spike_start).days,
            })
            in_spike = False
    if in_spike:
        spikes.append({
            "start": spike_start.strftime("%Y-%m-%d"),
            "peak": round(float(spike_peak), 4),
            "duration_days": (df["date"].iloc[-1] - spike_start).days,
            "active": True,
        })
    return spikes[-5:]  # last 5 spikes


def generate_executive_summary(report, oh_trend, regime_dur):
    """Resumo executivo de uma frase para um universo."""
    name = report["name"]
    status = report["status"]
    oh = report["oh_max"]
    nu_s = report.get("nu_s", 0)
    regime = report["regime"]
    pr = report.get("pr")
    tau = regime_dur.get("duration_days", 0)

    if status == "CRITICAL":
        if oh > 1.5:
            return f"Spike estrutural ativo (Oh={oh:.3f}). Rede em sincronia forcada. Quando resolver, espere dislocacao de preco."
        elif regime == "Katashi" and nu_s > 1000:
            return f"Regime Katashi ha {tau}d com viscosidade extrema (nu_s={nu_s:.0f}). Aparenta estabilidade mas fragilidade estrutural profunda."
        elif regime == "Katashi" and tau > 100:
            return f"Congelado ha {tau}d em Katashi. Oh pico {oh:.3f}. Rigidez acumulando — transicao tende a ser violenta."
        else:
            return f"Oh atingiu pico de {oh:.3f}. {'Acumulo endogeno (PR=' + str(round(pr*100)) + '%)' if pr and pr > 0.5 else 'Choque exogeno'}. Risco elevado."
    elif status == "PRESSURIZED":
        if regime == "Katashi" and tau > 300:
            return f"Congelado ha {tau}d em Katashi. Ossificacao estrutural (nu_s={nu_s:.0f}) — quando quebrar, espere mudanca violenta de regime."
        elif regime_dur["duration_days"] > 100:
            return f"Congelado ha {regime_dur['duration_days']}d em {regime}. Ossificacao estrutural — quando quebrar, espere mudanca violenta de regime."
        else:
            return f"Pressao acumulando (nu_s={report['nu_s']:.0f}). {regime_dur['duration_days']}d em {regime}. Monitorar escalacao."
    else:
        if oh_trend["direction"] == "rising":
            return f"Saudavel mas Oh em tendencia de alta ({oh_trend['delta_7d']:+.3f} em 7d). Observar transicao de regime."
        return f"Estruturalmente estavel. Regime {regime}, sem pressao."


# Ticker descriptions for fund managers
TICKER_INFO = {
    "SPY":"S&P 500","QQQ":"Nasdaq 100","IWM":"US Small Cap","EFA":"Dev ex-US","EEM":"EM Equities",
    "FXI":"China Large Cap","EWJ":"Japan","EWZ":"Brazil","TLT":"US Treasury 20y+","IEF":"US Treasury 7-10y",
    "SHY":"US Treasury 1-3y","LQD":"US IG Corp","HYG":"US HY Corp","EMB":"EM Sov Debt","GLD":"Gold",
    "SLV":"Silver","USO":"Crude Oil","UNG":"Natural Gas","DBA":"Agriculture","UUP":"USD Index",
    "VIXY":"VIX Futures","XLF":"US Financials","XLE":"US Energy","XLK":"US Tech","XLV":"US Healthcare",
    "XLI":"US Industrials","XLC":"US Comm Svcs","XLY":"US Cons Disc","XLP":"US Cons Staples",
    "XLU":"US Utilities","XLB":"US Materials","XLRE":"US Real Estate","IYT":"US Transport",
    "KBE":"US Banks","XHB":"US Homebuilders","EZU":"Eurozone","EWG":"Germany","EWQ":"France",
    "EWI":"Italy","EWP":"Spain","EWU":"UK","EWL":"Switzerland","EWD":"Sweden","ENOR":"Norway",
    "GREK":"Greece","EPOL":"Poland","TUR":"Turkey","KWEB":"China Internet","EWT":"Taiwan",
    "EWY":"South Korea","INDA":"India","EWA":"Australia","EWS":"Singapore","THD":"Thailand",
    "VNM":"Vietnam","EPHE":"Philippines","IDX":"Indonesia","EWW":"Mexico","ECH":"Chile",
    "ARGT":"Argentina","GXG":"Colombia","EPU":"Peru","ILF":"LatAm 40","KSA":"Saudi Arabia",
    "UAE":"UAE","QAT":"Qatar","EGPT":"Egypt","NGE":"Nigeria","EZA":"South Africa","AFK":"Africa",
    "XOP":"US Oil E&P","OIH":"Oil Services","AMLP":"MLPs","ICLN":"Clean Energy","TAN":"Solar",
    "LIT":"Lithium/Battery","URA":"Uranium","XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips",
    "SLB":"Schlumberger","SMH":"Semiconductors","SOXX":"Semis Index","IGV":"Software","SKYY":"Cloud",
    "BOTZ":"Robotics/AI","ARKK":"ARK Innovation","NVDA":"NVIDIA","MSFT":"Microsoft","GOOGL":"Google",
    "META":"Meta","AMD":"AMD","AVGO":"Broadcom","TSM":"TSMC","KRE":"US Regional Banks",
    "IAK":"Insurance","EUFN":"Europe Financials","FINX":"Fintech","IBIT":"Bitcoin ETF",
    "ETHA":"Ethereum ETF","JPM":"JPMorgan","GS":"Goldman Sachs","BAC":"Bank of America",
    "HSBC":"HSBC","PPLT":"Platinum","CPER":"Copper","WEAT":"Wheat","CORN":"Corn","SOYB":"Soybeans",
    "XME":"Metals/Mining","PICK":"Global Mining","WOOD":"Timber","REMX":"Rare Earth",
    "LMT":"Lockheed Martin","RTX":"RTX/Raytheon","NOC":"Northrop Grumman","GD":"General Dynamics",
    "HAL":"Halliburton","OXY":"Occidental","DVN":"Devon Energy","FANG":"Diamondback Energy",
    "SMCI":"Super Micro","ARM":"ARM Holdings","PLTR":"Palantir","SNOW":"Snowflake","AI":"C3.ai",
    "VST":"Vistra","CEG":"Constellation Energy","EQIX":"Equinix","MCHI":"China All Cap",
    "CHIQ":"China Cons Disc","GXC":"China Total","EWH":"Hong Kong","EMLC":"EM Local Currency",
    "AMAT":"Applied Materials",
    # Brazilian ADRs
    "PBR":"Petrobras","VALE":"Vale","GGB":"Gerdau","SID":"CSN","SBS":"SABESP","CIG":"CEMIG",
    "ITUB":"Itau Unibanco","BBD":"Bradesco","BSBR":"Santander Brasil","NU":"Nubank",
    "XP":"XP Inc","ABEV":"Ambev","STNE":"StoneCo","PAGS":"PagSeguro","EWZS":"Brazil Small-Cap",
}

def get_composition(uid):
    """Get ticker composition for a universe."""
    univ = ALL_UNIVERSES.get(uid)
    if not univ:
        return None
    tickers = univ.get("tickers", [])
    return {
        "description": univ.get("description", ""),
        "n_tickers": len(tickers),
        "tickers": [{"symbol": t, "name": TICKER_INFO.get(t, t)} for t in tickers],
        "level": "Cross-Layer" if uid.startswith("x_") else
                 "Temático" if uid in ("iran_war","ai_ecosystem","china_property") else
                 "Setorial" if uid in ("energy","tech_ai","financials","commodities") else
                 "Regional" if uid in ("us_sectors","europe","asia_pacific","latam","mena") else
                 "Macro",
    }


def compute_global_risk_score(reports, enrichments):
    """Single 0-100 number summarizing global structural risk."""
    status_mult = {"CRITICAL": 3.0, "PRESSURIZED": 2.0, "HEALTHY": 1.0}
    weighted_sum = 0
    weight_total = 0
    for r in reports:
        if r.get("error"):
            continue
        uid = r["universe"]
        oh = r.get("oh_max", 0) or 0
        mult = status_mult.get(r["status"], 1.0)
        w = 1.0
        weighted_sum += w * oh * mult
        weight_total += w
    raw = weighted_sum / weight_total if weight_total > 0 else 0
    score = min(100, raw * 25)  # scale: Oh=1 * HEALTHY = 25, Oh=1.5 * CRITICAL = 112 → capped at 100
    return round(score, 1)


def generate_cockpit():
    """Generate the complete cockpit enrichment JSON."""
    with open(SUMMARY_PATH) as f:
        raw = f.read().replace("NaN", "null")
        summary = json.loads(raw)

    reports = [r for r in summary["reports"] if not r.get("error")]
    enrichments = {}

    for r in reports:
        uid = r["universe"]
        print(f"  {uid}...", end=" ", flush=True)
        df = load_state_csv(uid)
        oh_trend = compute_oh_trend(df)
        regime_dur = compute_regime_duration(df)
        oh_pct = compute_oh_percentile(df)
        spikes = find_historical_spikes(df)
        summary_text = generate_executive_summary(r, oh_trend, regime_dur)

        enrichments[uid] = {
            "oh_trend": oh_trend,
            "regime_duration": regime_dur,
            "oh_percentile": oh_pct,
            "historical_spikes": spikes,
            "executive_summary": summary_text,
            "composition": get_composition(uid),
        }
        print("OK")

    # Global metrics
    grs = compute_global_risk_score(reports, enrichments)
    oh_values = [r["oh_max"] for r in reports if r.get("oh_max")]
    n_oh_above_1 = sum(1 for v in oh_values if v and v > 1.0)
    n_katashi = sum(1 for r in reports if r.get("regime") == "Katashi")
    max_frozen = max((enrichments[r["universe"]]["regime_duration"]["duration_days"]
                      for r in reports if r.get("regime") in ("Katashi",)), default=0)
    n_alerts = sum(r.get("n_alerts", 0) for r in reports)

    # Determine global trend
    rising = sum(1 for e in enrichments.values() if e["oh_trend"]["direction"] == "rising")
    falling = sum(1 for e in enrichments.values() if e["oh_trend"]["direction"] == "falling")
    if rising > falling + 2:
        global_trend = "deteriorating"
    elif falling > rising + 2:
        global_trend = "improving"
    else:
        global_trend = "mixed"

    cockpit = {
        "timestamp": summary["timestamp"],
        "global_risk_score": grs,
        "global_trend": global_trend,
        "key_metrics": {
            "oh_avg": round(float(np.mean([v for v in oh_values if v])), 3),
            "n_oh_above_1": n_oh_above_1,
            "n_total": len(reports),
            "max_frozen_days": max_frozen,
            "n_active_alerts": n_alerts,
            "n_katashi": n_katashi,
        },
        "universes": enrichments,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(sanitize(cockpit), f, indent=1)
    print(f"\n  Cockpit data saved: {OUT_PATH}")
    print(f"  Global Risk Score: {grs}/100 ({global_trend})")


if __name__ == "__main__":
    print("=" * 60)
    print("  Kappa Sentinel — Cockpit Enrichment")
    print("  David Ohio | odavidohio@gmail.com")
    print("=" * 60)
    generate_cockpit()
