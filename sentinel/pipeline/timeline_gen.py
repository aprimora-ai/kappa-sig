"""
Kappa Sentinel — Timeline Generator
Extracts key structural events from state CSVs for dashboard display.

David Ohio | odavidohio@gmail.com | March 2026
"""
import os, json, math
import pandas as pd
import numpy as np
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reports"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "dashboard" / "public" / "sentinel_timelines.json"


def extract_timeline(universe_id: str) -> list:
    """Extract structural events from a universe's state CSV."""
    state_path = REPORTS_DIR / f"sentinel_{universe_id}" / "kappa_v4_state.csv"
    if not state_path.exists():
        return []

    df = pd.read_csv(state_path, parse_dates=["date"])
    if df.empty or len(df) < 10:
        return []

    events = []
    phi_c = df["phi_c"].iloc[0] if "phi_c" in df.columns else 0
    oh_pre = df["Oh_pre"].iloc[0] if "Oh_pre" in df.columns else 1.0

    # 1. Regime transitions
    prev_regime = None
    for _, row in df.iterrows():
        r = row.get("regime", "")
        if prev_regime and r != prev_regime:
            events.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "type": "regime_change",
                "icon": "⬥",
                "title": f"{prev_regime} → {r}",
                "detail": f"Oh={row['Oh']:.3f}, Φ={row['phi']:.4f}",
                "severity": "critical" if r == "Katashi" else "warning" if r == "Utsuroi" else "healthy",
            })
        prev_regime = r

    # 2. Oh spikes above 1.0 (first day of each spike)
    in_spike = False
    for _, row in df.iterrows():
        oh = row.get("Oh", 0)
        if oh > 1.0 and not in_spike:
            events.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "type": "oh_spike",
                "icon": "▲",
                "title": f"Oh spike: {oh:.3f}",
                "detail": f"Ohio Number exceeds critical threshold (Oh > 1.0)",
                "severity": "critical",
            })
            in_spike = True
        elif oh <= 1.0:
            in_spike = False

    # 3. Phi crossings (first time phi > phi_c)
    phi_crossed = False
    for _, row in df.iterrows():
        phi = row.get("phi", 0)
        if phi > phi_c and phi_c > 1e-5 and not phi_crossed:
            events.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "type": "phi_crossing",
                "icon": "◆",
                "title": f"Φ crosses detection threshold",
                "detail": f"Φ={phi:.4f} > Φ_c={phi_c:.4f} — structural memory accumulation confirmed",
                "severity": "warning",
            })
            phi_crossed = True

    # 4. Katashi sustained periods (first day of runs > 10 days)
    katashi_run = 0
    katashi_reported = set()
    for _, row in df.iterrows():
        if row.get("regime") == "Katashi":
            katashi_run += 1
            if katashi_run == 10 and row["date"].strftime("%Y-%m") not in katashi_reported:
                start_date = (row["date"] - pd.Timedelta(days=9)).strftime("%Y-%m-%d")
                katashi_reported.add(row["date"].strftime("%Y-%m"))
                events.append({
                    "date": start_date,
                    "type": "sustained_katashi",
                    "icon": "■",
                    "title": f"Sustained rigidity (10+ days)",
                    "detail": f"Network topology frozen since {start_date}",
                    "severity": "warning",
                })
        else:
            katashi_run = 0

    # 5. Quarterly snapshots (Oh mean, regime fraction)
    df["quarter"] = df["date"].dt.to_period("Q")
    for q, qdf in df.groupby("quarter"):
        oh_mean = qdf["Oh"].mean()
        katashi_frac = (qdf["regime"] == "Katashi").mean()
        regime_dominant = qdf["regime"].mode().iloc[0] if len(qdf) > 0 else "?"
        events.append({
            "date": qdf["date"].iloc[0].strftime("%Y-%m-%d"),
            "type": "quarterly_snapshot",
            "icon": "●",
            "title": f"Q{q.quarter} {q.year}: Oh avg {oh_mean:.3f}",
            "detail": f"Katashi {katashi_frac*100:.0f}% | Dominant: {regime_dominant}",
            "severity": "info",
        })

    # 6. Current state (last row)
    last = df.iloc[-1]
    events.append({
        "date": last["date"].strftime("%Y-%m-%d"),
        "type": "current_state",
        "icon": "◉",
        "title": f"Current: Oh={last['Oh']:.3f}, {last['regime']}",
        "detail": f"Φ={last['phi']:.4f} | η={last['eta']:.2f} | Ξ={last.get('Xi', 0):.2f}",
        "severity": "critical" if last["regime"] == "Katashi" else "warning" if last["regime"] == "Utsuroi" else "info",
    })

    # Sort by date, deduplicate nearby events
    events.sort(key=lambda e: e["date"])
    return events


def extract_sparkline(universe_id: str) -> dict:
    """Extract compact time-series for sparkline charts."""
    state_path = REPORTS_DIR / f"sentinel_{universe_id}" / "kappa_v4_state.csv"
    if not state_path.exists():
        return {}
    df = pd.read_csv(state_path, parse_dates=["date"])
    if df.empty:
        return {}
    # Sample every 5th day for compactness
    sampled = df.iloc[::5]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in sampled["date"]],
        "oh": [round(float(v), 3) for v in sampled["Oh"]],
        "phi": [round(float(v), 4) for v in sampled["phi"]],
        "regime": [str(v) for v in sampled["regime"]],
    }


def generate_all():
    """Generate timelines for all universes."""
    summary_path = REPORTS_DIR / "sentinel_summary.json"
    with open(summary_path) as f:
        raw = f.read().replace("NaN", "null")
        summary = json.loads(raw)

    result = {}
    for report in summary["reports"]:
        uid = report["universe"]
        if report.get("error"):
            continue
        print(f"  {uid}...", end=" ", flush=True)
        events = extract_timeline(uid)
        sparkline = extract_sparkline(uid)
        result[uid] = {
            "events": events,
            "sparkline": sparkline,
        }
        print(f"{len(events)} events")

    # Sanitize NaN/Infinity
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(sanitize(result), f, indent=1)
    print(f"\n  Timelines saved: {OUT_PATH}")


if __name__ == "__main__":
    print("Generating Sentinel timelines...")
    generate_all()
