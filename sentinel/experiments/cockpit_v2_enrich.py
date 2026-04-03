r"""
Kappa Sentinel -- v2 Cockpit Enrichment
Merges v2 analysis results (Layers 1-3) into the dashboard cockpit JSON.

Reads:
  - data/v2_analysis/{universe}/kappa_v2_summary.json (per universe)
  - data/v2_monitoring/prospective_status.json (prospective cases)

Writes:
  - dashboard/public/sentinel_v2_cockpit.json (new file for v2 panel)
  - Updates dashboard/public/sentinel_cockpit.json (adds v2 fields to existing)

Run after run_v2_monitor.py in the pipeline.

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import json, math, os, sys
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "data" / "v2_analysis"
MONITOR_DIR = ROOT / "data" / "v2_monitoring"
COCKPIT_PATH = ROOT / "dashboard" / "public" / "sentinel_cockpit.json"
V2_COCKPIT_PATH = ROOT / "dashboard" / "public" / "sentinel_v2_cockpit.json"


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


def load_v2_summaries():
    """Load all v2 summary JSONs."""
    summaries = {}
    if not V2_DIR.exists():
        return summaries
    for d in sorted(V2_DIR.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / "kappa_v2_summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summaries[d.name] = json.load(f)
    return summaries


def load_prospective_status():
    """Load prospective monitoring status."""
    path = MONITOR_DIR / "prospective_status.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_v2_global_metrics(summaries):
    """Compute aggregate v2 metrics across all universes."""
    real = {k: v for k, v in summaries.items() if not v.get("phi_star_estimated")}

    alerts = [v.get("alert_level", "NOMINAL") for v in real.values()]
    n_emergency = alerts.count("EMERGENCY")
    n_warning = alerts.count("WARNING")
    n_watch = alerts.count("WATCH")
    n_nominal = alerts.count("NOMINAL")

    # Geometric activation stats
    geo_active = 0
    geo_crystallized = 0
    for v in real.values():
        ga = v.get("geo_activation", {})
        if ga.get("currently_active"):
            geo_active += 1
        if ga.get("crystallization_duration", 0) > 90:
            geo_crystallized += 1

    # Capacity stats
    c_norms = [v.get("C_norm_now", 1.0) for v in real.values()]
    c_below_50 = sum(1 for c in c_norms if c < 0.50)
    c_below_0 = sum(1 for c in c_norms if c < 0.0)

    return {
        "n_total": len(real),
        "n_emergency": n_emergency,
        "n_warning": n_warning,
        "n_watch": n_watch,
        "n_nominal": n_nominal,
        "n_geo_active": geo_active,
        "n_geo_crystallized_90d": geo_crystallized,
        "n_capacity_below_50": c_below_50,
        "n_capacity_exhausted": c_below_0,
        "avg_C_norm": float(np.mean(c_norms)) if c_norms else 1.0,
    }


def build_universe_v2(uid, s):
    """Build per-universe v2 data for cockpit."""
    ga = s.get("geo_activation", {})

    return {
        # Layer 1
        "C_norm": s.get("C_norm_now"),
        "kappa_F": s.get("kappa_F_now"),
        "rho_bar": s.get("rho_bar_now"),
        "T_exhaust": s.get("T_exhaust_now"),
        "n_false_recovery": s.get("n_false_recovery_days", 0),
        "phi_star_status": s.get("phi_star_status", "unknown"),
        # Layer 2
        "theta_A": s.get("theta_A_now"),
        "geo_reliability": s.get("geo_reliability"),
        "activation_type": ga.get("activation_type", "NONE"),
        "t_G": ga.get("t_G"),
        "t_G_date": ga.get("t_G_date"),
        "crystallization_duration": ga.get("crystallization_duration", 0),
        "currently_active": ga.get("currently_active", False),
        # Layer 3
        "h": s.get("h_now"),
        "P30": s.get("P_collapse_30d"),
        "P90": s.get("P_collapse_90d"),
        "T_half": s.get("T_half_now"),
        "alert_level": s.get("alert_level"),
    }


def main():
    print("[v2-cockpit] Loading v2 analysis results...")
    summaries = load_v2_summaries()
    prospective = load_prospective_status()

    if not summaries:
        print("[v2-cockpit] No v2 summaries found. Run run_v2_monitor.py first.")
        return

    print(f"[v2-cockpit] Loaded {len(summaries)} universe summaries")

    # Build v2 cockpit
    global_metrics = compute_v2_global_metrics(summaries)

    universes_v2 = {}
    for uid, s in summaries.items():
        universes_v2[uid] = build_universe_v2(uid, s)

    v2_cockpit = {
        "timestamp": datetime.now().isoformat(),
        "engine_version": "v5.5",
        "global_v2": global_metrics,
        "prospective_cases": sanitize(prospective),
        "universes": sanitize(universes_v2),
    }

    # Write v2 cockpit JSON
    with open(V2_COCKPIT_PATH, "w", encoding="utf-8") as f:
        json.dump(v2_cockpit, f, indent=2, default=str)
    print(f"[v2-cockpit] Written: {V2_COCKPIT_PATH}")

    # Also merge v2 fields into existing cockpit (if it exists)
    if COCKPIT_PATH.exists():
        with open(COCKPIT_PATH, encoding="utf-8") as f:
            cockpit = json.load(f)

        cockpit["v2_global"] = sanitize(global_metrics)
        cockpit["v2_prospective"] = sanitize(prospective)

        # Add v2 fields to each universe in existing cockpit
        for uid, v2data in universes_v2.items():
            if uid in cockpit.get("universes", {}):
                cockpit["universes"][uid]["v2"] = sanitize(v2data)

        with open(COCKPIT_PATH, "w", encoding="utf-8") as f:
            json.dump(cockpit, f, indent=2, default=str)
        print(f"[v2-cockpit] Updated: {COCKPIT_PATH} (v2 fields merged)")

    print(f"[v2-cockpit] Done. {global_metrics['n_emergency']} EMERGENCY, "
          f"{global_metrics['n_warning']} WARNING, "
          f"{global_metrics['n_geo_active']} geo-active")


if __name__ == "__main__":
    main()
