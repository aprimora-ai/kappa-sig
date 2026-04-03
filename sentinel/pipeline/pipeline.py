# -*- coding: utf-8 -*-
"""
Kappa Sentinel — Analysis Pipeline
=====================================
Runs Kappa-FIN engine_v4 on each universe and produces
structural state summary with alert detection.

David Ohio | odavidohio@gmail.com | March 2026
"""
import os, sys, json, warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Fix Windows console encoding for engine_v4 Unicode characters
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, r"C:\Users\ohiod\Projects\kappa-fin")
sys.path.insert(0, r"D:\TopoCML\Kappa-FIN")  # fallback: external HD

from config.universe import ALL_UNIVERSES, PRIORITY
from src.kappa.downloader import load_universe, CACHE_DIR

# Import engine_v4
try:
    from kappa_fin.engine_v4 import ConfigV3, run as kappa_run
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[Pipeline] WARNING: engine_v4 not found. Install from D:\\TopoCML\\Kappa-FIN")


RESULTS_DIR = Path(_ROOT) / "data" / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    universe:    str
    alert_type:  str      # "phi_crossing", "oh_spike", "regime_katashi"
    date:        str
    value:       float
    description: str
    severity:    str = "INFO"   # INFO, WARNING, CRITICAL


@dataclass
class UniverseReport:
    universe:    str
    name:        str
    n_tickers:   int
    date_range:  str
    nu_s:        float = 0.0
    pr:          float = 0.0
    tau_k_max:   int   = 0
    phi_max:     float = 0.0
    oh_max:      float = 0.0
    regime:      str   = "Nagare"
    alerts:      List[Alert] = field(default_factory=list)
    status:      str   = "HEALTHY"    # HEALTHY, MONITORING, PRESSURIZED, CRITICAL
    error:       str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — RUN KAPPA ON A SINGLE UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_universe(universe_id: str, start: str = "2022-01-01",
                     end: str = None) -> UniverseReport:
    """Run Kappa-FIN engine on a single universe and extract summary."""
    universe = ALL_UNIVERSES[universe_id]
    tickers = universe["tickers"]
    k = universe.get("k", 5)
    name = universe["name"]

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    report = UniverseReport(
        universe=universe_id, name=name,
        n_tickers=len(tickers), date_range=f"{start} to {end}",
    )

    if not HAS_ENGINE:
        report.error = "engine_v4 not available"
        return report

    # Create output directory for this universe
    out_dir = str(RESULTS_DIR / f"sentinel_{universe_id}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        cfg = ConfigV3(
            tickers=tickers,
            start=start,
            end=end,
            k=min(k, len(tickers) - 1),
            out=out_dir,
        )

        # Run engine
        state_df = kappa_run(cfg, scenario_title=f"Sentinel: {name}")

        if state_df is None or state_df.empty:
            report.error = "Empty state DataFrame"
            return report

        # Extract summary metrics from state CSV
        report = _extract_metrics(report, state_df, out_dir)

    except Exception as e:
        report.error = str(e)[:300]
        print(f"  [ERROR] {universe_id}: {e}")

    return report


def _extract_metrics(report: UniverseReport, state_df: pd.DataFrame,
                     out_dir: str) -> UniverseReport:
    """Extract summary metrics from Kappa state DataFrame."""

    # Read generated files
    state_path = os.path.join(out_dir, "kappa_v4_state.csv")
    visc_path = os.path.join(out_dir, "kappa_v4_viscosity.csv")

    if os.path.exists(state_path):
        df = pd.read_csv(state_path, index_col=0, parse_dates=True)
    else:
        df = state_df

    # Core metrics
    if "Oh" in df.columns:
        report.oh_max = float(df["Oh"].max())
    if "Phi" in df.columns:
        report.phi_max = float(df["Phi"].max())

    # Regime classification
    if "regime" in df.columns:
        last_regime = df["regime"].iloc[-1] if len(df) > 0 else "Nagare"
        report.regime = last_regime
        # Count Katashi days
        katashi_mask = df["regime"] == "Katashi"
        if katashi_mask.any():
            # Find max consecutive Katashi stretch
            groups = (katashi_mask != katashi_mask.shift()).cumsum()
            katashi_groups = katashi_mask.groupby(groups).sum()
            report.tau_k_max = int(katashi_groups.max())

    # Viscosity from separate file (key-value format)
    if os.path.exists(visc_path):
        try:
            visc = pd.read_csv(visc_path)
            # Format: col0=metric_name, col1=value
            metric_col = visc.columns[0]
            value_col = visc.columns[1]
            metrics = dict(zip(visc[metric_col], visc[value_col]))
            report.nu_s = float(metrics.get("nu_s", 0.0))
            report.pr = float(metrics.get("PR", 0.0))
            report.tau_k_max = int(float(metrics.get("tau_Katashi_max", 0)))
        except Exception as e:
            print(f"  [WARN] viscosity parse: {e}")

    # Phi crossing detection
    if "Phi" in df.columns:
        phi_vals = df["Phi"].values
        # Find where Phi first becomes > 0.001 (non-trivial)
        crossings = np.where(
            (phi_vals[1:] > 0.001) & (phi_vals[:-1] <= 0.001)
        )[0]
        if len(crossings) > 0:
            cross_idx = crossings[0] + 1
            cross_date = str(df.index[cross_idx].date())
            report.alerts.append(Alert(
                universe=report.universe,
                alert_type="phi_crossing",
                date=cross_date,
                value=float(phi_vals[cross_idx]),
                description=f"Phi crossed threshold on {cross_date}",
                severity="WARNING",
            ))

    # Oh spike detection (Oh > 1.0 for 3+ days)
    if "Oh" in df.columns:
        oh_vals = df["Oh"].values
        above = oh_vals > 1.0
        if above.any():
            groups = (above != np.roll(above, 1)).cumsum()
            for g in np.unique(groups[above]):
                g_mask = (groups == g) & above
                if g_mask.sum() >= 3:
                    start_idx = np.where(g_mask)[0][0]
                    spike_date = str(df.index[start_idx].date())
                    spike_val = float(oh_vals[g_mask].max())
                    report.alerts.append(Alert(
                        universe=report.universe,
                        alert_type="oh_spike",
                        date=spike_date,
                        value=spike_val,
                        description=f"Spike de Oh para {spike_val:.3f} por {g_mask.sum()}d a partir de {spike_date}",
                        severity="CRITICAL",
                    ))

    # Determine overall status
    report.status = _classify_status(report)

    return report


def _classify_status(report: UniverseReport) -> str:
    """Classify overall structural status of a universe."""
    if report.error:
        return "ERROR"
    if any(a.severity == "CRITICAL" for a in report.alerts):
        return "CRITICAL"
    if report.tau_k_max > 60 or report.nu_s > 100:
        return "PRESSURIZED"
    if any(a.alert_type == "phi_crossing" for a in report.alerts):
        return "MONITORING"
    return "HEALTHY"


# ══════════════════════════════════════════════════════════════════════════════
# SCAN — RUN ALL UNIVERSES
# ══════════════════════════════════════════════════════════════════════════════

def scan_all(universes: List[str] = None, start: str = "2022-01-01") -> List[UniverseReport]:
    """Run Kappa analysis on all specified universes."""
    if universes is None:
        # Run all non-thematic universes
        universes = []
        for level in ["critical", "high", "medium", "low"]:
            universes.extend(PRIORITY[level])

    print("=" * 70)
    print("  KAPPA SENTINEL — Global Structural Intelligence Scan")
    print(f"  David Ohio | odavidohio@gmail.com | {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)
    print(f"  Universes: {len(universes)}")
    print(f"  Start: {start}")
    print("=" * 70)

    reports = []
    for i, uid in enumerate(universes):
        uname = ALL_UNIVERSES[uid]["name"]
        print(f"\n[{i+1}/{len(universes)}] Analyzing: {uname} ({uid})")
        r = analyze_universe(uid, start=start)
        reports.append(r)

        status_icon = {
            "HEALTHY": "OK", "MONITORING": "(!)",
            "PRESSURIZED": "[!]", "CRITICAL": "[!!!]",
            "ERROR": "ERR",
        }.get(r.status, "?")

        print(f"  => {status_icon} {r.status} | nu_s={r.nu_s:.1f} "
              f"| tau_K={r.tau_k_max}d | Oh_max={r.oh_max:.3f} "
              f"| alerts={len(r.alerts)}")

    # Print summary
    _print_summary(reports)
    _save_summary(reports)

    return reports


def _print_summary(reports: List[UniverseReport]):
    """Print global summary table."""
    print(f"\n{'='*70}")
    print("  GLOBAL STRUCTURAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Universe':<25s} {'Status':<12s} {'nu_s':>8s} {'tau_K':>6s} "
          f"{'Oh_max':>7s} {'Alerts':>6s}")
    print(f"  {'-'*65}")

    for r in sorted(reports, key=lambda x: x.nu_s, reverse=True):
        icon = {"HEALTHY":".", "MONITORING":"~", "PRESSURIZED":"!",
                "CRITICAL":"!!!", "ERROR":"X"}.get(r.status, "?")
        print(f"  {r.name:<25s} {icon:<2s}{r.status:<10s} {r.nu_s:>8.1f} "
              f"{r.tau_k_max:>5d}d {r.oh_max:>7.3f} {len(r.alerts):>6d}")

    # Alerts section
    all_alerts = [a for r in reports for a in r.alerts]
    if all_alerts:
        print(f"\n  ACTIVE ALERTS ({len(all_alerts)}):")
        for a in sorted(all_alerts, key=lambda x: x.severity, reverse=True):
            icon = {"CRITICAL":"[!!!]", "WARNING":"[!]", "INFO":"[i]"}.get(a.severity, "")
            print(f"    {icon} [{a.universe}] {a.description}")


def _save_summary(reports: List[UniverseReport]):
    """Save summary as JSON for dashboard consumption."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_universes": len(reports),
        "reports": [
            {
                "universe": r.universe,
                "name": r.name,
                "status": r.status,
                "nu_s": r.nu_s,
                "tau_k_max": r.tau_k_max,
                "oh_max": r.oh_max,
                "phi_max": r.phi_max,
                "pr": r.pr,
                "regime": r.regime,
                "n_alerts": len(r.alerts),
                "alerts": [
                    {"type": a.alert_type, "date": a.date,
                     "value": a.value, "severity": a.severity,
                     "description": a.description}
                    for a in r.alerts
                ],
                "error": r.error,
            }
            for r in reports
        ],
    }
    out_path = RESULTS_DIR / "sentinel_summary.json"
    with open(out_path, "w") as f:
        # Replace NaN/Infinity with null for JavaScript compatibility
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj
        json.dump(sanitize(summary), f, indent=2)
    print(f"\n  Summary saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kappa Sentinel Pipeline")
    parser.add_argument("--universes", nargs="+", default=None,
                        help="Specific universes to analyze")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--all", action="store_true",
                        help="Include thematic universes")
    args = parser.parse_args()

    universes = args.universes
    if args.all:
        universes = list(ALL_UNIVERSES.keys())

    scan_all(universes=universes, start=args.start)
