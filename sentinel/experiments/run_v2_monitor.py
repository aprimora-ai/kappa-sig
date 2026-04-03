#!/usr/bin/env python3
r"""
Kappa v2 -- Prospective Monitoring System
==========================================
Tracks the 3 prospective test cases and all Sentinel universes daily.
Records longitudinal data for hypothesis validation.

Run daily after Sentinel pipeline completes:
  cd C:\Users\ohiod\Projects\Sentinel
  python run_v2_monitor.py

Outputs:
  - v2_analysis/ per-universe results
  - v2_monitoring/tracking.csv -- longitudinal record
  - v2_monitoring/alerts.log -- state transition alerts
  - v2_monitoring/prospective_status.json -- current status of test cases

David Ohio | odavidohio@gmail.com | Independent Researcher
March 2026
"""
import os, sys, json, csv
from datetime import datetime, date
from pathlib import Path
import io, contextlib

sys.path.insert(0, r"C:\Users\ohiod\Projects\kappa-fin")
from kappa_fin.engine_v5_fixed import run_v2, V2Config

REPORTS_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\reports")
V2_OUTPUT_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_analysis")
MONITOR_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel\data\v2_monitoring")
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_CSV = MONITOR_DIR / "tracking.csv"
ALERTS_LOG = MONITOR_DIR / "alerts.log"
PROSPECTIVE_JSON = MONITOR_DIR / "prospective_status.json"

# Prospective test cases with known t_G (audit-derived)
PROSPECTIVE_CASES = {
    "asia_pacific": {
        "t_G": "2025-11-25",
        "t_G_type": "ramp",
        "current_episode": 2,
        "notes": "Episode 2 on recovered system. Ramp Nov-Dec 2025.",
    },
    "x_brazil_vuln": {
        "t_G": "2023-01-17",
        "t_G_type": "ramp",
        "current_episode": 2,
        "notes": "Episode 2. Iran War transmission via commodities/BRL.",
    },
    "x_us_systemic": {
        "t_G": "2022-05-02",
        "t_G_type": "ramp",
        "current_episode": 2,
        "notes": "Episode 2. Longest crystallization (26+ months).",
    },
}

T_S_THRESHOLD = 0.90
T_W_THRESHOLD = 0.50
T_C_THRESHOLD = 0.00

# Phase labels (ASCII-safe for Windows console)
PHASE_LABELS = {
    "GEOMETRIC_ACTIVE_NO_DAMAGE": "[GEO]",
    "QUIESCENT":                  "[---]",
    "STRUCTURAL_ACTIVATION_MILD": "[S-1]",
    "STRUCTURAL_ACTIVATION_SEVERE":"[S-2]",
    "POST_IRREVERSIBILITY":       "[IRR]",
}


def find_universes():
    universes = []
    for d in sorted(REPORTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("sentinel_"):
            sc = d / "kappa_v4_state.csv"
            vc = d / "kappa_v4_viscosity.csv"
            if sc.exists() and vc.exists():
                universes.append({
                    "name": d.name.replace("sentinel_", ""),
                    "state_csv": str(sc),
                    "visc_csv": str(vc)
                })
    return universes


def log_alert(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"  [!] {msg}")


def init_tracking_csv():
    if not TRACKING_CSV.exists():
        with open(TRACKING_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_date", "universe", "data_date",
                "C_norm", "kappa_F", "theta_A",
                "h", "P30", "T_half", "alert_level",
                "geo_reliability", "phi_star_status",
                "rho_bar", "activation_type", "crystallization_dur",
                "is_prospective", "t_G", "days_since_tG",
                "t_S_detected", "dT_GS_days"
            ])


def append_tracking(run_date, uid, summary, data_date):
    is_prosp = uid in PROSPECTIVE_CASES
    t_G = PROSPECTIVE_CASES[uid]["t_G"] if is_prosp else ""
    days_since = ""
    t_S_detected = ""
    dT_GS = ""

    if is_prosp and t_G:
        tg_date = datetime.strptime(t_G, "%Y-%m-%d").date()
        dd = datetime.strptime(data_date, "%Y-%m-%d").date() if isinstance(data_date, str) else data_date
        days_since = (dd - tg_date).days

        cn = summary.get("C_norm_now", 1.0)
        if cn < T_S_THRESHOLD and not summary.get("phi_star_estimated", False):
            t_S_detected = data_date
            dT_GS = days_since

    # Read geo_activation from v5.5 summary
    ga = summary.get("geo_activation", {})
    act_type = ga.get("activation_type", "")
    crystal_dur = ga.get("crystallization_duration", 0)

    with open(TRACKING_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_date, uid, data_date,
            f"{summary.get('C_norm_now', 0):.6f}",
            f"{summary.get('kappa_F_now', 0):.4f}",
            f"{summary.get('theta_A_now', 0):.4f}",
            f"{summary.get('h_now', 0):.6f}",
            f"{summary.get('P_collapse_30d', 0):.4f}",
            f"{summary.get('T_half_now', 0):.1f}",
            summary.get("alert_level", ""),
            summary.get("geo_reliability", ""),
            summary.get("phi_star_status", ""),
            f"{summary.get('rho_bar_now', 0):.6f}",
            act_type, crystal_dur,
            is_prosp, t_G, days_since,
            t_S_detected, dT_GS
        ])


def check_state_transitions(uid, summary, previous_status):
    current = summary.get("alert_level", "NOMINAL")
    previous = previous_status.get(uid, {}).get("alert_level", "NOMINAL")

    if current != previous and previous != "NOMINAL":
        log_alert(f"{uid}: Alert changed {previous} -> {current}")
    elif current != previous:
        log_alert(f"{uid}: New alert level -> {current}")

    if uid in PROSPECTIVE_CASES:
        cn = summary.get("C_norm_now", 1.0)
        prev_cn = previous_status.get(uid, {}).get("C_norm_now", 1.0)

        if cn < T_S_THRESHOLD and prev_cn >= T_S_THRESHOLD:
            t_G = PROSPECTIVE_CASES[uid]["t_G"]
            log_alert(f"*** {uid}: t_S DETECTED! C/Phi* crossed below {T_S_THRESHOLD}. "
                      f"t_G was {t_G}. THIS IS A LEAD-TIME MEASUREMENT. ***")

        ta = summary.get("theta_A_now", 0)
        prev_ta = previous_status.get(uid, {}).get("theta_A_now", 0)
        if abs(ta - prev_ta) > 0.5:
            log_alert(f"{uid}: Theta_A shifted: {prev_ta:.2f} -> {ta:.2f}")

        if prev_ta > 2.0 and ta < 0.5:
            log_alert(f"*** {uid}: GEOMETRIC DE-CRYSTALLIZATION! "
                      f"Theta_A {prev_ta:.2f} -> {ta:.2f}. Claim C may weaken. ***")


def update_prospective_status(all_summaries):
    status = {}
    for uid, info in PROSPECTIVE_CASES.items():
        s = all_summaries.get(uid, {})
        cn = s.get("C_norm_now", 1.0)
        ta = s.get("theta_A_now", 0)
        ga = s.get("geo_activation", {})

        if cn >= T_S_THRESHOLD:
            phase = "GEOMETRIC_ACTIVE_NO_DAMAGE" if ta > 0.5 else "QUIESCENT"
        elif cn >= T_W_THRESHOLD:
            phase = "STRUCTURAL_ACTIVATION_MILD"
        elif cn >= T_C_THRESHOLD:
            phase = "STRUCTURAL_ACTIVATION_SEVERE"
        else:
            phase = "POST_IRREVERSIBILITY"

        tg_date = datetime.strptime(info["t_G"], "%Y-%m-%d").date()
        days_crystal = (date.today() - tg_date).days

        status[uid] = {
            **info,
            "phase": phase,
            "C_norm_now": cn,
            "theta_A_now": ta,
            "alert_level": s.get("alert_level", "?"),
            "h_now": s.get("h_now", 0),
            "P30": s.get("P_collapse_30d", 0),
            "T_half": s.get("T_half_now", 0),
            "days_crystallized": days_crystal,
            "t_S_detected": cn < T_S_THRESHOLD,
            "activation_type": ga.get("activation_type", "NONE"),
            "engine_crystal_dur": ga.get("crystallization_duration", 0),
            "last_update": str(date.today()),
        }

    with open(PROSPECTIVE_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    return status


def run_monitoring():
    universes = find_universes()
    run_date = datetime.now().strftime("%Y-%m-%d")
    cfg = V2Config()

    print("")
    print("=" * 70)
    print(f"  KAPPA v2 MONITORING -- {run_date}")
    print(f"  {len(universes)} universes | engine v5.5")
    print("=" * 70)

    init_tracking_csv()

    if PROSPECTIVE_JSON.exists():
        with open(PROSPECTIVE_JSON, encoding="utf-8") as f:
            previous_status = json.load(f)
    else:
        previous_status = {}

    all_summaries = {}

    for u in universes:
        nm = u["name"]
        od = V2_OUTPUT_DIR / nm
        od.mkdir(parents=True, exist_ok=True)

        try:
            f_out = io.StringIO()
            with contextlib.redirect_stdout(f_out):
                df, s = run_v2(u["state_csv"], u["visc_csv"], cfg)

            df.to_csv(od / "kappa_v2_state.csv")
            with open(od / "kappa_v2_summary.json", "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2, default=str)

            all_summaries[nm] = s
            data_date = str(df.index[-1].date())
            append_tracking(run_date, nm, s, data_date)
            check_state_transitions(nm, s, previous_status)

        except Exception as e:
            print(f"  ERROR {nm}: {e}")

    prosp_status = update_prospective_status(all_summaries)

    # -- PROSPECTIVE REPORT --
    print("")
    print("=" * 70)
    print("  PROSPECTIVE CASE STATUS")
    print("=" * 70)

    for uid, ps in prosp_status.items():
        label = PHASE_LABELS.get(ps["phase"], "[?]")
        act = ps.get("activation_type", "?")

        print(f"\n  {label} {uid}")
        print(f"     Phase: {ps['phase']}")
        print(f"     C/Phi* = {ps['C_norm_now']:.6f}  Theta_A = {ps['theta_A_now']:.3f}")
        print(f"     t_G = {ps['t_G']}  ({ps['days_crystallized']} days ago)")
        print(f"     Activation: {act}  Engine dur: {ps.get('engine_crystal_dur', '?')} steps")
        print(f"     P30 = {ps['P30']:.3f}  T1/2 = {ps['T_half']:.1f}  [{ps['alert_level']}]")
        if ps["t_S_detected"]:
            print(f"     *** t_S DETECTED! Lead time measurement available ***")

    # -- CROSS-UNIVERSE SUMMARY --
    real = {k: v for k, v in all_summaries.items() if not v.get("phi_star_estimated")}
    n_em = sum(1 for v in real.values() if v.get("alert_level") == "EMERGENCY")
    n_wr = sum(1 for v in real.values() if v.get("alert_level") == "WARNING")
    n_wt = sum(1 for v in real.values() if v.get("alert_level") == "WATCH")
    n_nm = sum(1 for v in real.values() if v.get("alert_level") == "NOMINAL")

    print(f"\n  Summary: {n_em} EMERGENCY  {n_wr} WARNING  {n_wt} WATCH  {n_nm} NOMINAL")
    print(f"  Tracking: {TRACKING_CSV}")
    print(f"  Alerts: {ALERTS_LOG}")
    print(f"  Prospective: {PROSPECTIVE_JSON}")
    print("\nDone.")


if __name__ == "__main__":
    run_monitoring()
