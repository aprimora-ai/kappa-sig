#!/usr/bin/env python3
"""
Kappa Sentinel — Pipeline History Archiver
==========================================
Archives current pipeline data into dated snapshots for the dashboard.

Run after each pipeline execution:
    python archive_pipeline.py

Creates:
    dashboard/public/history/YYYY-MM-DD/sentinel_summary.json
    dashboard/public/history/YYYY-MM-DD/sentinel_cockpit.json
    dashboard/public/history/YYYY-MM-DD/sentinel_timelines.json
    dashboard/public/history/YYYY-MM-DD/analyst_briefings.json
    dashboard/public/history/history_index.json  (updated)

David Ohio | odavidohio@gmail.com | March 2026
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).parent
DASHBOARD_PUBLIC = ROOT / "dashboard" / "public"
HISTORY_DIR = DASHBOARD_PUBLIC / "history"
DATA_REPORTS = ROOT / "data" / "reports"

FILES_TO_ARCHIVE = [
    "sentinel_summary.json",
    "sentinel_cockpit.json",
    "sentinel_timelines.json",
    "analyst_briefings.json",
]


def get_pipeline_date() -> str:
    """Extract date from sentinel_summary.json timestamp."""
    summary_path = DASHBOARD_PUBLIC / "sentinel_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", "")
        if ts:
            return ts[:10]  # YYYY-MM-DD
    return datetime.now().strftime("%Y-%m-%d")



def archive_snapshot(date_str: str | None = None):
    """Archive current pipeline data into a dated snapshot."""
    if date_str is None:
        date_str = get_pipeline_date()

    snapshot_dir = HISTORY_DIR / date_str
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in FILES_TO_ARCHIVE:
        src = DASHBOARD_PUBLIC / fname
        if src.exists():
            dst = snapshot_dir / fname
            shutil.copy2(src, dst)
            copied.append(fname)
            print(f"  [OK] {fname} -> history/{date_str}/")
        else:
            print(f"  [SKIP] {fname} not found in dashboard/public/")

    # Update history index
    update_index()

    print(f"\n[OK] Archived {len(copied)} files for {date_str}")
    return date_str, copied


def update_index():
    """Rebuild history_index.json from available dated folders."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for d in sorted(HISTORY_DIR.iterdir()):
        if d.is_dir() and len(d.name) == 10 and d.name[4] == '-':
            summary_path = d / "sentinel_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_universes = data.get("n_universes", 0)
                timestamp = data.get("timestamp", "")
                reports = data.get("reports", [])
                n_critical = sum(1 for r in reports if r.get("status") == "CRITICAL")
                n_pressurized = sum(1 for r in reports if r.get("status") == "PRESSURIZED")
                n_healthy = sum(1 for r in reports if r.get("status") == "HEALTHY")

                entries.append({
                    "date": d.name,
                    "timestamp": timestamp,
                    "n_universes": n_universes,
                    "n_critical": n_critical,
                    "n_pressurized": n_pressurized,
                    "n_healthy": n_healthy,
                })

    index = {
        "updated_at": datetime.now().isoformat(),
        "n_snapshots": len(entries),
        "snapshots": entries,
    }

    index_path = HISTORY_DIR / "history_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  [OK] history_index.json updated ({len(entries)} snapshots)")
    return entries


if __name__ == "__main__":
    print("=" * 50)
    print("  KAPPA SENTINEL — Pipeline Archiver")
    print("=" * 50)
    date_str, copied = archive_snapshot()
    print(f"\nDone. Dashboard can now view history/{date_str}/")
