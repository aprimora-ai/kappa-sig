#!/usr/bin/env python3
"""
Kappa-SIG MET: Obsessive Coherence in Atmospheric Dynamics
============================================================
Applies the Kappa framework to temperature correlation networks.
When the atmosphere enters "blocking" patterns, distant stations
become excessively correlated — the same obsessive coherence that
precedes financial crises and neural seizures.

Network:
  Nodes = 30 US weather stations
  Correlation = Pearson on daily mean temperature (rolling 30-day window)
  Events = Known extreme weather (2021 Texas freeze, 2023 heat waves)

Data source: Open-Meteo Archive API (free, no auth)

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import json, time, urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as sp_entropy

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "results"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

WINDOW = 30   # days rolling correlation
# 2 years spanning known extreme events
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# 30 US stations across climate regions (lat, lon, name)
STATIONS = [
    (42.36, -71.06, "Boston"), (40.71, -74.01, "New_York"),
    (38.91, -77.04, "Washington"), (33.75, -84.39, "Atlanta"),
    (25.76, -80.19, "Miami"), (29.76, -95.37, "Houston"),
    (30.27, -97.74, "Austin"), (32.78, -96.80, "Dallas"),
    (35.47, -97.52, "Oklahoma_City"), (38.63, -90.20, "St_Louis"),
    (41.88, -87.63, "Chicago"), (44.98, -93.27, "Minneapolis"),
    (39.10, -94.58, "Kansas_City"), (39.74, -104.99, "Denver"),
    (40.76, -111.89, "Salt_Lake"), (33.45, -112.07, "Phoenix"),
    (36.17, -115.14, "Las_Vegas"), (34.05, -118.24, "Los_Angeles"),
    (37.77, -122.42, "San_Francisco"), (47.61, -122.33, "Seattle"),
    (45.52, -122.68, "Portland"), (43.07, -89.40, "Madison"),
    (42.33, -83.05, "Detroit"), (39.96, -82.99, "Columbus"),
    (35.23, -80.84, "Charlotte"), (36.16, -86.78, "Nashville"),
    (30.33, -81.66, "Jacksonville"), (27.95, -82.46, "Tampa"),
    (41.26, -95.94, "Omaha"), (46.88, -96.79, "Fargo"),
]

# Known extreme events (for before/during/after comparison)
EVENTS = {
    "winter_storm_elliott": {
        "label": "Winter Storm Elliott (Dec 2022)",
        "pre": ("2022-11-15", "2022-12-15"),
        "during": ("2022-12-20", "2022-12-28"),
    },
    "heat_dome_2023": {
        "label": "July 2023 Heat Dome (Southwest US)",
        "pre": ("2023-06-01", "2023-06-30"),
        "during": ("2023-07-10", "2023-07-31"),
    },
    "arctic_blast_jan2024": {
        "label": "Arctic Blast Jan 2023",
        "pre": ("2022-12-01", "2022-12-31"),
        "during": ("2023-01-15", "2023-02-05"),
    },
}


def download_temperatures():
    """Download daily mean temperature from Open-Meteo for all stations."""
    cache = DATA_DIR / "temperatures.csv"
    if cache.exists() and cache.stat().st_size > 1000:
        print("    Using cached temperature data")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    
    print(f"    Downloading {len(STATIONS)} stations from Open-Meteo...")
    all_series = {}
    
    for lat, lon, name in STATIONS:
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}"
               f"&start_date={START_DATE}&end_date={END_DATE}"
               f"&daily=temperature_2m_mean&timezone=America/New_York")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KappaSIG/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_mean"]
            all_series[name] = pd.Series(temps, index=pd.to_datetime(dates), name=name)
            print(f"      {name}: {len(temps)} days OK")
        except Exception as e:
            print(f"      {name}: FAILED ({e})")
    
    df = pd.DataFrame(all_series)
    df = df.interpolate(method="linear", limit=3).dropna()
    df.to_csv(cache)
    print(f"    Saved: {cache} ({len(df)} days x {len(df.columns)} stations)")
    return df


def kappa_from_window(window_df):
    """Compute Kappa state from rolling window of station temperatures."""
    C = window_df.corr(method="pearson").values
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    n = C.shape[0]
    
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    total = eigvals.sum()
    
    Oh = float(eigvals[0] / (total / n))
    eig_norm = eigvals / total
    eig_pos = eig_norm[eig_norm > 1e-12]
    H = sp_entropy(eig_pos, base=2)
    H_max = np.log2(len(eig_pos)) if len(eig_pos) > 1 else 1.0
    eta = float(1.0 - H / H_max) if H_max > 0 else 0.0
    mask = ~np.eye(n, dtype=bool)
    mean_corr = float(np.mean(np.abs(C[mask])))
    DEF = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10))
    eff_rank = np.exp(sp_entropy(eig_pos))
    Xi = float(eff_rank / n)
    
    return {"Oh": Oh, "eta": eta, "mean_corr": mean_corr,
            "DEF": DEF, "Xi": Xi}


def rolling_kappa(df, window=WINDOW):
    """Compute rolling Kappa states over the full time series."""
    states = []
    dates = df.index
    for i in range(window, len(df)):
        w = df.iloc[i-window:i]
        s = kappa_from_window(w)
        s["date"] = dates[i].strftime("%Y-%m-%d")
        states.append(s)
    return pd.DataFrame(states)


def analyze_event(kappa_df, event_name, event_cfg):
    """Compare Kappa state before vs during an extreme event."""
    pre_start, pre_end = event_cfg["pre"]
    dur_start, dur_end = event_cfg["during"]
    
    pre = kappa_df[(kappa_df["date"] >= pre_start) & (kappa_df["date"] <= pre_end)]
    dur = kappa_df[(kappa_df["date"] >= dur_start) & (kappa_df["date"] <= dur_end)]
    
    if len(pre) < 5 or len(dur) < 5:
        return None
    
    metrics = ["Oh", "eta", "mean_corr", "DEF", "Xi"]
    pre_mean = {m: float(pre[m].mean()) for m in metrics}
    dur_mean = {m: float(dur[m].mean()) for m in metrics}

    dirs = {"Oh": dur_mean["Oh"]>pre_mean["Oh"],
            "eta": dur_mean["eta"]>pre_mean["eta"],
            "mean_corr": dur_mean["mean_corr"]>pre_mean["mean_corr"],
            "DEF": dur_mean["DEF"]>pre_mean["DEF"],
            "Xi": dur_mean["Xi"]<pre_mean["Xi"]}
    n_coh = sum(dirs.values())
    
    def oc(r):
        return (r["Oh"]/30 + r["DEF"])/2.0 - (r["Xi"]+(1.0-r["mean_corr"]))/2.0
    oc_pre = oc(pre_mean)
    oc_dur = oc(dur_mean)
    
    return {"event": event_name, "label": event_cfg["label"],
            "pre_mean": {k: round(v, 4) for k,v in pre_mean.items()},
            "during_mean": {k: round(v, 4) for k,v in dur_mean.items()},
            "directions": {k: bool(v) for k,v in dirs.items()},
            "n_coherent": n_coh, "confirmed": n_coh >= 4,
            "oc_pre": round(oc_pre, 4), "oc_during": round(oc_dur, 4),
            "oc_delta": round(oc_dur - oc_pre, 4),
            "n_pre_days": len(pre), "n_dur_days": len(dur)}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-SIG MET: Obsessive Coherence in Atmospheric Dynamics")
    print("  David Ohio | Independent Researcher | April 2026")
    print("  30 US stations, daily temp, Pearson rolling 30-day window")
    print("=" * 70)
    
    # Download
    df = download_temperatures()
    print(f"  Data: {len(df)} days x {len(df.columns)} stations")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Rolling Kappa
    print("\n  Computing rolling Kappa states...")
    kappa_df = rolling_kappa(df)
    print(f"  {len(kappa_df)} Kappa states computed")
    
    # Global baseline
    metrics = ["Oh", "eta", "mean_corr", "DEF", "Xi"]
    baseline = {m: round(float(kappa_df[m].mean()), 4) for m in metrics}
    print(f"\n  BASELINE (full period mean):")
    print(f"    Oh={baseline['Oh']:.4f}  eta={baseline['eta']:.4f}  "
          f"mean_corr={baseline['mean_corr']:.4f}  DEF={baseline['DEF']:.4f}  "
          f"Xi={baseline['Xi']:.4f}")

    # Per-event analysis
    event_results = {}
    print(f"\n  EXTREME EVENTS:")
    print(f"  {'='*65}")
    for ev_name, ev_cfg in EVENTS.items():
        r = analyze_event(kappa_df, ev_name, ev_cfg)
        if r is None:
            print(f"\n  [{ev_name}] Insufficient data"); continue
        event_results[ev_name] = r
        pre, dur = r["pre_mean"], r["during_mean"]
        print(f"\n  [{r['label']}]")
        print(f"    {'':12s} {'Pre':>10s} {'During':>10s} {'Coh':>5s}")
        for m in metrics:
            c = "+" if r["directions"][m] else "-"
            print(f"    {m:12s} {pre[m]:10.4f} {dur[m]:10.4f} {c:>5s}")
        st = "CONFIRMED" if r["confirmed"] else "PARTIAL"
        print(f"    Dirs: {r['n_coherent']}/5 -- {st}  "
              f"OC: pre={r['oc_pre']:+.4f} dur={r['oc_during']:+.4f} "
              f"delta={r['oc_delta']:+.4f}")

    # Overall: normal periods vs extreme event periods
    all_event_dates = set()
    for ev in EVENTS.values():
        d1, d2 = ev["during"]
        mask = (kappa_df["date"] >= d1) & (kappa_df["date"] <= d2)
        all_event_dates.update(kappa_df[mask]["date"].tolist())
    
    normal = kappa_df[~kappa_df["date"].isin(all_event_dates)]
    extreme = kappa_df[kappa_df["date"].isin(all_event_dates)]
    
    if len(extreme) > 0:
        norm_mean = {m: round(float(normal[m].mean()), 4) for m in metrics}
        ext_mean = {m: round(float(extreme[m].mean()), 4) for m in metrics}
        print(f"\n  {'='*65}")
        print(f"  AGGREGATE: Normal ({len(normal)} days) vs Extreme ({len(extreme)} days)")
        print(f"    {'':12s} {'Normal':>10s} {'Extreme':>10s} {'Delta':>10s}")
        for m in metrics:
            d = ext_mean[m] - norm_mean[m]
            print(f"    {m:12s} {norm_mean[m]:10.4f} {ext_mean[m]:10.4f} {d:+10.4f}")

    # Summary
    n_confirmed = sum(1 for r in event_results.values() if r["confirmed"])
    print(f"\n  {'='*65}")
    print(f"  SUMMARY: {n_confirmed}/{len(event_results)} events confirm obsessive coherence")
    
    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "MET", "substrate": "Atmospheric dynamics",
        "method": "Pearson correlation between 30 US station daily temperatures, "
                  "rolling 30-day window, eigenstructure -> Kappa state. "
                  "Same pipeline as FIN (asset correlations).",
        "data_source": "Open-Meteo Archive API (free, no auth)",
        "stations": len(STATIONS), "period": f"{START_DATE} to {END_DATE}",
        "window": WINDOW,
        "baseline": baseline,
        "events": event_results,
        "n_confirmed": n_confirmed,
    }
    out_file = OUT_DIR / "met_obsessive_coherence.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save Kappa time series
    kappa_df.to_csv(OUT_DIR / "met_kappa_timeseries.csv", index=False)
    
    print(f"\n  Results: {out_file}")
    print(f"  Time series: {OUT_DIR / 'met_kappa_timeseries.csv'}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
