#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Kappa-SIG Phase 1: Extract Raw Structural Data
================================================
Recomputes rolling correlation matrices for all Sentinel universes
and extracts eigenvalues, PH barcodes, LZ complexity, and ground truth.

Does NOT modify any production code.

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# Paths
SENTINEL_DIR = Path(r"C:\Users\ohiod\Projects\Sentinel")
KAPPA_DIR = Path(r"C:\Users\ohiod\Projects\kappa-fin")
V2_DIR = SENTINEL_DIR / "data" / "v2_analysis"
SIG_DIR = SENTINEL_DIR / "data" / "sig"
SIG_DIR.mkdir(parents=True, exist_ok=True)

# Config
START_DATE = "2022-01-01"
END_DATE   = "2026-04-01"
WINDOW     = 22          # Rolling correlation window (same as engine_v4)
SHRINK_LAM = 0.05        # Ledoit-Wolf shrinkage (same as engine_v4)
EXCLUDE    = {"latam", "x_commodity_chain"}  # Baseline insufficient

# Universe loader
sys.path.insert(0, str(SENTINEL_DIR))
from config.universe import ALL_UNIVERSES

# Optional: ripser for PH barcodes
try:
    from ripser import ripser as ripser_fn
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("[WARN] ripser not installed. PH barcodes will be skipped.")
    print("       Install: pip install ripser")

# yfinance
try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance required: pip install yfinance")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (mirror engine_v4 exactly)
# ══════════════════════════════════════════════════════════════════

def corr_spearman(R):
    """Spearman correlation via ranks (same as engine_v4)."""
    ranks = np.apply_along_axis(lambda x: pd.Series(x).rank().to_numpy(), 0, R)
    C = np.corrcoef(ranks, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def shrink_corr(C, lam=0.05):
    """Ledoit-Wolf shrinkage: C_tilde = (1-lam)*C + lam*I"""
    return (1.0 - lam) * C + lam * np.eye(C.shape[0])

def angular_distance(C):
    """d_ij = sqrt(2*(1-C_ij)) — angular distance (same as engine_v4)."""
    D = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(D, 0.0)
    return D


# ══════════════════════════════════════════════════════════════════
# S1: SPECTRAL ANALYSIS
# ══════════════════════════════════════════════════════════════════

def compute_spectral(C):
    """Compute spectral metrics from correlation matrix."""
    eigenvalues = np.sort(np.linalg.eigvalsh(C))[::-1]  # Descending
    N = len(eigenvalues)
    lam1 = eigenvalues[0]
    lam2 = eigenvalues[1] if N > 1 else 0.0
    # SCR: Spectral Concentration Ratio
    scr = lam1 / max(np.sum(eigenvalues), 1e-12)
    # SGR: Spectral Gap Ratio
    sgr = (lam1 - lam2) / max(lam1, 1e-12)
    return eigenvalues, scr, sgr

def mp_pdf(lam, q, sigma2=1.0, eps=1e-12):
    """Marchenko-Pastur density for q = N/T."""
    lam_plus = sigma2 * (1.0 + np.sqrt(q))**2
    lam_minus = sigma2 * (1.0 - np.sqrt(q))**2
    out = np.zeros_like(lam, dtype=float)
    mask = (lam >= lam_minus) & (lam <= lam_plus)
    x = lam[mask]
    out[mask] = np.sqrt((lam_plus - x) * (x - lam_minus)) / (
        2.0 * np.pi * q * sigma2 * np.maximum(x, eps))
    return out

def compute_mp_metrics(eigenvalues, T, N, n_grid=256, bandwidth=0.03, eps=1e-12):
    """True MP distance (KL divergence) + edge diagnostics."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]
    if len(eigenvalues) < 3:
        return {"d_mp_kl": 0.0, "mp_excess_var": 0.0,
                "n_above_mp": 0, "n_below_mp": 0}
    q = N / T
    sigma2 = 1.0
    lam_plus = sigma2 * (1.0 + np.sqrt(q))**2
    lam_minus = sigma2 * (1.0 - np.sqrt(q))**2
    # Grid for KDE
    grid_min = max(eps, min(eigenvalues.min(), lam_minus) * 0.8)
    grid_max = max(eigenvalues.max(), lam_plus) * 1.2
    grid = np.linspace(grid_min, grid_max, n_grid)
    # Empirical density via Gaussian KDE
    bw = max(bandwidth, 1e-3)
    rho_emp = np.zeros_like(grid)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bw * len(eigenvalues))
    for lam in eigenvalues:
        rho_emp += np.exp(-0.5 * ((grid - lam) / bw)**2)
    rho_emp *= norm
    # MP density
    rho_mp = mp_pdf(grid, q=q, sigma2=sigma2, eps=eps)
    # Normalize
    dx = grid[1] - grid[0]
    rho_emp /= max(np.sum(rho_emp) * dx, eps)
    rho_mp /= max(np.sum(rho_mp) * dx, eps)
    # KL(empirical || MP)
    d_mp_kl = float(np.sum(rho_emp * np.log((rho_emp + eps) / (rho_mp + eps))) * dx)
    # Edge diagnostics
    n_above = int(np.sum(eigenvalues > lam_plus * 1.05))
    n_below = int(np.sum((eigenvalues > 0) & (eigenvalues < lam_minus * 0.95)))
    mp_excess = float(np.sum(eigenvalues[eigenvalues > lam_plus])) / max(
        np.sum(eigenvalues), eps)
    return {"d_mp_kl": d_mp_kl, "mp_excess_var": mp_excess,
            "n_above_mp": n_above, "n_below_mp": n_below}


# ══════════════════════════════════════════════════════════════════
# S2: LEMPEL-ZIV COMPLEXITY
# ══════════════════════════════════════════════════════════════════

def lempel_ziv_complexity(binary_string):
    """LZ76 complexity of a binary string."""
    n = len(binary_string)
    if n == 0:
        return 0
    i, c, l = 0, 1, 1
    while l + i < n:
        if binary_string[i + l] in binary_string[i:i + l]:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    return c

def compute_lz_from_corr_upper(C_upper, median_vals):
    """Symbolize upper-triangular correlations and compute LZ complexity."""
    binary = ''.join(['1' if c < m else '0' for c, m in zip(C_upper, median_vals)])
    lz = lempel_ziv_complexity(binary)
    L = len(binary)
    lz_norm = lz / max(L / max(np.log2(L + 1), 1), 1e-12) if L > 0 else 1.0
    return lz, lz_norm


# ══════════════════════════════════════════════════════════════════
# S3: PERSISTENT HOMOLOGY BARCODES
# ══════════════════════════════════════════════════════════════════

def compute_ph_barcodes(D):
    """Compute H0 and H1 persistence diagrams from distance matrix."""
    if not HAS_RIPSER:
        return None, None, 0.0, 0.0
    result = ripser_fn(D, maxdim=1, distance_matrix=True)
    dgm0 = result['dgms'][0]  # H0: connected components
    dgm1 = result['dgms'][1]  # H1: cycles

    # Persistence entropy for H1
    if len(dgm1) > 0:
        lifetimes = dgm1[:, 1] - dgm1[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
        if len(lifetimes) > 0:
            p = lifetimes / lifetimes.sum()
            pe1 = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            pe1 = 0.0
    else:
        pe1 = 0.0

    # Number of significant H1 features
    n_h1 = len(dgm1) if len(dgm1) > 0 else 0
    return dgm0, dgm1, pe1, n_h1


def wasserstein_distance_ph(dgm_a, dgm_b):
    """Wasserstein-2 distance between two H1 persistence diagrams."""
    if not HAS_RIPSER:
        return 0.0
    try:
        from persim import wasserstein as wass_dist
        # Filter out infinite deaths
        a = dgm_a[np.isfinite(dgm_a[:, 1])] if len(dgm_a) > 0 else np.empty((0, 2))
        b = dgm_b[np.isfinite(dgm_b[:, 1])] if len(dgm_b) > 0 else np.empty((0, 2))
        if len(a) == 0 and len(b) == 0:
            return 0.0
        return float(wass_dist(a, b, matching=False))
    except Exception:
        return 0.0


def compute_cdr(lz_history, W_cdr=10):
    """Complexity Depletion Rate: negative slope of LZ_norm over trailing window."""
    if len(lz_history) < W_cdr:
        return 0.0
    recent = np.array(lz_history[-W_cdr:])
    t = np.arange(W_cdr, dtype=float)
    t_mean = t.mean()
    t_var = np.sum((t - t_mean)**2)
    if t_var < 1e-12:
        return 0.0
    slope = np.sum((t - t_mean) * (recent - recent.mean())) / t_var
    return float(-slope)  # negative slope = CDR positive when complexity drops


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def download_returns(tickers, start, end):
    """Download prices and compute log-returns."""
    raw = yf.download(tickers=tickers, start=start, end=end,
                      interval="1d", auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]
    # Drop tickers with insufficient coverage
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def load_ground_truth(uid):
    """Load v2 state CSV and extract labels."""
    csv_path = V2_DIR / uid / "kappa_v2_state.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    return df


# ══════════════════════════════════════════════════════════════════
# MAIN EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════

def extract_universe(uid, uconfig):
    """Extract all SIG data for a single universe."""
    tickers = uconfig["tickers"]
    name = uconfig["name"]
    print(f"\n  [{uid}] {name} ({len(tickers)} tickers)...")

    # Download returns
    returns = download_returns(tickers, START_DATE, END_DATE)
    if returns is None or returns.shape[1] < 5:
        print(f"    SKIP: insufficient data ({returns.shape if returns is not None else 'None'})")
        return None

    R = returns.to_numpy(dtype=float)
    dates = returns.index
    N_assets = R.shape[1]
    ticker_names = list(returns.columns)
    print(f"    Returns: {R.shape[0]} days, {N_assets} assets")

    # Load ground truth
    gt_df = load_ground_truth(uid)

    # CALM median for LZ symbolization: first 40% of windows
    n_calm = int(R.shape[0] * 0.4)
    calm_corrs = []
    for t in range(WINDOW, min(WINDOW + n_calm, R.shape[0])):
        Rw = R[t - WINDOW:t]
        if Rw.shape[0] < WINDOW:
            continue
        C = shrink_corr(corr_spearman(Rw), SHRINK_LAM)
        iu = np.triu_indices(N_assets, k=1)
        calm_corrs.append(C[iu])
    if calm_corrs:
        median_corr = np.median(np.array(calm_corrs), axis=0)
    else:
        median_corr = np.zeros(N_assets * (N_assets - 1) // 2)

    # Rolling extraction
    all_eigenvalues = []
    all_scr = []
    all_sgr = []
    all_d_mp_kl = []
    all_mp_excess = []
    all_lz_norm = []
    all_cdr = []        # Complexity Depletion Rate
    all_pe1 = []
    all_tcs = []        # Topological Crystallization Score
    all_n_h1 = []
    all_mean_corr = []  # Baseline: mean pairwise correlation
    all_lam1 = []       # Baseline: leading eigenvalue raw
    all_rvol = []       # Baseline: realized volatility
    all_dates = []
    all_labels = []
    n_pairs = N_assets * (N_assets - 1) // 2

    # LZ buffer for windowed complexity
    lz_buffer = []
    lz_history = []     # for CDR slope
    W_LZ = 15
    W_CDR = 10          # trailing window for CDR slope

    # CALM reference barcode for TCS (first 40% of H1 diagrams)
    calm_dgms = []

    n_steps = R.shape[0]
    t0 = time.time()

    for t in range(WINDOW, n_steps):
        Rw = R[t - WINDOW:t]
        date_t = dates[t]

        # Step 1: Correlation + shrinkage (mirror engine_v4)
        C = shrink_corr(corr_spearman(Rw), SHRINK_LAM)
        D = angular_distance(C)
        iu = np.triu_indices(N_assets, k=1)
        C_upper = C[iu]

        # S1: Spectral
        eigenvalues, scr, sgr = compute_spectral(C)
        mp = compute_mp_metrics(eigenvalues, WINDOW, N_assets)

        all_eigenvalues.append(eigenvalues.astype(np.float32))
        all_scr.append(scr)
        all_sgr.append(sgr)
        all_d_mp_kl.append(mp["d_mp_kl"])
        all_mp_excess.append(mp["mp_excess_var"])


        # S2: Lempel-Ziv (windowed)
        lz_buffer.append(C_upper)
        if len(lz_buffer) > W_LZ:
            lz_buffer.pop(0)
        if len(lz_buffer) >= 3:
            concat_upper = np.concatenate(lz_buffer)
            concat_median = np.tile(median_corr, len(lz_buffer))
            _, lz_norm = compute_lz_from_corr_upper(concat_upper, concat_median)
        else:
            lz_norm = 1.0
        all_lz_norm.append(lz_norm)
        lz_history.append(lz_norm)
        # CDR: complexity depletion rate
        all_cdr.append(compute_cdr(lz_history, W_CDR))

        # S3: PH barcodes + TCS
        dgm0, dgm1, pe1, n_h1 = compute_ph_barcodes(D)
        all_pe1.append(pe1)
        all_n_h1.append(n_h1)
        # Collect CALM barcodes (first 40%) for TCS reference
        step_idx = t - WINDOW
        if step_idx < n_calm and dgm1 is not None and len(dgm1) > 0:
            calm_dgms.append(dgm1)
        # TCS computed after loop (needs CALM reference)

        # Baselines
        mean_c = float(np.mean(C[np.triu_indices(N_assets, k=1)])) if N_assets > 1 else 0.0
        all_mean_corr.append(mean_c)
        all_lam1.append(float(eigenvalues[0]))
        # Realized vol: std of returns in this window
        rvol = float(np.mean(np.std(Rw, axis=0)))
        all_rvol.append(rvol)

        # Date
        all_dates.append(str(date_t.date()) if hasattr(date_t, 'date') else str(date_t))


        # Ground truth label from v2 state
        label = "NOMINAL"
        if gt_df is not None:
            date_str = all_dates[-1]
            if date_str in gt_df.index.astype(str).values:
                row = gt_df.loc[gt_df.index.astype(str) == date_str]
                if len(row) > 0:
                    row = row.iloc[0]
                    theta_a = row.get("theta_A", 0.0)
                    c_norm = row.get("C_norm", 1.0)
                    if c_norm < 0.90:
                        label = "DAMAGED"
                    elif theta_a >= 2.0:
                        label = "CRYSTALLIZED"
                    elif theta_a > 0.1:
                        label = "CRYSTALLIZING"
        all_labels.append(label)

        # Progress
        if (t - WINDOW) % 200 == 0:
            elapsed = time.time() - t0
            pct = (t - WINDOW) / max(n_steps - WINDOW, 1) * 100
            print(f"    Step {t - WINDOW}/{n_steps - WINDOW} ({pct:.0f}%) — {elapsed:.1f}s")

    elapsed = time.time() - t0
    n_extracted = len(all_dates)
    print(f"    Extracted {n_extracted} steps in {elapsed:.1f}s")

    # Post-loop: compute TCS (Wasserstein distance to CALM barcode)
    if calm_dgms and HAS_RIPSER:
        # Use median-length CALM diagram as reference
        calm_ref = calm_dgms[len(calm_dgms) // 2]
        # Re-run PH to get barcodes for TCS (stored during main loop would use too much memory)
        print(f"    Computing TCS (Wasserstein vs CALM ref)...")
        for t in range(WINDOW, n_steps):
            Rw = R[t - WINDOW:t]
            C = shrink_corr(corr_spearman(Rw), SHRINK_LAM)
            D = angular_distance(C)
            _, dgm1_t, _, _ = compute_ph_barcodes(D)
            if dgm1_t is not None and len(dgm1_t) > 0:
                all_tcs.append(wasserstein_distance_ph(dgm1_t, calm_ref))
            else:
                all_tcs.append(0.0)
        print(f"    TCS computed ({len(all_tcs)} steps)")
    else:
        all_tcs = [0.0] * n_extracted


    # Save as compressed npz
    out_path = SIG_DIR / f"{uid}_sig.npz"
    np.savez_compressed(
        out_path,
        dates=np.array(all_dates),
        labels=np.array(all_labels),
        # S1: Spectral
        eigenvalues=np.array(all_eigenvalues, dtype=np.float32),
        scr=np.array(all_scr, dtype=np.float32),
        sgr=np.array(all_sgr, dtype=np.float32),
        d_mp_kl=np.array(all_d_mp_kl, dtype=np.float32),
        mp_excess=np.array(all_mp_excess, dtype=np.float32),
        # S2: LZ
        lz_norm=np.array(all_lz_norm, dtype=np.float32),
        cdr=np.array(all_cdr, dtype=np.float32),
        # S3: PH
        pe1=np.array(all_pe1, dtype=np.float32),
        n_h1=np.array(all_n_h1, dtype=np.int16),
        tcs=np.array(all_tcs, dtype=np.float32),
        # Baselines
        mean_corr=np.array(all_mean_corr, dtype=np.float32),
        lam1=np.array(all_lam1, dtype=np.float32),
        rvol=np.array(all_rvol, dtype=np.float32),
        # Meta
        tickers=np.array(ticker_names),
        n_assets=N_assets,
        window=WINDOW,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"    Saved: {out_path.name} ({size_kb:.0f} KB)")


    # Label distribution
    from collections import Counter
    lc = Counter(all_labels)
    print(f"    Labels: {dict(lc)}")

    return {
        "uid": uid, "n_steps": n_extracted, "n_assets": N_assets,
        "labels": dict(lc), "file": str(out_path),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  KAPPA-SIG Phase 1: Raw Structural Data Extraction")
    print("  David Ohio | Independent Researcher | April 2026")
    print("=" * 70)
    print(f"\n  Output: {SIG_DIR}")
    print(f"  Universes: {len(ALL_UNIVERSES)} total, excluding {EXCLUDE}")
    print(f"  Window: {WINDOW} days")
    print(f"  Methods: S1(Spectral) + S2(LZ) + S3(PH)")
    if not HAS_RIPSER:
        print("  [!] ripser not available — S3 will output zeros")


    results = []
    t_start = time.time()

    for uid, uconfig in sorted(ALL_UNIVERSES.items()):
        if uid in EXCLUDE:
            print(f"\n  [{uid}] SKIPPED (baseline insufficient)")
            continue
        try:
            r = extract_universe(uid, uconfig)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\n  [{uid}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Universes processed: {len(results)}/{len(ALL_UNIVERSES) - len(EXCLUDE)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Output directory: {SIG_DIR}")


    total_steps = sum(r["n_steps"] for r in results)
    print(f"  Total steps extracted: {total_steps}")
    print(f"\n  {'Universe':<28s} {'Steps':>6s} {'Assets':>6s} {'NOM':>5s} {'CRYZ':>5s} {'CRYD':>5s} {'DMG':>5s}")
    print(f"  {'-'*70}")
    for r in sorted(results, key=lambda x: x["uid"]):
        lc = r["labels"]
        print(f"  {r['uid']:<28s} {r['n_steps']:>6d} {r['n_assets']:>6d}"
              f" {lc.get('NOMINAL',0):>5d} {lc.get('CRYSTALLIZING',0):>5d}"
              f" {lc.get('CRYSTALLIZED',0):>5d} {lc.get('DAMAGED',0):>5d}")

    # Save manifest
    manifest = {
        "extraction_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_universes": len(results),
        "total_steps": total_steps,
        "window": WINDOW,
        "methods": ["S1_spectral", "S2_lz", "S3_ph"],
        "has_ripser": HAS_RIPSER,
        "universes": results,
    }
    manifest_path = SIG_DIR / "extraction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
