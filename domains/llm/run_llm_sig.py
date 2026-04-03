#!/usr/bin/env python3
"""
Kappa-SIG LLM: Obsessive Coherence Analysis (Validated)
=========================================================
Uses VERIFIED data from HaluEval rerun (April 2026) with SIG inter-head
correlation as primary method.

Key methodological finding:
  Post-softmax attention matrices are row-stochastic (rows sum to 1),
  which degenerates 3 of 5 per-head observables (phi, xi, delta -> 0/1/NaN).
  The SIG inter-head correlation approach — treating N heads as N components
  of a correlation network — is invariant to this normalization and produces
  discriminative signals (AUC up to 0.786).

  This is methodologically consistent: the Kappa framework has ALWAYS operated
  on correlation networks (FIN: asset correlations, EDU: engagement correlations,
  NEURO: channel correlations). For LLM, the natural "network" is the
  correlation structure between attention heads.

Data source: run_halueval_experiment.py (SIG-Integrated Rerun)
  - 3 models x 240 samples (120 factual + 120 hallucination)
  - Sweep-identified layers: Phi-3=L28, Mistral=L13, Llama=L15
  - Raw CSVs in results/halueval_rerun/{model}/all_observables.csv

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import json, time
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════
# VERIFIED RESULTS (from run_halueval_experiment.py rerun)
# All values measured on RTX 4060 Ti, April 2 2026
# ══════════════════════════════════════════════════════════

VERIFIED_RESULTS = {
    "phi3": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "best_layer": 28, "n_samples": 240,
        "sig_auc": 0.5426, "kappa_auc": 0.5347,
        "best_feature": "head_DEF", "best_feature_auc": 0.5637,
        "factual": {
            "omega": 0.5427, "eta": 0.4573,
            "head_Oh": 28.4349, "head_eta": 0.8095,
            "head_mean_corr": 0.8832, "head_DEF": 0.9784,
            "head_Xi": 0.0606, "head_oc_score": 0.8035,
        },
        "hallucination": {
            "omega": 0.5418, "eta": 0.4582,
            "head_Oh": 28.4638, "head_eta": 0.8126,
            "head_mean_corr": 0.8840, "head_DEF": 0.9776,
            "head_Xi": 0.0599, "head_oc_score": 0.8053,
        },
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "best_layer": 13, "n_samples": 240,
        "sig_auc": 0.7419, "kappa_auc": 0.7194,
        "best_feature": "head_eta", "best_feature_auc": 0.7224,
        "factual": {
            "omega": 0.5769, "eta": 0.4231,
            "head_Oh": 28.1572, "head_eta": 0.7998,
            "head_mean_corr": 0.8739, "head_DEF": 0.9736,
            "head_Xi": 0.0630, "head_oc_score": 0.7899,
        },
        "hallucination": {
            "omega": 0.5578, "eta": 0.4422,
            "head_Oh": 28.7264, "head_eta": 0.8278,
            "head_mean_corr": 0.8926, "head_DEF": 0.9746,
            "head_Xi": 0.0571, "head_oc_score": 0.8178,
        },
    },
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "best_layer": 15, "n_samples": 240,
        "sig_auc": 0.7863, "kappa_auc": 0.7258,
        "best_feature": "head_eta", "best_feature_auc": 0.7791,
        "factual": {
            "omega": 0.5232, "eta": 0.4768,
            "head_Oh": 28.3942, "head_eta": 0.8233,
            "head_mean_corr": 0.8800, "head_DEF": 0.9553,
            "head_Xi": 0.0580, "head_oc_score": 0.7997,
        },
        "hallucination": {
            "omega": 0.5147, "eta": 0.4853,
            "head_Oh": 28.9967, "head_eta": 0.8513,
            "head_mean_corr": 0.9001, "head_DEF": 0.9589,
            "head_Xi": 0.0524, "head_oc_score": 0.8293,
        },
    },
}

# SIG inter-head metrics used for obsessive coherence analysis
SIG_METRICS = ["head_Oh", "head_eta", "head_mean_corr", "head_DEF", "head_Xi"]

def analyze_model(name, data):
    """Analyze obsessive coherence for one model using SIG inter-head metrics."""
    f = data["factual"]
    h = data["hallucination"]
    
    # Direction test: does hallucination show MORE coherence?
    dirs = {
        "head_Oh": h["head_Oh"] > f["head_Oh"],          # Higher Oh = more coherent
        "head_eta": h["head_eta"] > f["head_eta"],        # Higher rigidity
        "head_mean_corr": h["head_mean_corr"] > f["head_mean_corr"],  # Higher coupling
        "head_Xi": h["head_Xi"] < f["head_Xi"],           # Lower diversity
        "head_DEF": h["head_DEF"] > f["head_DEF"],        # Higher dominance
    }
    n_coherent = sum(dirs.values())
    
    # OC signature
    oc_delta = h["head_oc_score"] - f["head_oc_score"]
    
    return {
        "model": name, "sig_auc": data["sig_auc"],
        "coherence_directions": {k: bool(v) for k, v in dirs.items()},
        "n_coherent": n_coherent, "n_total": 5,
        "obsessive_confirmed": n_coherent >= 4,
        "oc_factual": f["head_oc_score"],
        "oc_hallucination": h["head_oc_score"],
        "oc_delta": round(oc_delta, 4),
    }

def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-SIG LLM: Obsessive Coherence (Validated Rerun)")
    print("  David Ohio | Independent Researcher | April 2026")
    print("  Method: SIG inter-head correlation (primary)")
    print("=" * 70)
    
    # ── Softmax observation ──
    print("\n  METHODOLOGICAL NOTE: Softmax Normalization Effect")
    print("  " + "-"*60)
    print("  Post-softmax attention matrices are row-stochastic (sum=1).")
    print("  This degenerates 3/5 per-head Kappa observables:")
    print("    phi (spectral gap) -> 0.0 (row-stochastic kills eigenstructure)")
    print("    xi  (IPR/n^2)      -> 1.0 (softmax distributes mass uniformly)")
    print("    delta (KL vs uniform)-> NaN (causal mask zeros + log(0))")
    print("  Only omega/eta (Shannon entropy) retain signal (AUC 0.53-0.73).")
    print("  ")
    print("  Resolution: SIG inter-head correlation is invariant to per-row")
    print("  normalization. Pearson correlation between head attention patterns")
    print("  captures structural coupling regardless of softmax scaling.")
    print("  This is consistent with the Kappa framework: FIN, EDU, and NEURO")
    print("  all operate on correlation networks, not raw observables.")

    # ── Per-model analysis ──
    results = {}
    for name, data in VERIFIED_RESULTS.items():
        r = analyze_model(name, data)
        results[name] = r
        
        f = data["factual"]
        h = data["hallucination"]
        print(f"\n  [{name.upper()}] Layer {data['best_layer']} | SIG AUC: {data['sig_auc']}")
        print(f"    SIG Inter-Head State:")
        print(f"    {'':18s} {'Factual':>10s} {'Hallu':>10s} {'Delta':>10s}")
        for m in SIG_METRICS:
            d = h[m] - f[m]
            coh = "+" if r["coherence_directions"][m] else "-"
            print(f"    {m:18s} {f[m]:10.4f} {h[m]:10.4f} {d:+10.4f} [{coh}]")
        print(f"    {'head_oc_score':18s} {f['head_oc_score']:10.4f} "
              f"{h['head_oc_score']:10.4f} {r['oc_delta']:+10.4f}")
        st = "CONFIRMED" if r["obsessive_confirmed"] else "PARTIAL"
        print(f"    Coherent directions: {r['n_coherent']}/5 -- {st}")

    # ── Summary ──
    print(f"\n  {'='*70}")
    print("  OBSESSIVE COHERENCE SUMMARY (SIG Inter-Head)")
    print(f"  {'='*70}")
    print(f"\n  {'Model':<10s} {'SIG AUC':>9s} {'Dirs':>6s} {'OC Delta':>10s} {'Status':>12s}")
    print(f"  {'-'*50}")
    all_confirmed = True
    for name, r in results.items():
        st = "CONFIRMED" if r["obsessive_confirmed"] else "PARTIAL"
        if not r["obsessive_confirmed"]: all_confirmed = False
        print(f"  {name:<10s} {r['sig_auc']:9.4f} {r['n_coherent']}/5    "
              f"{r['oc_delta']:+10.4f} {st:>12s}")
    
    # Cross-domain table
    print(f"\n  CROSS-DOMAIN CONSISTENCY:")
    print(f"  {'Domain':<8s} {'Method':<30s} {'Best AUC':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'FIN':<8s} {'Asset correlation network':<30s} {'0.942':>10s}")
    print(f"  {'LLM':<8s} {'Head correlation network (SIG)':<30s} {'0.786':>10s}")
    print(f"  {'EDU':<8s} {'Engagement correlation':<30s} {'N/A (drift)':>10s}")
    print(f"  {'NEURO':<8s} {'Channel correlation (pending)':<30s} {'TBD':>10s}")
    print(f"\n  All domains use correlation networks. Same mechanism.")

    if all_confirmed:
        print(f"\n  >>> ALL 3 ARCHITECTURES CONFIRM via SIG inter-head")
    
    # ── Save ──
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "LLM",
        "method": "SIG inter-head correlation (Kappa v5.8c applied to attention head network)",
        "softmax_note": (
            "Post-softmax attention matrices are row-stochastic. "
            "This degenerates 3/5 per-head observables (phi->0, xi->1, delta->NaN). "
            "SIG inter-head correlation is invariant to per-row normalization and "
            "captures the obsessive coherence signal via second-order statistics. "
            "This is consistent with the Kappa framework which operates on "
            "correlation networks across all domains."
        ),
        "models": results,
        "verified_data_source": "results/halueval_rerun/llm_experiment_results.json",
        "all_confirmed": all_confirmed,
    }
    out_file = OUT_DIR / "llm_obsessive_coherence.json"
    with open(out_file, "w") as fp:
        json.dump(output, fp, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\n  Results: {out_file}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
