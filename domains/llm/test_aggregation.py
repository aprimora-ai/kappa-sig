#!/usr/bin/env python3
"""
Aggregation Comparison Test: Mean vs Max Pooling
=================================================
Tests both HEIMDALL-original (mean) and max-pooling aggregation
on Phi-3 best layer (L28) to determine which captures the
obsessive coherence signal better.

Scientific protocol:
  - Same model, same data, same layer
  - Only difference: aggregation method
  - Both results reported transparently

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import torch, warnings, json, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "results" / "aggregation_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
BEST_LAYER = 28
N_SAMPLES = 120
MAX_SEQ_LEN = 256

def compute_obs(A):
    eps = 1e-10; n = A.shape[0]
    A = np.maximum(A, 0); flat = A.flatten()
    flat_pos = flat[flat > eps]
    if len(flat_pos) > 0:
        probs = flat_pos / (flat_pos.sum() + eps)
        H = sp_entropy(probs, base=2)
        H_max = np.log2(n**2) if n > 0 else 1.0
        omega = float(H / H_max) if H_max > 0 else 0.0
    else: omega = 0.0
    eta = 1.0 - omega
    total = flat.sum() + eps
    pa = flat / total
    ipr = 1.0 / (np.sum(pa**4) + eps)
    xi = float(ipr / (n**2))
    ps = np.maximum(pa, eps)
    u = 1.0 / (n**2)
    kl = np.sum(ps * np.log(ps / u))
    klm = np.log(n**2) if n > 0 else 1.0
    delta = float(kl / klm) if klm > 0 else 0.0
    try:
        ev = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
        phi = float((ev[0]-ev[1])/(ev[0]+eps)) if len(ev)>=2 else 0.0
    except: phi = 0.0
    rscore = float(np.log1p(phi/(1+eps))) if phi > 0 else 0.0
    return {k: np.clip(v,0,1) for k,v in
            {"omega":omega,"phi":phi,"eta":eta,"xi":xi,"delta":delta,"rscore":rscore}.items()}

def main():
    t0 = time.time()
    print("="*70)
    print("  AGGREGATION COMPARISON: Mean (HEIMDALL) vs Max/Min (Kappa-LLM)")
    print("  Model: Phi-3 | Layer: 28 | Protocol: Same data, different aggregation")
    print("="*70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    model.eval()

    # Load HaluEval
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa_samples")
    factual, hallu = [], []
    for s in ds["data"]:
        q, a, h = s.get("question",""), s.get("answer",""), s.get("hallucination","")
        if q and a and h:
            factual.append(f"Question: {q} Answer: {a}")
            hallu.append(f"Question: {q} Answer: {h}")
        if len(factual) >= N_SAMPLES: break
    print(f"  Samples: {len(factual)} per class")

    # Process both classes
    rows = []
    for label_name, texts, label in [("factual", factual, 0), ("hallu", hallu, 1)]:
        print(f"  Processing {label_name}...")
        for i, text in enumerate(texts):
            try:
                inputs = tok(text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, padding=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs, output_attentions=True, use_cache=False)
                attn = out.attentions[BEST_LAYER][0].cpu().numpy()  # (n_heads, seq, seq)
                
                # METHOD A: Mean aggregation (HEIMDALL original)
                mean_attn = attn.mean(axis=0)  # Average all heads
                obs_mean = compute_obs(mean_attn)
                obs_mean = {f"mean_{k}": v for k, v in obs_mean.items()}
                
                # METHOD B: Max/Min pooling (current Kappa-LLM)
                head_obs_list = [compute_obs(attn[h]) for h in range(attn.shape[0])]
                obs_maxmin = {}
                for key in ["omega","phi","eta","xi","delta","rscore"]:
                    vals = [ho[key] for ho in head_obs_list]
                    obs_maxmin[f"maxmin_{key}"] = float(np.min(vals)) if key in ["omega","xi"] else float(np.max(vals))

                # METHOD C: HEIMDALL R-Score approach (mean heads, contrast amplification)
                mean_attn_c = np.power(mean_attn, 0.5)  # Contrast amplification from HEIMDALL
                m_min, m_max = mean_attn_c.min(), mean_attn_c.max()
                if m_max - m_min > 1e-12:
                    mean_attn_c = (mean_attn_c - m_min) / (m_max - m_min)
                obs_heimdall = compute_obs(mean_attn_c)
                obs_heimdall = {f"heimdall_{k}": v for k, v in obs_heimdall.items()}
                
                rows.append({"label": label, "idx": i, **obs_mean, **obs_maxmin, **obs_heimdall})
            except Exception as e:
                if i == 0: print(f"    ERROR: {e}")
            if (i+1) % 40 == 0: print(f"    {i+1}/{len(texts)}")
    
    del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "phi3_aggregation_comparison.csv", index=False)

    # Compute AUCs for all three methods
    labels = df["label"].values
    print(f"\n  {'='*70}")
    print(f"  RESULTS: Aggregation Comparison (n={len(df)})")
    print(f"  {'='*70}")
    
    methods = {"A_MEAN (HEIMDALL)": "mean_", "B_MAXMIN (current)": "maxmin_", 
               "C_HEIMDALL+contrast": "heimdall_"}
    
    for method_name, prefix in methods.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        print(f"\n  [{method_name}]")
        for col in cols:
            vals = df[col].values
            try:
                auc = roc_auc_score(labels, vals)
                if auc < 0.5: auc = 1.0 - auc
            except: auc = 0.5
            obs_name = col.replace(prefix, "")
            mf = df[df["label"]==0][col].mean()
            mh = df[df["label"]==1][col].mean()
            d = mh - mf
            print(f"    {obs_name:10s} AUC={auc:.4f}  fact={mf:.4f}  hallu={mh:.4f}  delta={d:+.4f}")

    # Composite Kappa score for each method
    for method_name, prefix in methods.items():
        try:
            k = (0.30*df[f"{prefix}rscore"] + 0.25*df[f"{prefix}eta"] + 
                 0.20*(1-df[f"{prefix}xi"]) + 0.15*df[f"{prefix}delta"] - 
                 0.10*df[f"{prefix}omega"]).values
            k = np.nan_to_num(k, nan=0.0)
            kauc = roc_auc_score(labels, k)
            if kauc < 0.5: kauc = 1.0 - kauc
            print(f"\n  {method_name} Kappa composite AUC: {kauc:.4f}")
        except Exception as e:
            print(f"\n  {method_name} Kappa composite: ERROR ({e})")
    
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed/60:.1f} min")
    print(f"  CSV: {OUT_DIR / 'phi3_aggregation_comparison.csv'}")
    print("="*70)

if __name__ == "__main__":
    main()
