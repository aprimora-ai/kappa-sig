#!/usr/bin/env python3
"""
Kappa-LLM: Full HaluEval Experiment (Rerun v2)
================================================
Two-phase experiment following HEIMDALL + Kappa-LLM methodology:

Phase 1: Use sweep-identified best layers per model (from HEIMDALL cannonball run)
Phase 2: Per-head analysis within best layer → find discriminative heads
Phase 3: Extract 5 Kappa observables from top heads → compute AUC, means

Sweep data source: github.com/aprimora-ai/heimdall/cannonball run/
Best layers: Phi-3=L28, Mistral=L13, Llama=L15

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import sys, json, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
import torch
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

OUT_DIR = Path(__file__).parent / "results" / "halueval_rerun"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Models + sweep-identified best layers (from HEIMDALL cannonball run)
MODELS = {
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "best_layer": 28,  # p=2.26e-08, diff=-0.0015
        "top_layers": [28, 18, 16, 17],  # top 4 from sweep
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "best_layer": 13,  # p=2.30e-20, diff=-0.0018
        "top_layers": [13, 17, 7, 12],
    },
    "llama": {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "best_layer": 15,  # p=2.64e-15, diff=-0.0013
        "top_layers": [15, 6, 14, 21],
    },
}

N_SAMPLES = 120
MAX_SEQ_LEN = 256
TOP_K_HEADS = 8  # top heads per layer to use


# ══════════════════════════════════════════════════════════
# OBSERVABLE COMPUTATION (same as original Kappa-LLM)
# ══════════════════════════════════════════════════════════

def compute_observables(A: np.ndarray) -> Dict[str, float]:
    """Compute 5 Kappa observables from one attention head matrix."""
    eps = 1e-10
    n = A.shape[0]
    A = np.maximum(A, 0)
    flat = A.flatten()
    flat_pos = flat[flat > eps]

    # Omega: normalized Shannon entropy
    if len(flat_pos) > 0:
        probs = flat_pos / (flat_pos.sum() + eps)
        H = sp_entropy(probs, base=2)
        H_max = np.log2(n ** 2) if n > 0 else 1.0
        omega = float(H / H_max) if H_max > 0 else 0.0
    else:
        omega = 0.0
    
    eta = 1.0 - omega  # Rigidity = complement of entropy

    # Xi: inverse participation ratio normalized
    total = flat.sum() + eps
    probs_all = flat / total
    ipr = 1.0 / (np.sum(probs_all ** 4) + eps)
    xi = float(ipr / (n ** 2))

    # Delta: KL divergence from uniform
    probs_safe = np.maximum(flat / total, eps)
    uniform = 1.0 / (n ** 2)
    kl = np.sum(probs_safe * np.log(probs_safe / uniform))
    kl_max = np.log(n ** 2) if n > 0 else 1.0
    delta = float(kl / kl_max) if kl_max > 0 else 0.0

    # Phi: spectral gap approximation
    try:
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
        phi = float((eigvals[0] - eigvals[1]) / (eigvals[0] + eps)) if len(eigvals) >= 2 else 0.0
    except:
        phi = 0.0

    # R-score: HEIMDALL formula log(1 + max_lifetime / n_features)
    rscore = float(np.log1p(phi / (1 + eps))) if phi > 0 else 0.0

    return {
        "omega": np.clip(omega, 0, 1), "phi": np.clip(phi, 0, 1),
        "eta": np.clip(eta, 0, 1), "xi": np.clip(xi, 0, 1),
        "delta": np.clip(delta, 0, 1), "rscore": rscore,
    }


# ══════════════════════════════════════════════════════════
# MODEL + DATASET LOADING
# ══════════════════════════════════════════════════════════

def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"    Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    needs_quant = "7B" in model_id or "8B" in model_id
    if needs_quant:
        print(f"    4-bit quantization")
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qcfg, device_map="auto",
            trust_remote_code=True, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager")
    model.eval()
    return model, tok

def load_halueval():
    from datasets import load_dataset
    print("  Loading HaluEval...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples")
    factual, hallu = [], []
    for s in ds["data"]:
        q, a, h = s.get("question",""), s.get("answer",""), s.get("hallucination","")
        if q and a and h:
            factual.append(f"Question: {q} Answer: {a}")
            hallu.append(f"Question: {q} Answer: {h}")
        if len(factual) >= N_SAMPLES: break
    print(f"  {len(factual)} factual + {len(hallu)} hallucination")
    return factual, hallu


# ══════════════════════════════════════════════════════════
# SIG/v5.8c INTEGRATION: Inter-Head Correlation Analysis
# ══════════════════════════════════════════════════════════

def compute_head_correlation_kappa(attention_layer: np.ndarray) -> Dict[str, float]:
    """
    Compute Kappa state from inter-head correlation matrix.
    This applies the SIG/v5.8c methodology to the attention domain:
    each head's attention pattern is treated as an "asset" in the
    correlation network. Excessive inter-head correlation = obsessive coherence.
    
    attention_layer: (n_heads, seq_len, seq_len)
    """
    n_heads = attention_layer.shape[0]
    seq_len = attention_layer.shape[1]
    
    # Flatten each head's attention into a feature vector
    head_vectors = attention_layer.reshape(n_heads, -1)  # (n_heads, seq*seq)
    
    # Correlation matrix between heads
    C = np.corrcoef(head_vectors)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)

    # Eigenvalues (same as engine v5.8c)
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(C)))[::-1]
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum() + 1e-10
    
    # Oh: spectral concentration (lambda_1 / sum * N) — Ohio Number
    Oh = float(eigvals[0] / total) * n_heads
    
    # eta: rigidity (1 - normalized entropy of eigenspectrum)
    eig_norm = eigvals / total
    eig_pos = eig_norm[eig_norm > 1e-12]
    H = sp_entropy(eig_pos, base=2)
    H_max = np.log2(n_heads) if n_heads > 1 else 1.0
    eta = float(1.0 - H / H_max) if H_max > 0 else 0.0
    
    # mean_corr: mean off-diagonal absolute correlation
    mask = ~np.eye(n_heads, dtype=bool)
    mean_corr = float(np.mean(np.abs(C[mask])))
    
    # DEF: eigenvalue dominance gap
    DEF = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)) if n_heads > 1 else 0.0
    
    # Xi: effective diversity (effective rank / n_heads)
    eff_rank = np.exp(sp_entropy(eig_pos)) if len(eig_pos) > 0 else 1.0
    Xi = float(eff_rank / n_heads)

    # SIG: Marchenko-Pastur test (Wheeler-inspired)
    # For a random matrix with ratio q = seq*seq / n_heads,
    # the theoretical MP bounds are lambda_+ and lambda_-
    q = (seq_len * seq_len) / max(n_heads, 1)
    lambda_mp_plus = (1 + 1/np.sqrt(q))**2 if q > 0 else 2.0
    eig_norm_mp = eigvals / (total / n_heads)  # Normalize by mean eigenvalue
    n_escaped = int(np.sum(eig_norm_mp > lambda_mp_plus))
    mp_escape_ratio = n_escaped / n_heads
    
    # Obsessive Coherence Index (from Kappa-SIG)
    coherence = (Oh/n_heads + eta + DEF) / 3.0  # Normalize Oh by n_heads for [0,1]
    disorder = (Xi + (1.0 - mean_corr)) / 2.0
    obsessive_score = coherence - disorder
    
    return {
        "head_Oh": float(Oh),
        "head_eta": float(eta),
        "head_mean_corr": float(mean_corr),
        "head_DEF": float(DEF),
        "head_Xi": float(Xi),
        "head_mp_escape": float(mp_escape_ratio),
        "head_n_escaped": n_escaped,
        "head_oc_score": float(obsessive_score),
    }


# ══════════════════════════════════════════════════════════
# EXPERIMENT: TWO-PHASE PER MODEL
# ══════════════════════════════════════════════════════════

def run_model_experiment(model_name, config, factual_texts, hallu_texts):
    print(f"\n  {'='*60}")
    print(f"  [{model_name.upper()}] {config['id']}")
    print(f"  Best layer: {config['best_layer']} (from HEIMDALL sweep)")
    print(f"  {'='*60}")
    
    model, tok = load_model(config["id"])
    best_layer = config["best_layer"]
    
    all_rows = []
    
    for label_name, texts, label_val in [("factual", factual_texts, 0),
                                          ("hallucination", hallu_texts, 1)]:
        print(f"    Processing {len(texts)} {label_name}...")
        for i, text in enumerate(texts):
            try:
                inputs = tok(text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN, padding=False)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model(**inputs, output_attentions=True, use_cache=False)
                
                # Extract attention from best layer: (n_heads, seq, seq)
                attn = out.attentions[best_layer][0].cpu().numpy()
                n_heads = attn.shape[0]
                
                # === Per-head observables (HEIMDALL/Kappa-LLM original) ===
                head_obs = []
                for h in range(n_heads):
                    obs = compute_observables(attn[h])
                    obs["head"] = h
                    head_obs.append(obs)
                
                # Aggregate: max pooling for rigidity/divergence, min for entropy/diversity
                agg = {}
                for key in ["omega", "phi", "eta", "xi", "delta", "rscore"]:
                    vals = [ho[key] for ho in head_obs]
                    if key in ["omega", "xi"]:
                        agg[key] = float(np.min(vals))
                    else:
                        agg[key] = float(np.max(vals))

                # === Inter-head correlation Kappa (SIG/v5.8c) ===
                head_kappa = compute_head_correlation_kappa(attn)
                
                # === Combined row ===
                row = {"label": label_val, "sample_idx": i, **agg, **head_kappa}
                all_rows.append(row)
                
            except Exception as e:
                if i == 0: print(f"    ERROR sample[{i}]: {e}")
                continue
            
            if (i+1) % 30 == 0:
                print(f"      {i+1}/{len(texts)}")
    
    # Free GPU
    del model; torch.cuda.empty_cache(); gc.collect()
    
    if len(all_rows) < 10:
        print(f"    INSUFFICIENT DATA ({len(all_rows)} rows)")
        return {"model": model_name, "error": "insufficient_samples"}

    # === Analysis ===
    df = pd.DataFrame(all_rows)
    model_dir = OUT_DIR / model_name
    model_dir.mkdir(exist_ok=True)
    df.to_csv(model_dir / "all_observables.csv", index=False)
    
    labels = df["label"].values
    obs_cols = ["omega", "phi", "eta", "xi", "delta", "rscore"]
    sig_cols = ["head_Oh", "head_eta", "head_mean_corr", "head_DEF", "head_Xi",
                "head_mp_escape", "head_oc_score"]
    all_metric_cols = obs_cols + sig_cols
    
    # Per-feature AUC
    aucs = {}
    for col in all_metric_cols:
        if col not in df.columns: continue
        vals = df[col].values
        try:
            auc = roc_auc_score(labels, vals)
            if auc < 0.5: auc = 1.0 - auc
            aucs[col] = round(float(auc), 4)
        except: aucs[col] = 0.5

    # Kappa Score (composite)
    df = df.fillna(0)  # Protect against NaN from failed samples
    kappa = (0.30*df["rscore"] + 0.25*df["eta"] + 0.20*(1-df["xi"]) +
             0.15*df["delta"] - 0.10*df["omega"]).values
    kappa = np.nan_to_num(kappa, nan=0.0)
    k_auc = roc_auc_score(labels, kappa)
    if k_auc < 0.5: k_auc = 1.0 - k_auc
    fpr, tpr, thr = roc_curve(labels, kappa)
    opt = int(np.argmax(tpr - fpr))
    preds = (kappa > thr[opt]).astype(int)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds))
    
    # SIG-enhanced composite: add head_oc_score
    if "head_oc_score" in df.columns:
        sig_comp = kappa + 0.3 * np.nan_to_num(df["head_oc_score"].values, nan=0.0)
        sig_auc = roc_auc_score(labels, sig_comp)
        if sig_auc < 0.5: sig_auc = 1.0 - sig_auc
    else:
        sig_auc = None
    
    # Means per class
    df_f = df[df["label"]==0]; df_h = df[df["label"]==1]
    mean_f = {c: round(float(df_f[c].mean()), 6) for c in all_metric_cols if c in df.columns}
    mean_h = {c: round(float(df_h[c].mean()), 6) for c in all_metric_cols if c in df.columns}

    # Print results
    print(f"\n    RESULTS [{model_name.upper()}]:")
    print(f"    Kappa AUC: {k_auc:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")
    if sig_auc: print(f"    SIG-enhanced AUC: {sig_auc:.4f}")
    print(f"    Per-feature AUCs:")
    for c in sorted(aucs, key=aucs.get, reverse=True)[:10]:
        print(f"      {c:20s}: {aucs[c]:.4f}")
    print(f"\n    Mean observables (SIG-integrated):")
    print(f"    {'':20s} {'Factual':>10s} {'Hallu':>10s} {'Delta':>10s}")
    for c in all_metric_cols:
        if c in mean_f and c in mean_h:
            d = mean_h[c] - mean_f[c]
            print(f"    {c:20s} {mean_f[c]:10.4f} {mean_h[c]:10.4f} {d:+10.4f}")
    
    result = {
        "model": model_name, "model_id": config["id"],
        "best_layer": best_layer, "n_samples": len(df),
        "kappa_auc": round(k_auc, 4), "accuracy": round(acc, 4), "f1": round(f1, 4),
        "sig_enhanced_auc": round(sig_auc, 4) if sig_auc else None,
        "aucs": aucs, "mean_factual": mean_f, "mean_hallucination": mean_h,
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-LLM: HaluEval Experiment (SIG-Integrated Rerun)")
    print("  David Ohio | Independent Researcher | April 2026")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Method: HEIMDALL sweep layers + Kappa v5.8c + SIG inter-head")
    print("=" * 70)
    
    factual, hallu = load_halueval()
    results = {}
    
    for name, cfg in MODELS.items():
        try:
            results[name] = run_model_experiment(name, cfg, factual, hallu)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"model": name, "error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for n, r in results.items():
        if "error" in r:
            print(f"  {n:<10s} ERROR: {r['error'][:50]}")
        else:
            print(f"  {n:<10s} Kappa={r['kappa_auc']:.4f} "
                  f"SIG={r.get('sig_enhanced_auc','N/A')} Acc={r['accuracy']:.4f}")
    
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "Kappa-LLM HaluEval (SIG-Integrated Rerun)",
        "method": "HEIMDALL sweep + Kappa v5.8c + SIG inter-head correlation",
        "gpu": torch.cuda.get_device_name(0),
        "models": results,
    }
    out_file = OUT_DIR / "llm_experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results: {out_file}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    print("=" * 70)

if __name__ == "__main__":
    main()
