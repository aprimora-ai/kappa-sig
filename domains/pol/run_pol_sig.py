#!/usr/bin/env python3
"""
Kappa-SIG POL: Obsessive Coherence in Political Polarization
==============================================================
Static network domain. Adjacency matrix = correlation analog.
Echo chambers should show higher Oh (spectral concentration).

David Ohio | odavidohio@gmail.com | Independent Researcher
April 2026
"""
import json, time, re, os
import numpy as np
from pathlib import Path
from scipy.stats import entropy as sp_entropy

DATA_FILE = Path(__file__).parent / "data" / "polblogs.gml"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_polblogs_gml(path):
    """Parse polblogs.gml manually (handles duplicate edges)."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Parse nodes: id + value (0=liberal, 1=conservative)
    node_pattern = r'node\s*\[\s*id\s+(\d+).*?value\s+(\d+).*?\]'
    nodes = re.findall(node_pattern, content, re.DOTALL)
    labels = {int(nid): int(val) for nid, val in nodes}
    
    # Parse edges
    edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)\s*\]'
    edges_raw = re.findall(edge_pattern, content)
    edges = set()
    for s, t in edges_raw:
        edges.add((int(s), int(t)))
    
    print(f"  Parsed: {len(labels)} nodes, {len(edges)} unique edges")
    print(f"  Liberal (0): {sum(1 for v in labels.values() if v==0)}")
    print(f"  Conservative (1): {sum(1 for v in labels.values() if v==1)}")
    return labels, edges

def split_network(labels, edges):
    """Split edges into intra-community (echo) and inter-community (bridge)."""
    intra_edges = []  # Same label
    inter_edges = []  # Different label
    for s, t in edges:
        if s in labels and t in labels:
            if labels[s] == labels[t]:
                intra_edges.append((s, t))
            else:
                inter_edges.append((s, t))
    print(f"  Intra (echo): {len(intra_edges)} edges")
    print(f"  Inter (bridge): {len(inter_edges)} edges")
    print(f"  Ratio: {len(intra_edges)/max(len(inter_edges),1):.1f}x")
    return intra_edges, inter_edges


def build_adjacency(nodes, edges):
    """Build symmetric adjacency matrix from node list and edge list."""
    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)
    A = np.zeros((n, n), dtype=np.float32)
    for s, t in edges:
        if s in idx and t in idx:
            A[idx[s], idx[t]] = 1.0
            A[idx[t], idx[s]] = 1.0  # Symmetrize
    return A, node_list

def compute_kappa_from_adjacency(A):
    """
    Compute Kappa state from adjacency matrix eigenstructure.
    Same methodology as FIN (correlation matrix) — the adjacency matrix
    IS the structural coupling matrix for a social network.
    """
    n = A.shape[0]
    if n < 3:
        return {"Oh": 0, "eta": 0, "mean_corr": 0, "DEF": 0, "Xi": 1, "n": n}
    
    # For large matrices, use sparse eigendecomposition
    if n > 500:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        A_sparse = csr_matrix(A)
        k = min(50, n-2)
        eigvals = eigsh(A_sparse, k=k, which='LM', return_eigenvectors=False)
        eigvals = np.sort(np.abs(eigvals))[::-1]
    else:
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1]
    
    eigvals = np.maximum(eigvals, 1e-12)
    total = eigvals.sum()

    # Oh: spectral concentration
    Oh = float(eigvals[0] / (total / n)) if total > 0 else 0.0
    
    # eta: rigidity (1 - normalized spectral entropy)
    eig_norm = eigvals / total
    eig_pos = eig_norm[eig_norm > 1e-12]
    H = sp_entropy(eig_pos, base=2)
    H_max = np.log2(len(eig_pos)) if len(eig_pos) > 1 else 1.0
    eta = float(1.0 - H / H_max) if H_max > 0 else 0.0
    
    # mean_corr: mean adjacency (edge density)
    mask = ~np.eye(n, dtype=bool)
    mean_corr = float(np.mean(A[mask]))
    
    # DEF: eigenvalue dominance gap
    DEF = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)) if len(eigvals) > 1 else 0.0
    
    # Xi: effective diversity
    eff_rank = np.exp(sp_entropy(eig_pos)) if len(eig_pos) > 0 else 1.0
    Xi = float(eff_rank / len(eig_pos))
    
    return {"Oh": round(Oh, 4), "eta": round(eta, 4), "mean_corr": round(mean_corr, 6),
            "DEF": round(DEF, 4), "Xi": round(Xi, 4), "n_nodes": n,
            "lambda_1": round(float(eigvals[0]), 2),
            "eff_rank": round(float(eff_rank), 2)}

def main():
    t0 = time.time()
    print("=" * 70)
    print("  KAPPA-SIG POL: Obsessive Coherence in Political Polarization")
    print("  David Ohio | Independent Researcher | April 2026")
    print("  Dataset: Political Blogosphere 2004 (Adamic & Glance)")
    print("  Method: Adjacency matrix eigenstructure (SIG/v5.8c)")
    print("=" * 70)
    
    # Load network
    labels, edges = parse_polblogs_gml(DATA_FILE)
    all_nodes = set(labels.keys())
    
    # Split
    intra_edges, inter_edges = split_network(labels, edges)
    
    # Build subgraphs
    # 1. Full network
    # 2. Intra-community only (echo chambers)
    # 3. Inter-community only (bridging)
    # 4. Liberal subgraph (intra only)
    # 5. Conservative subgraph (intra only)
    
    lib_nodes = {n for n, v in labels.items() if v == 0}
    con_nodes = {n for n, v in labels.items() if v == 1}
    lib_intra = [(s,t) for s,t in intra_edges if s in lib_nodes]
    con_intra = [(s,t) for s,t in intra_edges if s in con_nodes]

    conditions = {
        "full_network": (all_nodes, list(edges)),
        "echo_chamber": (all_nodes, intra_edges),
        "bridging": (all_nodes, inter_edges),
        "liberal_echo": (lib_nodes, lib_intra),
        "conservative_echo": (con_nodes, con_intra),
    }
    
    results = {}
    print(f"\n  {'Condition':<22s} {'N':>6s} {'Oh':>8s} {'eta':>8s} {'DEF':>8s} "
          f"{'Xi':>8s} {'m_corr':>10s} {'lam1':>8s}")
    print(f"  {'-'*80}")
    
    for cond_name, (nodes, edge_list) in conditions.items():
        A, node_list = build_adjacency(nodes, edge_list)
        kappa = compute_kappa_from_adjacency(A)
        results[cond_name] = kappa
        print(f"  {cond_name:<22s} {kappa['n_nodes']:6d} {kappa['Oh']:8.4f} "
              f"{kappa['eta']:8.4f} {kappa['DEF']:8.4f} {kappa['Xi']:8.4f} "
              f"{kappa['mean_corr']:10.6f} {kappa['lambda_1']:8.2f}")

    # ── Hypothesis test ──
    echo = results["echo_chamber"]
    bridge = results["bridging"]
    
    print(f"\n  {'='*70}")
    print("  OBSESSIVE COHERENCE HYPOTHESIS")
    print(f"  {'='*70}")
    print(f"\n  Echo chamber vs Bridging:")
    print(f"    {'Metric':<15s} {'Echo':>10s} {'Bridge':>10s} {'Delta':>10s} {'Coherent':>10s}")
    print(f"    {'-'*55}")
    
    dirs = {
        "Oh": echo["Oh"] > bridge["Oh"],
        "eta": echo["eta"] > bridge["eta"],
        "DEF": echo["DEF"] > bridge["DEF"],
        "Xi": echo["Xi"] < bridge["Xi"],
        "mean_corr": echo["mean_corr"] > bridge["mean_corr"],
    }
    
    for m in ["Oh", "eta", "DEF", "Xi", "mean_corr"]:
        d = echo[m] - bridge[m]
        coh = "YES" if dirs[m] else "no"
        print(f"    {m:<15s} {echo[m]:10.4f} {bridge[m]:10.4f} {d:+10.4f} {coh:>10s}")
    
    n_coherent = sum(dirs.values())
    confirmed = n_coherent >= 4
    print(f"\n  Coherent directions: {n_coherent}/5 -- "
          f"{'CONFIRMED' if confirmed else 'PARTIAL'}")

    # OC scores
    def oc_score(r):
        coh = (r["Oh"]/r["n_nodes"] + r["DEF"]) / 2.0
        dis = (r["Xi"] + (1.0 - r["mean_corr"])) / 2.0
        return coh - dis
    
    oc_echo = oc_score(echo)
    oc_bridge = oc_score(bridge)
    print(f"\n  OC Score: echo={oc_echo:+.4f}  bridge={oc_bridge:+.4f}  "
          f"delta={oc_echo - oc_bridge:+.4f}")
    
    # Liberal vs Conservative echo chambers
    lib = results["liberal_echo"]
    con = results["conservative_echo"]
    print(f"\n  Liberal vs Conservative echo chambers:")
    print(f"    Liberal:      Oh={lib['Oh']:.4f}  eta={lib['eta']:.4f}  "
          f"Xi={lib['Xi']:.4f}  DEF={lib['DEF']:.4f}")
    print(f"    Conservative: Oh={con['Oh']:.4f}  eta={con['eta']:.4f}  "
          f"Xi={con['Xi']:.4f}  DEF={con['DEF']:.4f}")
    
    if confirmed:
        print(f"\n  >>> CONFIRMED: Echo chambers show obsessive coherence")
        print(f"  >>> Higher Oh, eta, DEF; lower Xi than bridging structures")
    
    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "POL", "dataset": "Political Blogosphere 2004 (Adamic & Glance)",
        "method": "Adjacency matrix eigenstructure (SIG/v5.8c). "
                  "Static network: adjacency = correlation analog.",
        "conditions": results,
        "hypothesis": {
            "echo_vs_bridge": dirs,
            "n_coherent": n_coherent, "confirmed": confirmed,
            "oc_echo": round(oc_echo, 4), "oc_bridge": round(oc_bridge, 4),
        },
    }
    out_file = OUT_DIR / "pol_obsessive_coherence.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\n  Results: {out_file}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
