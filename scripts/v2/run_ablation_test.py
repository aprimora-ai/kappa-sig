"""
Kappa Analyst — S3 Ablation Test
=================================
Teste decisivo: roda o Analista com campo evoluído (step 178)
MAS com reforços textuais DESABILITADOS.

Se a melhoria persiste → campo tem efeito causal real
Se o erro volta → era apenas o reforço textual fazendo o trabalho

David Ohio | odavidohio@gmail.com | Março 2026
"""
import sys, os, json, io, re
sys.path.insert(0, r"C:\Users\ohiod\Projects\Sentinel")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SUMMARY_PATH = r"C:\Users\ohiod\Projects\Sentinel\data\reports\sentinel_summary.json"

from src.kappa.pair_analyzer import analyze_universe_pairs
from src.kappa.analyst.agent import KappaAnalystAgent

CRITICAL = ["brazil_sectors", "x_brazil_vuln", "global_macro",
            "commodities", "financials", "x_global_contagion"]

print("=" * 70)
print("  KAPPA ANALYST — ABLATION TEST")
print("  Campo evoluído (step 178) + Reforços DESABILITADOS")
print("  Se melhoria persiste = aprendizado real do campo")
print("  Se erro volta = era apenas instrução textual")
print("=" * 70)

# Load pair data
print("\n[PREP] Pair analysis...")
pair_results = {}
for uid in CRITICAL:
    result = analyze_universe_pairs(uid, top_n=5)
    if "error" not in result:
        pair_results[uid] = result

# Load summary
with open(SUMMARY_PATH, "r") as f:
    raw = f.read().replace("NaN", "null")
    summary = json.loads(raw)
report_map = {r["universe"]: r for r in summary["reports"] if not r.get("error")}

# Load agent with evolved field
agent = KappaAnalystAgent(mind_id="kappa_analyst_v1.1")
print(f"\n  Mind loaded: step {agent.field.step}")
print(f"  H = {[f'{h:.3f}' for h in agent.field.H]}")

# ═══════════════════════════════════════════════════════════════
# ABLATION: Disable reinforcement injections
# Set threshold to 0.0 so NO H[i] triggers reinforcement
# The field state is preserved — only the text injections are removed
# ═══════════════════════════════════════════════════════════════
original_threshold = agent.FIELD_THRESHOLD
agent.FIELD_THRESHOLD = 0.0  # No reinforcements will fire

# Also record which reinforcements WOULD have fired
would_fire = []
for i, dim in agent.FIELD_DIMS.items():
    if agent.field.H[i] < original_threshold:
        would_fire.append(f"  H[{i}] {dim} = {agent.field.H[i]:.3f} < {original_threshold} (WOULD reinforce)")
    else:
        would_fire.append(f"  H[{i}] {dim} = {agent.field.H[i]:.3f} >= {original_threshold} (no reinforce)")

print(f"\n  Reinforcement threshold: {original_threshold} → 0.0 (DISABLED)")
print(f"  Reinforcements that WOULD have fired:")
for w in would_fire:
    print(w)

# ═══════════════════════════════════════════════════════════════
# RUN ABLATION ANALYSES
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print(f"  RUNNING ANALYSES (no reinforcements)")
print(f"{'─'*70}")

ablation_briefings = {}
ablation_scores = []

for uid in CRITICAL:
    report = report_map.get(uid)
    if not report:
        continue
    data = {
        "status": report.get("status", "?"),
        "oh_max": report.get("oh_max", 0),
        "nu_s": report.get("nu_s", 0),
        "tau_k_max": report.get("tau_k_max", 0),
        "regime": report.get("regime", "?"),
    }
    pr = pair_results.get(uid, {})
    if pr.get("top_pairs"):
        data["pair_data"] = pr["top_pairs"][:5]
    
    print(f"  {uid}...", end=" ", flush=True)
    analysis = agent.analyze(uid, data)
    ablation_briefings[uid] = analysis
    
    # Get feedback score
    latest = [t for t in agent.training_log if t.get("type") == "analysis_feedback"]
    if latest:
        last = latest[-1]
        ablation_scores.append({"uid": uid, "overall": last["overall"], "scores": last["scores"]})
        print(f"score={last['overall']:.2f}")
    else:
        print("OK")

# ═══════════════════════════════════════════════════════════════
# QUALITY CHECKS — same as _quality_check.py
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ABLATION RESULTS")
print(f"{'='*70}")

hallucination_names = ['Eaton Vance', 'Aggregate Bond', 'Crude Oil ETN']
uso_refugio_pattern = re.compile(r'USO.*ref[uú]gio|petr[oó]leo.*ref[uú]gio', re.IGNORECASE)
delta_pct_pattern = re.compile(r'desacelera[çc][ãa]o de \d+[\.,]\d+%', re.IGNORECASE)

n_hallucinations = 0
n_uso_refugio = 0
n_delta_pct = 0
n_cross_ref = 0
other_universes = ['x_brazil_vuln', 'brazil_sectors', 'global_macro', 
                   'commodities', 'x_energy_geopolitics', 'x_global_contagion']

for uid, brief in ablation_briefings.items():
    print(f"\n  [{uid}]")
    
    # Check hallucinations
    for bad in hallucination_names:
        if bad.lower() in brief.lower():
            n_hallucinations += 1
            print(f"    ❌ ALUCINAÇÃO: '{bad}'")
    
    # Check USO = refugio
    if uso_refugio_pattern.search(brief):
        n_uso_refugio += 1
        print(f"    ❌ USO = refúgio")
    else:
        print(f"    ✅ USO ≠ refúgio")
    
    # Check delta as %
    if delta_pct_pattern.search(brief):
        n_delta_pct += 1
        print(f"    ❌ Delta como %")
    else:
        print(f"    ✅ Delta correto")
    
    # Check cross-universe
    cross = [u for u in other_universes if u != uid and u in brief]
    if cross:
        n_cross_ref += 1
        print(f"    ✅ Cross-ref: {cross}")
    else:
        print(f"    ⚠️ Sem cross-ref")

# ═══════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════
mean_score = sum(s["overall"] for s in ablation_scores) / len(ablation_scores) if ablation_scores else 0

print(f"\n{'='*70}")
print(f"  ABLATION VERDICT")
print(f"{'='*70}")
print(f"  Alucinações:     {n_hallucinations} (v1.1=5, S3=0)")
print(f"  USO=refúgio:     {n_uso_refugio}/6 (v1.1=2/6, S3-36fb=0/6)")
print(f"  Delta como %:    {n_delta_pct}/6 (v1.1=sim, S3=0)")
print(f"  Cross-universo:  {n_cross_ref}/6")
print(f"  Score médio:     {mean_score:.3f} (S3-36fb=0.577)")
print(f"")

if n_hallucinations == 0 and n_uso_refugio == 0 and n_delta_pct == 0:
    print(f"  ✅ APRENDIZADO REAL CONFIRMADO")
    print(f"     Sem reforços textuais, os erros NÃO voltaram.")
    print(f"     O campo HUGO tem efeito causal na qualidade do output.")
elif n_uso_refugio > 0 or n_hallucinations > 0:
    regressed = []
    if n_hallucinations > 0: regressed.append(f"alucinações ({n_hallucinations})")
    if n_uso_refugio > 0: regressed.append(f"USO=refúgio ({n_uso_refugio}/6)")
    if n_delta_pct > 0: regressed.append(f"delta% ({n_delta_pct})")
    print(f"  ⚠️ REGRESSÃO PARCIAL DETECTADA")
    print(f"     Erros que voltaram: {', '.join(regressed)}")
    print(f"     Conclusão: parte da melhoria era do reforço textual,")
    print(f"     parte era do campo (os erros que NÃO voltaram).")
else:
    print(f"  ✅ MELHORIAS CONSOLIDADAS (sem regressão nos erros críticos)")

print(f"\n  H final: {[f'{h:.3f}' for h in agent.field.H]}")
print(f"  Step: {agent.field.step}")
print(f"{'='*70}")

# Restore threshold (don't save this ablated state)
agent.FIELD_THRESHOLD = original_threshold
