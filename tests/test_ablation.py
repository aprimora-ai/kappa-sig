"""
Kappa Analyst — Ablation Test
PERGUNTA: O campo HUGO produz aprendizado real, ou a melhoria
vem apenas das instrucoes de reforco injetadas no prompt?

TESTE: Rodar o Analista com o campo evoluido (step 178, 36 feedbacks)
mas com FIELD_THRESHOLD = -1.0 (desliga todos os reforcos).
Se a qualidade se mantem -> aprendizado real do campo.
Se os erros voltam -> era apenas o reforco textual.

David Ohio | odavidohio@gmail.com | Marco 2026
"""
import sys, os, json, io, re, copy
sys.path.insert(0, r"C:\Users\ohiod\Projects\Sentinel")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.kappa.analyst.agent import KappaAnalystAgent
from src.kappa.pair_analyzer import analyze_universe_pairs

SUMMARY_PATH = r"C:\Users\ohiod\Projects\Sentinel\data\reports\sentinel_summary.json"
CRITICAL = ["brazil_sectors", "x_brazil_vuln", "global_macro",
            "commodities", "financials", "x_global_contagion"]

print("=" * 70)
print("  ABLATION TEST: Campo HUGO vs Reforco Textual")
print("  Condicao: reforcos DESLIGADOS (threshold = -1)")
print("  David Ohio | odavidohio@gmail.com")
print("=" * 70)

# Pair data
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
print(f"\nCampo carregado: step={agent.field.step}")
H = list(agent.field.H)
dims = ['identity', 'pairs', 'roles', 'cross', 'triggers']
print(f"H = {[f'{h:.3f}' for h in H]}")
for i, d in enumerate(dims):
    below = H[i] < 0.4
    print(f"  H[{i}] {d}: {H[i]:.3f} {'<-- REFORCO NORMALMENTE ATIVO' if below else ''}")

# ABLATION: Desligar TODOS os reforcos
print(f"\n>>> DESLIGANDO REFORCOS (threshold -1.0) <<<")
original_threshold = agent.FIELD_THRESHOLD
agent.FIELD_THRESHOLD = -1.0  # nenhum H sera < -1, nenhum reforco ativa

# Rodar analises SEM reforco
print(f"\nGerando briefings sem reforco...")
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

    # Get auto-eval score
    fbs = [t for t in agent.training_log if t.get("type") == "analysis_feedback"]
    if fbs:
        last = fbs[-1]
        ablation_scores.append({"uid": uid, "scores": last["scores"], "overall": last["overall"]})
        print(f"score={last['overall']:.2f}")
    else:
        print("OK")

# Restaurar threshold
agent.FIELD_THRESHOLD = original_threshold

# Quality checks
print(f"\n{'='*70}")
print(f"  RESULTADOS DA ABLACAO")
print(f"{'='*70}")

checks = {
    'brazil_sectors': {
        'nomes_errados': ['Eaton Vance', 'Aggregate Bond'],
        'erro_conceitual': [r'refúgio.*USO', r'USO.*refúgio'],
    },
    'x_brazil_vuln': {
        'erro_conceitual': [r'USO.*refúgio', r'petróleo.*refúgio'],
    },
    'financials': {
        'nomes_errados': ['Eaton Vance', 'Aggregate Bond', 'Crude Oil ETN'],
    },
}

total_halluc = 0
total_concept_err = 0
total_cross = 0
total_triggers = 0

for uid, brief in ablation_briefings.items():
    ch = checks.get(uid, {})
    halluc = 0
    concept = 0

    for bad in ch.get('nomes_errados', []):
        if bad.lower() in brief.lower():
            halluc += 1
    for err in ch.get('erro_conceitual', []):
        if re.search(err, brief, re.IGNORECASE):
            concept += 1

    other_u = ['x_brazil_vuln', 'brazil_sectors', 'global_macro',
               'commodities', 'x_energy_geopolitics']
    cross = sum(1 for u in other_u if u != uid and u in brief)

    trigger_p = [r'Oh\s*[><=]\s*[\d.]', r'[ntv]_[sK]\s*[><=]',
                 r'PRESSURIZED', r'threshold']
    triggers = sum(1 for p in trigger_p if re.search(p, brief))

    total_halluc += halluc
    total_concept_err += concept
    total_cross += (1 if cross > 0 else 0)
    total_triggers += (1 if triggers >= 2 else 0)

    status = "OK" if halluc == 0 and concept == 0 else "FALHA"
    print(f"  {uid}: halluc={halluc} concept_err={concept} "
          f"cross_ref={'SIM' if cross else 'NAO'} triggers={triggers} [{status}]")

# Comparacao com resultados anteriores
print(f"\n{'='*70}")
print(f"  COMPARACAO: COM REFORCO vs SEM REFORCO (ABLACAO)")
print(f"{'='*70}")
print(f"                          | v1.1(S1) | S3+reforco | ABLACAO(S3-reforco)")
print(f"  Alucinacoes de nomes    |    5     |     0      |     {total_halluc}")
print(f"  Erros conceituais       |    2     |     0      |     {total_concept_err}")
print(f"  Cross-universo (de 6)   |    0     |     1      |     {total_cross}")
print(f"  Gatilhos numericos (6)  |    0     |     1      |     {total_triggers}")

# Scores
if ablation_scores:
    mean_score = sum(s["overall"] for s in ablation_scores) / len(ablation_scores)
    print(f"\n  Score medio ablacao: {mean_score:.3f}")
    print(f"  Score medio S3+reforco (ciclo 4): 0.717")
    print(f"  Score medio S3+reforco (ciclo 5): 0.577")

# VEREDITO
print(f"\n{'='*70}")
if total_halluc == 0 and total_concept_err == 0:
    print("  VEREDITO: APRENDIZADO REAL CONFIRMADO")
    print("  As melhorias persistem SEM as instrucoes de reforco.")
    print("  O campo HUGO tem efeito causal na qualidade do output.")
    print("  Isso NAO e apenas um wrapper de LLM.")
elif total_halluc <= 1 and total_concept_err <= 1:
    print("  VEREDITO: APRENDIZADO PARCIAL")
    print("  A maioria das melhorias persiste, mas algumas regressam")
    print("  sem reforco. O campo contribui, mas nao e suficiente sozinho.")
else:
    print("  VEREDITO: REFORCO TEXTUAL DOMINANTE")
    print("  Os erros voltaram sem as instrucoes de reforco.")
    print("  O campo HUGO nao tem efeito causal suficiente.")
    print("  A melhoria observada era primariamente do prompt engineering.")
print(f"{'='*70}")

# Save results
results = {
    "test": "ablation",
    "field_step": agent.field.step,
    "H_at_test": list(agent.field.H),
    "reinforcements_disabled": True,
    "total_hallucinations": total_halluc,
    "total_concept_errors": total_concept_err,
    "total_cross_refs": total_cross,
    "total_trigger_quality": total_triggers,
    "per_universe": ablation_scores,
    "mean_score": mean_score if ablation_scores else 0,
}
abl_path = r"C:\Users\ohiod\Projects\Sentinel\data\reports\ablation_results.json"
with open(abl_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {abl_path}")
