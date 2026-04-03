"""
4 MÉTRICAS OBJETIVAS - Lei de Ohio Refinada
============================================

Implementa as 4 métricas sem ajuste pós-hoc:

1. Distribuição de duração de excursões (CDF)
2. Índice de recorrência (R = #excursões / T_total)
3. Persistência máxima normalizada (L_max / T_total)
4. Memória condicional: Δφ | (Oh > 1, L ≥ L₀)

Para TODOS saudáveis (109) e epilépticos (3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Configuração
OUT_DIR = Path(r"C:\Users\ohiod\Projects\TopoCML\scripts\kappa_eegs")
HEALTHY_DIR = OUT_DIR / "out_eegmmidb"
EPILEPTIC_DIR = OUT_DIR / "out_chbmit"

L0_THRESHOLD = 100  # Limiar para "excursões longas" (100 janelas = 200s)

def compute_metrics(subject_dir: Path, subject_id: str, group: str):
    """Calcula as 4 métricas para um sujeito."""
    
    katashi_file = subject_dir / "kappa_run" / "katashi_state.csv"
    if not katashi_file.exists():
        return None
    
    df = pd.read_csv(katashi_file)
    T_total = len(df)  # janelas totais
    
    # Identificar excursões Oh > 1
    df['Oh_gt_1'] = df['Oh'] > 1.0
    df['run_id'] = (df['Oh_gt_1'] != df['Oh_gt_1'].shift()).cumsum()
    
    runs = df[df['Oh_gt_1']].groupby('run_id').agg({
        'Oh': ['max', 'mean', 'count'],
        'phi': ['max', 'mean', 'first', 'last']
    }).reset_index()
    
    runs.columns = ['run_id', 'Oh_max', 'Oh_mean', 'length', 
                    'phi_max', 'phi_mean', 'phi_start', 'phi_end']
    
    runs['phi_delta'] = runs['phi_end'] - runs['phi_start']
    
    if len(runs) == 0:
        return {
            'subject': subject_id,
            'group': group,
            'T_total': T_total,
            'n_excursions': 0,
            'R_recurrence': 0.0,
            'L_max': 0,
            'L_max_normalized': 0.0,
            'durations': [],
            'durations_p50': 0,
            'durations_p95': 0,
            'durations_p99': 0,
            'n_long_excursions': 0,
            'phi_delta_long_mean': np.nan,
            'phi_delta_long_median': np.nan,
            'phi_accumulation_rate': 0.0
        }
    
    # MÉTRICA 1: Distribuição de duração
    durations = runs['length'].values
    L_max = durations.max()
    
    # MÉTRICA 2: Índice de recorrência
    R = len(runs) / T_total
    
    # MÉTRICA 3: Persistência máxima normalizada
    L_max_normalized = L_max / T_total
    
    # MÉTRICA 4: Memória condicional (excursões longas)
    long_runs = runs[runs['length'] >= L0_THRESHOLD]
    n_long = len(long_runs)
    
    if n_long > 0:
        phi_delta_long_mean = long_runs['phi_delta'].mean()
        phi_delta_long_median = long_runs['phi_delta'].median()
        phi_accumulation_rate = (long_runs['phi_delta'] > 0.05).sum() / n_long
    else:
        phi_delta_long_mean = np.nan
        phi_delta_long_median = np.nan
        phi_accumulation_rate = 0.0
    
    return {
        'subject': subject_id,
        'group': group,
        'T_total': T_total,
        'n_excursions': len(runs),
        'R_recurrence': R,
        'L_max': int(L_max),
        'L_max_normalized': L_max_normalized,
        'durations': durations.tolist(),
        'durations_p50': float(np.percentile(durations, 50)),
        'durations_p95': float(np.percentile(durations, 95)),
        'durations_p99': float(np.percentile(durations, 99)),
        'n_long_excursions': n_long,
        'phi_delta_long_mean': phi_delta_long_mean,
        'phi_delta_long_median': phi_delta_long_median,
        'phi_accumulation_rate': phi_accumulation_rate
    }

# ============================================================================
# PROCESSAR TODOS
# ============================================================================

print("="*80)
print("4 MÉTRICAS OBJETIVAS - Lei de Ohio Refinada")
print("="*80)
print(f"Limiar para excursões longas: L0 = {L0_THRESHOLD} janelas (200s)")
print("="*80)

all_metrics = []

# SAUDÁVEIS
print("\n[1/2] Processando SAUDÁVEIS (S001-S109)...")
healthy_subjects = sorted([d for d in HEALTHY_DIR.iterdir() if d.is_dir() and d.name.startswith('S')])

for subject_dir in tqdm(healthy_subjects, desc="Saudáveis"):
    result = compute_metrics(subject_dir, subject_dir.name, 'healthy')
    if result:
        all_metrics.append(result)

print(f"  OK: {len([m for m in all_metrics if m['group']=='healthy'])} saudáveis processados")

# EPILÉPTICOS
print("\n[2/2] Processando EPILÉPTICOS (chb01-chb03)...")
epileptic_subjects = sorted([d for d in EPILEPTIC_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')])

for subject_dir in tqdm(epileptic_subjects, desc="Epilépticos"):
    result = compute_metrics(subject_dir, subject_dir.name, 'epileptic')
    if result:
        all_metrics.append(result)

print(f"  OK: {len([m for m in all_metrics if m['group']=='epileptic'])} epilépticos processados")

# ============================================================================
# SALVAR
# ============================================================================

output_file = OUT_DIR / "four_metrics.json"
with open(output_file, 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"\n[SALVO] {output_file}")

# ============================================================================
# ANÁLISE COMPARATIVA
# ============================================================================

df_metrics = pd.DataFrame(all_metrics)

healthy = df_metrics[df_metrics['group'] == 'healthy']
epileptic = df_metrics[df_metrics['group'] == 'epileptic']

print("\n" + "="*80)
print("ANÁLISE COMPARATIVA")
print("="*80)

print(f"\nSAUDAVEIS (n={len(healthy)}):")
print(f"  METRICA 1 - Duracao mediana: [{healthy['durations_p50'].min():.1f}, {healthy['durations_p50'].max():.1f}] janelas")
print(f"  METRICA 1 - Duracao P95: [{healthy['durations_p95'].min():.1f}, {healthy['durations_p95'].max():.1f}] janelas")
print(f"  METRICA 2 - Recorrencia: [{healthy['R_recurrence'].min():.6f}, {healthy['R_recurrence'].max():.6f}]")
print(f"  METRICA 3 - L_max norm: [{healthy['L_max_normalized'].min():.4f}, {healthy['L_max_normalized'].max():.4f}]")
print(f"  METRICA 4 - Delta-phi long (mean): [{healthy['phi_delta_long_mean'].min():.6f}, {healthy['phi_delta_long_mean'].max():.6f}]")
print(f"  METRICA 4 - Acumulo rate: [{healthy['phi_accumulation_rate'].min():.3f}, {healthy['phi_accumulation_rate'].max():.3f}]")

print(f"\nEPILEPTICOS (n={len(epileptic)}):")
print(f"  METRICA 1 - Duracao mediana: [{epileptic['durations_p50'].min():.1f}, {epileptic['durations_p50'].max():.1f}] janelas")
print(f"  METRICA 1 - Duracao P95: [{epileptic['durations_p95'].min():.1f}, {epileptic['durations_p95'].max():.1f}] janelas")
print(f"  METRICA 2 - Recorrencia: [{epileptic['R_recurrence'].min():.6f}, {epileptic['R_recurrence'].max():.6f}]")
print(f"  METRICA 3 - L_max norm: [{epileptic['L_max_normalized'].min():.4f}, {epileptic['L_max_normalized'].max():.4f}]")
print(f"  METRICA 4 - Delta-phi long (mean): [{epileptic['phi_delta_long_mean'].min():.6f}, {epileptic['phi_delta_long_mean'].max():.6f}]")
print(f"  METRICA 4 - Acumulo rate: [{epileptic['phi_accumulation_rate'].min():.3f}, {epileptic['phi_accumulation_rate'].max():.3f}]")

# Comparação específica
print("\n" + "="*80)
print("SEPARAÇÃO POR MÉTRICA")
print("="*80)

# MÉTRICA 2: Recorrência (chave!)
print(f"\nMETRICA 2 (Recorrencia):")
print(f"  Saudaveis medio: {healthy['R_recurrence'].mean():.6f}")
print(f"  Epilepticos medio: {epileptic['R_recurrence'].mean():.6f}")
print(f"  Ratio: {epileptic['R_recurrence'].mean() / healthy['R_recurrence'].mean():.1f}x")

# MÉTRICA 4: Memória condicional
print(f"\nMETRICA 4 (Memoria condicional em excursoes longas):")
healthy_with_long = healthy[healthy['n_long_excursions'] > 0]
epileptic_with_long = epileptic[epileptic['n_long_excursions'] > 0]

print(f"  Saudaveis com excursoes longas: {len(healthy_with_long)}/{len(healthy)} ({len(healthy_with_long)/len(healthy)*100:.1f}%)")
print(f"  Epilepticos com excursoes longas: {len(epileptic_with_long)}/{len(epileptic)} ({len(epileptic_with_long)/len(epileptic)*100:.1f}%)")

if len(healthy_with_long) > 0:
    print(f"  Delta-phi saudaveis (com longas): {healthy_with_long['phi_delta_long_mean'].mean():.6f}")
if len(epileptic_with_long) > 0:
    print(f"  Delta-phi epilepticos (com longas): {epileptic_with_long['phi_delta_long_mean'].mean():.6f}")

print("\n" + "="*80)
print("CONCLUSÃO")
print("="*80)

# Teste simples de separacao
R_threshold = healthy['R_recurrence'].quantile(0.95)
print(f"\nUsando R_threshold = P95 dos saudaveis = {R_threshold:.6f}:")
print(f"  Saudaveis acima: {(healthy['R_recurrence'] > R_threshold).sum()}/{len(healthy)} ({(healthy['R_recurrence'] > R_threshold).sum()/len(healthy)*100:.1f}%)")
print(f"  Epilepticos acima: {(epileptic['R_recurrence'] > R_threshold).sum()}/{len(epileptic)} ({(epileptic['R_recurrence'] > R_threshold).sum()/len(epileptic)*100:.1f}%)")

print("="*80)
