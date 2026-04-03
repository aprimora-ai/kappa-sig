"""
PASSO 1: CONSOLIDAÇÃO AUTOMÁTICA
=================================

Gera resumo estatístico por sujeito SEM olhar dados individuais.
Protocolo: estatística agregada ANTES de interpretação.

Output: consolidated_summary.csv com métricas estruturais por sujeito.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
OUT_DIR = Path(r"C:\Users\ohiod\Projects\TopoCML\scripts\kappa_eegs")

# Diretórios de dados
HEALTHY_DIR = OUT_DIR / "out_eegmmidb"
EPILEPTIC_DIR = OUT_DIR / "out_chbmit"

# Threshold Oh (definido no processamento original)
OH_THRESHOLD = 1.0

# ============================================================================
# FUNÇÕES DE CONSOLIDAÇÃO
# ============================================================================

def consolidate_subject(subject_dir: Path, subject_id: str, group: str):
    """
    Extrai métricas agregadas de um sujeito SEM olhar dados individuais.
    
    Métricas calculadas:
    - n_windows: Total de janelas
    - Oh_max, Oh_mean, Oh_p95: Estatísticas de Oh
    - pct_Oh_gt_threshold: % janelas com Oh > threshold
    - max_consecutive_Oh_gt_threshold: Maior sequência contínua
    - phi_max, phi_mean: Estatísticas de phi
    - phi_drift: Tendência temporal de phi (coef angular)
    """
    
    # Carregar dados
    katashi_file = subject_dir / "kappa_run" / "katashi_state.csv"
    
    if not katashi_file.exists():
        return None
    
    df = pd.read_csv(katashi_file)
    
    # Validar colunas essenciais
    required_cols = ['Xi', 'Oh', 'phi']
    if not all(col in df.columns for col in required_cols):
        print(f"  [WARNING] {subject_id}: Colunas faltando")
        return None
    
    # Métricas básicas
    n_windows = len(df)
    
    # Estatísticas Oh
    Oh_max = df['Oh'].max()
    Oh_mean = df['Oh'].mean()
    Oh_p95 = df['Oh'].quantile(0.95)
    
    # Excursões Oh > threshold
    Oh_gt = (df['Oh'] > OH_THRESHOLD).astype(int)
    pct_Oh_gt = (Oh_gt.sum() / n_windows) * 100
    
    # Sequências consecutivas Oh > threshold
    # Detectar runs (sequências contínuas de 1s)
    runs = []
    current_run = 0
    for val in Oh_gt:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:  # última run
        runs.append(current_run)
    
    max_consecutive_Oh_gt = max(runs) if runs else 0
    n_excursions = len(runs)
    
    # Estatísticas phi
    phi_max = df['phi'].max()
    phi_mean = df['phi'].mean()
    
    # Drift de phi (tendência temporal)
    # Coeficiente angular de regressão linear simples
    if n_windows > 1:
        t = np.arange(n_windows)
        phi_vals = df['phi'].values
        # y = a*t + b → a = cov(t,phi) / var(t)
        phi_drift = np.cov(t, phi_vals)[0, 1] / np.var(t) if np.var(t) > 0 else 0.0
    else:
        phi_drift = 0.0
    
    # Estatísticas Xi (adicional)
    Xi_max = df['Xi'].max()
    Xi_mean = df['Xi'].mean()
    
    return {
        'subject': subject_id,
        'group': group,
        'n_windows': n_windows,
        'Oh_max': Oh_max,
        'Oh_mean': Oh_mean,
        'Oh_p95': Oh_p95,
        'pct_Oh_gt_threshold': pct_Oh_gt,
        'max_consecutive_Oh_gt_threshold': max_consecutive_Oh_gt,
        'n_excursions': n_excursions,
        'phi_max': phi_max,
        'phi_mean': phi_mean,
        'phi_drift': phi_drift,
        'Xi_max': Xi_max,
        'Xi_mean': Xi_mean
    }

# ============================================================================
# PROCESSAMENTO
# ============================================================================

def main():
    print("="*80)
    print("PASSO 1: CONSOLIDAÇÃO AUTOMÁTICA")
    print("="*80)
    print("Protocolo: Estatística agregada ANTES de interpretação")
    print(f"Threshold Oh: {OH_THRESHOLD}")
    print("="*80)
    
    all_results = []
    
    # ========================================================================
    # SAUDÁVEIS (S001-S109)
    # ========================================================================
    print("\n[1/2] Consolidando SAUDÁVEIS (S001-S109)...")
    
    healthy_subjects = sorted([d for d in HEALTHY_DIR.iterdir() if d.is_dir() and d.name.startswith('S')])
    
    for subject_dir in tqdm(healthy_subjects, desc="Saudáveis"):
        subject_id = subject_dir.name
        result = consolidate_subject(subject_dir, subject_id, group='healthy')
        if result:
            all_results.append(result)
    
    print(f"  [OK] {len([r for r in all_results if r['group']=='healthy'])} saudaveis consolidados")
    
    # ========================================================================
    # EPILÉPTICOS (chb01)
    # ========================================================================
    print("\n[2/2] Consolidando EPILÉPTICOS (chb01)...")
    
    epileptic_subjects = sorted([d for d in EPILEPTIC_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')])
    
    for subject_dir in tqdm(epileptic_subjects, desc="Epilépticos"):
        subject_id = subject_dir.name
        result = consolidate_subject(subject_dir, subject_id, group='epileptic')
        if result:
            all_results.append(result)
    
    print(f"  [OK] {len([r for r in all_results if r['group']=='epileptic'])} epilepticos consolidados")
    
    # ========================================================================
    # SALVAR CONSOLIDADO
    # ========================================================================
    print("\n[SALVANDO] Consolidado...")
    
    df_consolidated = pd.DataFrame(all_results)
    
    # Ordenar: epilépticos primeiro, depois saudáveis
    df_consolidated = df_consolidated.sort_values(['group', 'subject'], ascending=[False, True])
    
    output_file = OUT_DIR / "consolidated_summary.csv"
    df_consolidated.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"  [OK] Salvo em: {output_file}")
    
    # ========================================================================
    # RESUMO FINAL (SEM INTERPRETAÇÃO)
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL (raw stats, zero interpretação)")
    print("="*80)
    
    healthy = df_consolidated[df_consolidated['group'] == 'healthy']
    epileptic = df_consolidated[df_consolidated['group'] == 'epileptic']
    
    print(f"\nSAUDÁVEIS (n={len(healthy)}):")
    print(f"  Oh_max:     [{healthy['Oh_max'].min():.4f}, {healthy['Oh_max'].max():.4f}]")
    print(f"  Oh_mean:    [{healthy['Oh_mean'].min():.4f}, {healthy['Oh_mean'].max():.4f}]")
    print(f"  pct_Oh>1:   [{healthy['pct_Oh_gt_threshold'].min():.2f}%, {healthy['pct_Oh_gt_threshold'].max():.2f}%]")
    print(f"  max_consec: [{healthy['max_consecutive_Oh_gt_threshold'].min()}, {healthy['max_consecutive_Oh_gt_threshold'].max()}]")
    
    print(f"\nEPILÉPTICOS (n={len(epileptic)}):")
    print(f"  Oh_max:     [{epileptic['Oh_max'].min():.4f}, {epileptic['Oh_max'].max():.4f}]")
    print(f"  Oh_mean:    [{epileptic['Oh_mean'].min():.4f}, {epileptic['Oh_mean'].max():.4f}]")
    print(f"  pct_Oh>1:   [{epileptic['pct_Oh_gt_threshold'].min():.2f}%, {epileptic['pct_Oh_gt_threshold'].max():.2f}%]")
    print(f"  max_consec: [{epileptic['max_consecutive_Oh_gt_threshold'].min()}, {epileptic['max_consecutive_Oh_gt_threshold'].max()}]")
    
    print("\n" + "="*80)
    print("PASSO 1 COMPLETO")
    print("Próximo: PASSO 2 (comparação populacional)")
    print("="*80)

if __name__ == "__main__":
    main()
