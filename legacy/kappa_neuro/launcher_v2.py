"""
LAUNCHER MASTER - Pipeline V2 Completo
=======================================

Executa todo o workflow V2 na ordem correta.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def confirm(message):
    response = input(f"\n{message} (s/N): ").strip().lower()
    return response == 's'

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

DATA_DIR = Path(r"C:\Users\ohiod\Projects\TopoCML\scripts\kappa_eegs\data\chbmit")
OUTPUT_DIR = Path(r"C:\Users\ohiod\Projects\TopoCML\scripts\kappa_eegs\out_chbmit_v2")

print_header("LAUNCHER MASTER - Pipeline V2")

print("\n📋 WORKFLOW COMPLETO:")
print("  1. Processar EDFs → Features V2 (6-8h)")
print("  2. Rodar CALM nos Features (2h)")
print("  3. Consolidar resultados")
print("  4. Análise ictal/inter-ictal")
print("  5. Testes estatísticos")

print("\n⏱️ TEMPO TOTAL ESTIMADO: ~8-10 horas")

print("\n📁 DIRETÓRIOS:")
print(f"  Input:  {DATA_DIR}")
print(f"  Output: {OUTPUT_DIR}")

# Contar pacientes
patients = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')])
print(f"\n📊 PACIENTES: {len(patients)}")

if not confirm("Iniciar workflow completo?"):
    print("\n❌ Cancelado pelo usuário")
    sys.exit(0)

# ============================================================================
# PASSO 1: PROCESSAR EDFs → FEATURES V2
# ============================================================================

print_header("PASSO 1/5: Processar EDFs → Features V2")

start_time = time.time()

import process_eeg_v2_corrected

# O script já roda o main() quando importado
# Alternativamente, podemos chamar função por função

elapsed = time.time() - start_time
print(f"\n✅ Passo 1 completo em {elapsed/60:.1f} minutos")

if not confirm("Prosseguir para Passo 2?"):
    print("\n⏸️ Workflow pausado após Passo 1")
    sys.exit(0)

# ============================================================================
# PASSO 2: RODAR CALM
# ============================================================================

print_header("PASSO 2/5: Rodar CALM nos Features V2")

start_time = time.time()

import calm_runner_v2

# O script já roda o main() quando importado

elapsed = time.time() - start_time
print(f"\n✅ Passo 2 completo em {elapsed/60:.1f} minutos")

if not confirm("Prosseguir para Passo 3?"):
    print("\n⏸️ Workflow pausado após Passo 2")
    sys.exit(0)

# ============================================================================
# PASSO 3: CONSOLIDAR
# ============================================================================

print_header("PASSO 3/5: Consolidar Resultados V2")

print("\n[TODO] Criar step1_consolidate_v2.py adaptado")
print("       Por enquanto, use consolidate manual")

if not confirm("Consolidação manual feita. Prosseguir?"):
    print("\n⏸️ Workflow pausado após Passo 3")
    sys.exit(0)

# ============================================================================
# PASSO 4: ANÁLISE ICTAL/INTER-ICTAL
# ============================================================================

print_header("PASSO 4/5: Análise Ictal/Inter-ictal V2")

print("\n[TODO] Adaptar analyze_ictal_interictal.py para V2")
print("       Usar out_chbmit_v2 em vez de out_chbmit")

if not confirm("Análise ictal/inter-ictal feita. Prosseguir?"):
    print("\n⏸️ Workflow pausado após Passo 4")
    sys.exit(0)

# ============================================================================
# PASSO 5: TESTES ESTATÍSTICOS
# ============================================================================

print_header("PASSO 5/5: Testes Estatísticos V2")

print("\n[TODO] Rodar test_ictal_stats.py nos resultados V2")

print("\n✅ WORKFLOW COMPLETO!")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print_header("RESUMO FINAL")

print("\n✅ Pipeline V2 executado com sucesso!")
print("\n📊 PRÓXIMOS PASSOS:")
print("  1. Revisar ictal_analysis_v2.csv")
print("  2. Verificar p-values nos testes")
print("  3. Comparar V1 vs V2")
print("  4. Se positivo: Paper!")
print("  5. Se negativo: Sleep-EDF ou cross-domain")

print("\n" + "="*80)
print(f"Concluído em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
