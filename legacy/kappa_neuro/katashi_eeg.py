"""
katashi_eeg.py - Wrapper para processar EEG com Katashi
========================================================

Importa e expõe função para rodar Katashi em X_features.csv
"""

import sys
import subprocess
from pathlib import Path

def run_katashi_on_X(X_path: str, out_dir: str, subject_name: str):
    """
    Roda katashi_run.py em um arquivo X_features.csv.
    
    Parameters:
    -----------
    X_path : str
        Caminho para X_features.csv
    out_dir : str
        Diretório de saída
    subject_name : str
        Nome do sujeito (ex: S001, chb01)
    """
    
    # Caminho para katashi_run.py
    katashi_script = Path(__file__).parent / "katashi_run.py"
    
    if not katashi_script.exists():
        raise FileNotFoundError(f"katashi_run.py não encontrado em {katashi_script}")
    
    # Validar inputs
    X_path_obj = Path(X_path)
    if not X_path_obj.exists():
        raise FileNotFoundError(f"X_features.csv não encontrado: {X_path}")
    
    # Criar output dir
    out_dir_obj = Path(out_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # Rodar como subprocess (mesmo padrão do s001_run_katashi.py)
    cmd = [
        "python",
        str(katashi_script),
        "--input", str(X_path_obj),
        "--out", str(out_dir_obj),
        "--window", "10",
        "--step", "1",
        "--k", "4",
        "--calm_length", "12",
        "--calm_step", "1",
        "--calm_top_n", "3",
        "--phi_gamma", "0.97",
        "--phi_delta", "0.08",
        "--theta_mode", "pq",
        "--theta_q", "0.95",
        "--persist_sens", "2",
        "--persist_confirm", "5"
    ]
    
    print(f"  [DEBUG] Comando Katashi: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [DEBUG] STDOUT: {result.stdout}")
        print(f"  [DEBUG] STDERR: {result.stderr}")
        raise RuntimeError(f"Katashi falhou: {result.stderr}")
    
    return result.stdout
