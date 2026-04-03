"""
STATE MAPPER - Katashi EEG State Classification
================================================
Maps each time point to a structural state:
- baseline: normal activity (>120s from seizures)
- pre_ictal: 30-120s before seizure onset
- ictal: during seizure (annotated intervals)
- pos_ictal: 0-120s after seizure end
- artifact: extreme Xi/Oh values outside seizures

Author: Claude + David
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Get script directory
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent / 'out_chbmit' / 'chb01'

# Seizure annotations from chb01-summary.txt
SEIZURES = [
    {'file': 'chb01_03', 'start': 2996, 'end': 3036},
    {'file': 'chb01_04', 'start': 1467, 'end': 1494},
    {'file': 'chb01_15', 'start': 1732, 'end': 1772},
    {'file': 'chb01_16', 'start': 1015, 'end': 1066},
    {'file': 'chb01_18', 'start': 1720, 'end': 1810},
    {'file': 'chb01_21', 'start': 327, 'end': 420},
    {'file': 'chb01_26', 'start': 1862, 'end': 1963},
]

def load_data():
    """Load Katashi state and features data"""
    print("Loading data...")
    
    # Load Katashi state
    state_path = BASE_DIR / 'kappa_run' / 'katashi_state.csv'
    state_df = pd.read_csv(state_path)
    print(f"  Loaded {len(state_df)} Katashi state points")
    
    # Load X_features for file/time mapping
    X_path = BASE_DIR / 'X_features.csv'
    X_df = pd.read_csv(X_path, usecols=['t', 'file', 'start_s', 'end_s'])
    print(f"  Loaded {len(X_df)} feature windows")
    
    # Merge
    df = state_df.merge(X_df, on='t', how='left')
    print(f"  Merged dataset: {len(df)} rows")
    
    return df

def map_states(df):
    """Map each time point to a structural state"""
    print("\nMapping states...")
    
    # Initialize state column
    df['state'] = 'baseline'
    df['seizure_id'] = -1
    
    # Iterate through seizures
    for idx, seizure in enumerate(SEIZURES):
        file_mask = df['file'] == seizure['file']
        
        # Ictal (during seizure)
        ictal_mask = file_mask & (df['start_s'] >= seizure['start']) & (df['end_s'] <= seizure['end'])
        df.loc[ictal_mask, 'state'] = 'ictal'
        df.loc[ictal_mask, 'seizure_id'] = idx
        
        # Pre-ictal (30-120s before)
        pre_start = max(0, seizure['start'] - 120)
        pre_end = seizure['start'] - 30
        pre_mask = file_mask & (df['start_s'] >= pre_start) & (df['end_s'] <= pre_end)
        df.loc[pre_mask, 'state'] = 'pre_ictal'
        df.loc[pre_mask, 'seizure_id'] = idx
        
        # Pos-ictal (0-120s after)
        pos_start = seizure['end']
        pos_end = seizure['end'] + 120
        pos_mask = file_mask & (df['start_s'] >= pos_start) & (df['end_s'] <= pos_end)
        df.loc[pos_mask, 'state'] = 'pos_ictal'
        df.loc[pos_mask, 'seizure_id'] = idx
    
    # Artifacts (extreme values, non-seizure)
    xi_threshold = df['Xi'].quantile(0.999)
    artifact_mask = (df['Xi'] > xi_threshold) & (df['state'] == 'baseline')
    df.loc[artifact_mask, 'state'] = 'artifact'
    
    # Print statistics
    print("\nState distribution:")
    print(df['state'].value_counts())
    print(f"\nSeizures mapped: {df['seizure_id'].nunique() - 1}")  # -1 for the -1 value
    
    return df

def add_temporal_features(df):
    """Add temporal derivatives and rolling statistics"""
    print("\nAdding temporal features...")
    
    # Sort by time
    df = df.sort_values('t').reset_index(drop=True)
    
    # Derivatives (rate of change)
    df['Xi_slope'] = df['Xi'].diff() / 2.0  # window is 2s step
    df['Oh_slope'] = df['Oh'].diff() / 2.0
    df['dominance_slope'] = df['dominance'].diff() / 2.0
    
    # Rolling statistics (10 windows = 20s)
    window = 10
    df['Xi_std_10'] = df['Xi'].rolling(window).std()
    df['Oh_std_10'] = df['Oh'].rolling(window).std()
    
    # Fill NaN from diff/rolling
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print(f"  Added {6} temporal features")
    
    return df

def save_mapped_data(df):
    """Save state-mapped dataset"""
    output_path = SCRIPT_DIR / 'states_mapped.csv'
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows with {len(df.columns)} columns")
    
    # Save summary
    summary_path = SCRIPT_DIR / 'states_mapped_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("STATE MAPPING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("State Distribution:\n")
        f.write(df['state'].value_counts().to_string())
        f.write("\n\n")
        
        f.write("State Statistics:\n")
        f.write("-" * 60 + "\n")
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            f.write(f"\n{state.upper()}:\n")
            f.write(f"  Count: {len(state_df)}\n")
            f.write(f"  Xi:   mean={state_df['Xi'].mean():.2f}, std={state_df['Xi'].std():.2f}\n")
            f.write(f"  Oh:   mean={state_df['Oh'].mean():.2f}, std={state_df['Oh'].std():.2f}\n")
            f.write(f"  Dom:  mean={state_df['dominance'].mean():.3f}, std={state_df['dominance'].std():.3f}\n")
    
    print(f"  Saved summary to {summary_path}")

def main():
    print("=" * 60)
    print("STATE MAPPER - Katashi EEG State Classification")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Map states
    df = map_states(df)
    
    # Add temporal features
    df = add_temporal_features(df)
    
    # Save
    save_mapped_data(df)
    
    print("\n" + "=" * 60)
    print("STATE MAPPING COMPLETE!")
    print("=" * 60)
    
    return df

if __name__ == '__main__':
    df = main()
