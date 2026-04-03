#!/usr/bin/env python3
"""
Example: OULAD Analysis with Kappa-Education

Run: python examples/example_oulad.py
Requires: pip install -e .
"""

from pathlib import Path
import pandas as pd
import numpy as np
from katashi_analyzer import KatashiAnalyzer, KatashiConfig


def main():
    print("\n" + "="*70)
    print("  KAPPA-EDUCATION: OULAD Analysis Example")
    print("="*70)
    
    # Look for data file
    root_dir = Path(__file__).parent.parent
    data_paths = [
        root_dir / 'data' / 'data_AAA_2014J_pass.csv',
        Path('data/data_AAA_2014J_pass.csv'),
        Path('../data/data_AAA_2014J_pass.csv'),
    ]
    
    data_file = None
    for path in data_paths:
        if path.exists():
            data_file = path
            break
    
    if not data_file:
        print("\n❌ Data file not found!")
        print("Expected: data/data_AAA_2014J_pass.csv")
        print("\nPlease download OULAD data - see data/README.md")
        return
    
    # Load data
    print(f"\n📂 Loading: {data_file}")
    df = pd.read_csv(data_file)
    
    # Extract features
    features = df[[c for c in df.columns if c.startswith('clicks_')]]
    print(f"   Features: {len(features.columns)} columns")
    print(f"   Weeks: {len(features)} rows")
    
    # Configure analyzer
    config = KatashiConfig(window=10, calm_length=12, k=4)
    analyzer = KatashiAnalyzer(config=config)
    
    # Run analysis
    print("\n⚙️  Running Katashi analysis...")
    state = analyzer.analyze(features)
    
    # Display results
    print("\n✅ Analysis complete!")
    print(f"\nStructural Observables:")
    print(f"  Ξ  (intensity): {state['Xi'].mean():.3f} ± {state['Xi'].std():.3f}")
    print(f"  Oh (regime):    {state['Oh'].mean():.3f} ± {state['Oh'].std():.3f}")  
    print(f"  Φ  (memory):    {state['phi'].mean():.3f} ± {state['phi'].std():.3f}")
    print(f"  η  (friction):  {state['eta'].mean():.3f} ± {state['eta'].std():.3f}")
    
    if 'entropy' in state.columns:
        print(f"\nSpectral Metrics:")
        print(f"  Entropy:    {state['entropy'].mean():.3f}")
        print(f"  Dominance:  {state['dominance'].mean():.3f}")
    
    # CALM period
    if analyzer.calm_info and analyzer.calm_info['best']:
        calm = analyzer.calm_info['best']
        print(f"\n🎯 CALM Period:")
        print(f"  Start: week {calm['start']}")
        print(f"  End:   week {calm['end']}")
        print(f"  Score: {calm['score']:.4f}")
    
    print("\n" + "="*70)
    
    return state, analyzer


if __name__ == '__main__':
    try:
        state, analyzer = main()
        print("\n✅ Success! Results available in 'state' DataFrame")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
