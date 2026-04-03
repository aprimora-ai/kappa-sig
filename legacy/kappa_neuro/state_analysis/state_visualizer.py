"""
STATE VISUALIZER - Katashi EEG State Space Visualization
==========================================================
Visualizes the structural state space in multiple projections:
1. 3D scatter plot (Xi, Oh, Dominance)
2. 2D PCA projection
3. Pairwise feature distributions
4. Temporal trajectories around seizures

Author: Claude + David
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path

# Get script directory
SCRIPT_DIR = Path(__file__).parent

# State colors
STATE_COLORS = {
    'baseline': 'gray',
    'pre_ictal': 'orange',
    'ictal': 'red',
    'pos_ictal': 'purple',
    'artifact': 'blue'
}

def load_mapped_data():
    """Load state-mapped dataset"""
    print("Loading mapped data...")
    path = SCRIPT_DIR / 'states_mapped.csv'
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} points with {df['state'].nunique()} states")
    return df

def plot_3d_state_space(df):
    """3D scatter plot of state space"""
    print("\nCreating 3D state space plot...")
    output_path = SCRIPT_DIR / 'fig_3d_state_space.png'
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each state
    for state in ['baseline', 'pre_ictal', 'ictal', 'pos_ictal', 'artifact']:
        if state not in df['state'].values:
            continue
            
        state_df = df[df['state'] == state]
        
        # Sample baseline for visibility (too many points)
        if state == 'baseline' and len(state_df) > 5000:
            state_df = state_df.sample(5000, random_state=42)
        
        alpha = 0.05 if state == 'baseline' else 0.7
        size = 1 if state == 'baseline' else 30
        
        ax.scatter(state_df['Xi'], state_df['Oh'], state_df['dominance'],
                  c=STATE_COLORS[state], alpha=alpha, s=size, label=state)
    
    ax.set_xlabel('Xi (Complexity)', fontsize=12)
    ax.set_ylabel('Oh (Cohesion)', fontsize=12)
    ax.set_zlabel('Dominance', fontsize=12)
    ax.set_title('Katashi State Space - EEG Epileptic States', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def plot_pca_projection(df):
    """2D PCA projection of state space"""
    print("\nCreating PCA projection...")
    output_path = SCRIPT_DIR / 'fig_pca_projection.png'
    
    # Features for PCA
    features = ['Xi', 'Oh', 'dominance', 'entropy', 'phi']
    X = df[features].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for state in ['baseline', 'pre_ictal', 'ictal', 'pos_ictal', 'artifact']:
        if state not in df['state'].values:
            continue
            
        mask = df['state'] == state
        state_pca = X_pca[mask]
        
        # Sample baseline
        if state == 'baseline' and len(state_pca) > 5000:
            indices = np.random.choice(len(state_pca), 5000, replace=False)
            state_pca = state_pca[indices]
        
        alpha = 0.05 if state == 'baseline' else 0.6
        size = 5 if state == 'baseline' else 50
        
        ax.scatter(state_pca[:, 0], state_pca[:, 1],
                  c=STATE_COLORS[state], alpha=alpha, s=size, label=state)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA Projection of Katashi State Space', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()
    
    # Print explained variance
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%} variance")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%} variance")
    print(f"  Total: {pca.explained_variance_ratio_[:2].sum():.1%} variance")

def plot_pairwise_distributions(df):
    """Pairwise feature distributions by state"""
    print("\nCreating pairwise distributions...")
    output_path = SCRIPT_DIR / 'fig_pairwise.png'
    
    # Select features
    features = ['Xi', 'Oh', 'dominance', 'entropy']
    
    # Prepare data
    plot_df = df[features + ['state']].copy()
    
    # Sample baseline for visibility
    baseline_mask = plot_df['state'] == 'baseline'
    baseline_sample = plot_df[baseline_mask].sample(min(5000, baseline_mask.sum()), random_state=42)
    non_baseline = plot_df[~baseline_mask]
    plot_df = pd.concat([baseline_sample, non_baseline])
    
    # Create pairplot
    g = sns.pairplot(plot_df, hue='state', palette=STATE_COLORS,
                     diag_kind='kde', plot_kws={'alpha': 0.4, 's': 10},
                     corner=False)
    g.fig.suptitle('Pairwise Feature Distributions by State', y=1.01, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def plot_seizure_trajectories(df):
    """Plot temporal trajectories around each seizure"""
    print("\nCreating seizure trajectories...")
    output_path = SCRIPT_DIR / 'fig_seizure_trajectories.png'
    
    # Get seizures
    seizure_ids = df[df['state'] == 'ictal']['seizure_id'].unique()
    n_seizures = len(seizure_ids)
    
    if n_seizures == 0:
        print("  No seizures found, skipping trajectories")
        return
    
    # Create subplots (3 rows: Xi, Oh, Dominance)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for seizure_id in seizure_ids:
        # Get window around seizure (±300s = ±150 windows)
        seizure_df = df[df['seizure_id'] == seizure_id]
        if len(seizure_df) == 0:
            continue
        
        # Find seizure center
        center_idx = int(np.mean(seizure_df.index.values))
        window_start = max(0, center_idx - 150)
        window_end = min(len(df), center_idx + 150)
        
        window_df = df.iloc[window_start:window_end].copy()
        
        # Relative time (seconds from seizure onset)
        seizure_start_idx = seizure_df.index.min()
        window_df['rel_time'] = (window_df.index - seizure_start_idx) * 2  # 2s step
        
        # Plot Xi
        axes[0].plot(window_df['rel_time'], window_df['Xi'], 
                     alpha=0.5, linewidth=1, label=f'Seizure {seizure_id+1}')
        
        # Plot Oh
        axes[1].plot(window_df['rel_time'], window_df['Oh'], 
                     alpha=0.5, linewidth=1)
        
        # Plot Dominance
        axes[2].plot(window_df['rel_time'], window_df['dominance'], 
                     alpha=0.5, linewidth=1)
        
        # Mark seizure period
        ictal_window = window_df[window_df['state'] == 'ictal']
        if len(ictal_window) > 0:
            t_start = ictal_window['rel_time'].min()
            t_end = ictal_window['rel_time'].max()
            for ax in axes:
                ax.axvspan(t_start, t_end, alpha=0.2, color='red')
    
    # Format axes
    axes[0].set_ylabel('Xi', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Oh', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Dominance', fontsize=11)
    axes[2].set_xlabel('Time from Seizure Onset (seconds)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Temporal Trajectories Around Seizures', fontsize=14, y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def plot_state_statistics(df):
    """Box plots of key metrics by state"""
    print("\nCreating state statistics plots...")
    output_path = SCRIPT_DIR / 'fig_state_stats.png'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Xi
    df.boxplot(column='Xi', by='state', ax=axes[0, 0], patch_artist=True)
    axes[0, 0].set_title('Xi by State')
    axes[0, 0].set_xlabel('')
    
    # Oh
    df.boxplot(column='Oh', by='state', ax=axes[0, 1], patch_artist=True)
    axes[0, 1].set_title('Oh by State')
    axes[0, 1].set_xlabel('')
    
    # Dominance
    df.boxplot(column='dominance', by='state', ax=axes[1, 0], patch_artist=True)
    axes[1, 0].set_title('Dominance by State')
    axes[1, 0].set_xlabel('State')
    
    # Entropy
    df.boxplot(column='entropy', by='state', ax=axes[1, 1], patch_artist=True)
    axes[1, 1].set_title('Entropy by State')
    axes[1, 1].set_xlabel('State')
    
    plt.suptitle('Distribution of Katashi Metrics by State', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("STATE VISUALIZER - Katashi EEG State Space")
    print("=" * 60)
    
    # Load data
    df = load_mapped_data()
    
    # Create visualizations
    plot_3d_state_space(df)
    plot_pca_projection(df)
    plot_state_statistics(df)
    plot_pairwise_distributions(df)
    plot_seizure_trajectories(df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated 5 figures in state_analysis/")

if __name__ == '__main__':
    main()
