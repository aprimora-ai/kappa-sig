"""
Visualization - Plotting functions for Katashi analysis.

Implements all figures from the educational papers:
- Figure 1: Regime evolution over time
- Figure 2: Distribution boxplots
- Figure 3: Phase space diagrams
- Figure 4: Temporal dynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Any


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


def plot_regime_evolution(
    state: pd.DataFrame,
    calm_info: Dict[str, Any],
    thresholds: Dict[str, float],
    crossings: Dict[str, Dict],
    figsize: tuple = (14, 10)
):
    """
    Plot regime evolution with CALM period and threshold crossings.
    
    Creates 3-panel plot:
    - Top: Oh (regime number) evolution
    - Middle: Φ (structural memory) evolution
    - Bottom: Ξ (structural intensity) evolution
    
    Args:
        state: DataFrame with structural observables
        calm_info: CALM period information
        thresholds: Calibrated thresholds
        crossings: Detected crossings
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Extract CALM period
    best_calm = calm_info['best']
    if best_calm:
        calm_start = best_calm['start']
        calm_end = best_calm['end']
    else:
        calm_start = calm_end = None
    
    # Plot 1: Oh evolution
    axes[0].plot(state.index, state['Oh'], marker='o', linewidth=2,
                label='Oh', color='#1976D2', markersize=5)
    
    # Reference lines
    axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='Oh = 1.0 (Critical)')
    axes[0].axhline(thresholds['theta'], color='orange', linestyle=':',
                   linewidth=2, alpha=0.7, label=f'θ = {thresholds["theta"]:.3f}')
    
    # CALM shading
    if calm_start and calm_end:
        axes[0].axvspan(calm_start, calm_end, alpha=0.15, color='green',
                       label='CALM period')
    
    # Mark crossings
    for label, crossing in crossings.items():
        if crossing.get('sens'):
            axes[0].scatter(crossing['sens'], state.loc[crossing['sens'], 'Oh'],
                          s=150, marker='v', color='orange', edgecolors='black',
                          linewidths=2, zorder=5)
        if crossing.get('confirm'):
            axes[0].scatter(crossing['confirm'], state.loc[crossing['confirm'], 'Oh'],
                          s=200, marker='*', color='red', edgecolors='black',
                          linewidths=2, zorder=5)
    
    axes[0].set_ylabel('Oh (Regime Number)', fontweight='bold', fontsize=12)
    axes[0].set_title('Regime Evolution (Oh)', fontweight='bold', fontsize=13)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Φ evolution
    axes[1].plot(state.index, state['phi'], marker='o', linewidth=2,
                label='Φ (phi)', color='#D32F2F', markersize=5)
    
    axes[1].axhline(thresholds['phi_c'], color='purple', linestyle=':',
                   linewidth=2, alpha=0.7, label=f'Φ_c = {thresholds["phi_c"]:.3f}')
    
    if calm_start and calm_end:
        axes[1].axvspan(calm_start, calm_end, alpha=0.15, color='green')
    
    # Mark phi crossings
    phi_crossing = crossings.get('phi>phi_c', {})
    if phi_crossing.get('sens'):
        axes[1].scatter(phi_crossing['sens'], state.loc[phi_crossing['sens'], 'phi'],
                       s=150, marker='v', color='orange', edgecolors='black',
                       linewidths=2, zorder=5)
    if phi_crossing.get('confirm'):
        axes[1].scatter(phi_crossing['confirm'], state.loc[phi_crossing['confirm'], 'phi'],
                       s=200, marker='*', color='red', edgecolors='black',
                       linewidths=2, zorder=5)
    
    axes[1].set_ylabel('Φ (Structural Memory)', fontweight='bold', fontsize=12)
    axes[1].set_title('Structural Memory (Φ)', fontweight='bold', fontsize=13)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Ξ evolution
    axes[2].plot(state.index, state['Xi'], marker='o', linewidth=2,
                label='Ξ (Xi)', color='#388E3C', markersize=5)
    
    axes[2].axhline(thresholds['Xi_c'], color='orange', linestyle=':',
                   linewidth=2, alpha=0.7, label=f'Ξ_c = {thresholds["Xi_c"]:.3f}')
    
    if calm_start and calm_end:
        axes[2].axvspan(calm_start, calm_end, alpha=0.15, color='green')
    
    axes[2].set_xlabel('Date', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Ξ (Structural Intensity)', fontweight='bold', fontsize=12)
    axes[2].set_title('Structural Intensity (Ξ)', fontweight='bold', fontsize=13)
    axes[2].legend(loc='best', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_cohort_comparison(
    cohorts: Dict[str, pd.DataFrame],
    metric: str = 'Oh',
    figsize: tuple = (12, 6)
):
    """
    Compare metric distributions across cohorts (boxplot).
    
    Reproduces Figure 2 from the paper.
    
    Args:
        cohorts: Dictionary of {cohort_name: state_dataframe}
        metric: Metric to compare ('Oh', 'Xi', 'phi')
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot
    data = []
    labels = []
    
    for cohort_name, state in cohorts.items():
        data.append(state[metric].values)
        labels.append(cohort_name)
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    notch=True, showmeans=True)
    
    # Color boxes
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference line for Oh
    if metric == 'Oh':
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Oh = 1.0 (Critical)')
        ax.legend()
    
    ax.set_ylabel(f'{metric} Distribution', fontweight='bold', fontsize=12)
    ax.set_title(f'{metric} Distribution by Cohort', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_phase_space(
    cohorts: Dict[str, pd.DataFrame],
    figsize: tuple = (10, 8)
):
    """
    Plot phase space diagram (Ξ vs Oh).
    
    Reproduces Figure 3 from the paper - Educational regime map.
    
    Args:
        cohorts: Dictionary of {cohort_name: state_dataframe}
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'Pass': '#2196F3', 'Fail': '#FF9800', 
              'Distinction': '#4CAF50', 'Withdrawn': '#F44336'}
    
    for cohort_name, state in cohorts.items():
        color = colors.get(cohort_name, '#666666')
        ax.scatter(state['Xi'], state['Oh'], 
                  label=cohort_name, color=color, alpha=0.6, s=50)
    
    # Add reference lines
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
              alpha=0.5, label='Oh = 1.0')
    
    ax.set_xlabel('Ξ (Structural Intensity)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Oh (Regime Number)', fontweight='bold', fontsize=12)
    ax.set_title('Educational Regime Map (Ξ vs Oh)', fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_temporal_dynamics(
    cohorts: Dict[str, pd.DataFrame],
    figsize: tuple = (14, 8)
):
    """
    Plot temporal dynamics of Oh for all cohorts.
    
    Reproduces Figure 4 from the paper - Persistence of structural regimes.
    
    Args:
        cohorts: Dictionary of {cohort_name: state_dataframe}
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'Pass': '#2196F3', 'Fail': '#FF9800', 
              'Distinction': '#4CAF50', 'Withdrawn': '#F44336'}
    
    for cohort_name, state in cohorts.items():
        color = colors.get(cohort_name, '#666666')
        ax.plot(state.index, state['Oh'], marker='o', linewidth=2,
               label=cohort_name, color=color, markersize=4, alpha=0.8)
    
    # Reference line
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
              alpha=0.5, label='Oh = 1.0 (Critical)')
    
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('Oh (Regime Number)', fontweight='bold', fontsize=12)
    ax.set_title('Temporal Dynamics: Persistence of Structural Regimes',
                fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_lead_time_analysis(
    lead_times: Dict[str, Dict[str, float]],
    figsize: tuple = (10, 6)
):
    """
    Plot lead time comparison across cohorts.
    
    Args:
        lead_times: Dictionary of {cohort: lead_time_metrics}
        figsize: Figure size
    """
    cohorts = list(lead_times.keys())
    dt_total = [lead_times[c]['Δt_total'] for c in cohorts]
    dt_cascade = [lead_times[c]['Δt_cascade'] for c in cohorts]
    
    x = np.arange(len(cohorts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width/2, dt_total, width, label='Δt_total', 
           color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, dt_cascade, width, label='Δt_cascade', 
           color='#FF9800', alpha=0.8)
    
    ax.set_xlabel('Cohort', fontweight='bold', fontsize=12)
    ax.set_ylabel('Lead Time (days)', fontweight='bold', fontsize=12)
    ax.set_title('Lead Time Analysis by Cohort', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_summary_dashboard(
    state: pd.DataFrame,
    calm_info: Dict,
    thresholds: Dict,
    crossings: Dict,
    cohort_name: str = "Cohort"
):
    """
    Create comprehensive summary dashboard.
    
    Args:
        state: State DataFrame
        calm_info: CALM information
        thresholds: Thresholds
        crossings: Crossings
        cohort_name: Name of cohort
    """
    # Check if DEF is present
    has_def = 'DEF' in state.columns
    
    if has_def:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Extract CALM
    best_calm = calm_info['best']
    calm_start = best_calm['start'] if best_calm else None
    calm_end = best_calm['end'] if best_calm else None
    
    # Plot 1: Oh evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(state.index, state['Oh'], marker='o', linewidth=2, color='#1976D2')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(thresholds['theta'], color='orange', linestyle=':', linewidth=2, alpha=0.7)
    if calm_start and calm_end:
        ax1.axvspan(calm_start, calm_end, alpha=0.15, color='green')
    ax1.set_title(f'{cohort_name} - Oh Evolution', fontweight='bold')
    ax1.set_ylabel('Oh')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ξ vs Oh (phase space)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(state['Xi'], state['Oh'], alpha=0.6, s=50, color='#2196F3')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ξ (Structural Intensity)')
    ax2.set_ylabel('Oh (Regime Number)')
    ax2.set_title('Phase Space', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Φ evolution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(state.index, state['phi'], marker='o', linewidth=2, color='#D32F2F')
    ax3.axhline(thresholds['phi_c'], color='purple', linestyle=':', linewidth=2, alpha=0.7)
    if calm_start and calm_end:
        ax3.axvspan(calm_start, calm_end, alpha=0.15, color='green')
    ax3.set_title('Structural Memory (Φ)', fontweight='bold')
    ax3.set_ylabel('Φ')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Metrics distribution
    ax4 = fig.add_subplot(gs[2, 0])
    plot_cols = ['Xi', 'Oh', 'phi']
    if has_def:
        plot_cols.append('DEF')
    metrics_df = state[plot_cols].describe().T
    metrics_df[['mean', 'std', 'min', 'max']].plot(kind='bar', ax=ax4)
    ax4.set_title('Metrics Summary Statistics', fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Crossing timeline
    ax5 = fig.add_subplot(gs[2, 1])
    crossing_dates = []
    crossing_labels = []
    
    for label, crossing in crossings.items():
        if crossing.get('confirm'):
            crossing_dates.append(crossing['confirm'])
            crossing_labels.append(label)
    
    if crossing_dates:
        y_pos = np.arange(len(crossing_labels))
        dates_numeric = [(d - state.index[0]).days for d in crossing_dates]
        
        ax5.barh(y_pos, dates_numeric, color='#FF9800', alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(crossing_labels)
        ax5.set_xlabel('Days from start')
        ax5.set_title('Crossing Timeline', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # Plot 6: DEF evolution (if available)
    if has_def:
        ax6 = fig.add_subplot(gs[3, :])
        ax6.plot(state.index, state['DEF'], marker='o', linewidth=2, 
                color='#9C27B0', label='DEF')
        
        # Add reference line at 95th percentile
        def_threshold = state['DEF'].quantile(0.95)
        ax6.axhline(def_threshold, color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'P95 = {def_threshold:.3f}')
        
        if calm_start and calm_end:
            ax6.axvspan(calm_start, calm_end, alpha=0.15, color='green')
        
        ax6.set_xlabel('Date', fontweight='bold')
        ax6.set_ylabel('DEF (State-Phase Divergence)', fontweight='bold')
        ax6.set_title('State-Phase Divergence (DEF)', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Katashi Analysis Dashboard - {cohort_name}',
                fontsize=16, fontweight='bold')
    plt.show()


def plot_def_evolution(
    state: pd.DataFrame,
    calm_info: Dict = None,
    threshold: float = None,
    figsize: tuple = (14, 6)
):
    """
    Plot DEF (State-Phase Divergence) evolution.
    
    Args:
        state: State DataFrame with DEF column
        calm_info: CALM period information (optional)
        threshold: DEF threshold to highlight (optional)
        figsize: Figure size
    """
    if 'DEF' not in state.columns:
        raise ValueError("State DataFrame must contain 'DEF' column")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot DEF
    ax.plot(state.index, state['DEF'], marker='o', linewidth=2,
           color='#9C27B0', markersize=5, label='DEF')
    
    # Add threshold if provided
    if threshold is None:
        threshold = state['DEF'].quantile(0.95)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Threshold = {threshold:.3f}')
    
    # Add CALM period if provided
    if calm_info and calm_info.get('best'):
        calm_start = calm_info['best']['start']
        calm_end = calm_info['best']['end']
        ax.axvspan(calm_start, calm_end, alpha=0.15, color='green',
                  label='CALM period')
    
    # Highlight high DEF regions
    high_def = state['DEF'] > threshold
    if high_def.any():
        high_def_dates = state.index[high_def]
        ax.scatter(high_def_dates, state.loc[high_def, 'DEF'],
                  s=100, marker='X', color='red', edgecolors='black',
                  linewidths=2, zorder=5, label='High DEF')
    
    ax.set_xlabel('Date', fontweight='bold', fontsize=12)
    ax.set_ylabel('DEF (State-Phase Divergence)', fontweight='bold', fontsize=12)
    ax.set_title('State-Phase Divergence Evolution', fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_def_components(
    components_dict: dict,
    figsize: tuple = (10, 6)
):
    """
    Plot DEF component contributions.
    
    Args:
        components_dict: Output from compute_def_components()
        figsize: Figure size
    """
    components = components_dict['components']
    observables = components_dict['observables']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    values = [components[obs] for obs in observables]
    colors = ['#1976D2', '#FF9800', '#D32F2F', '#388E3C'][:len(observables)]
    
    bars = ax.bar(observables, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('DEF Contribution', fontweight='bold', fontsize=12)
    ax.set_title(f'DEF Component Analysis (Total DEF = {components_dict["DEF"]:.3f})',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
