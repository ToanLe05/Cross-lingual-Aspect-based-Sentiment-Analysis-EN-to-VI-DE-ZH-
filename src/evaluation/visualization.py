"""
src/evaluation/visualization.py
───────────────────────────────
Plotting functions for evaluation results (Premium Edition).
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.evaluation.metrics import ErrorAnalyzer

# Premium Seaborn Aesthetics
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

def plot_recovery_curves(df: pd.DataFrame, fig_dir: Path):
    """
    Plot macro F1 vs number of target samples (N) for each (domain, target).
    """
    plot_df = df[df['setting'].isin(['s1', 's2'])].copy()
    plot_df.loc[plot_df['setting'] == 's1', 'n'] = 0
    full_df = df[df['setting'] == 's3'].copy()
    full_avg = full_df.groupby(['model', 'domain', 'target'])['macro_f1'].mean().reset_index()

    for (domain, target), sub in plot_df.groupby(['domain', 'target']):
        plt.figure(figsize=(10, 6), dpi=200)
        
        # Plot with seaborn for shaded error bands and robust markers
        ax = sns.lineplot(
            data=sub, 
            x='n', y='macro_f1', hue='model', style='model',
            markers=True, dashes=False, linewidth=2.5, markersize=10, errorbar='sd'
        )
        
        # Add full-data horizontal lines
        colors = sns.color_palette("deep")
        models = sub['model'].unique()
        model_color_map = {m: colors[i % len(colors)] for i, m in enumerate(models)}
        
        for _, row in full_avg[(full_avg['domain'] == domain) & (full_avg['target'] == target)].iterrows():
            m = row['model']
            if m in model_color_map:
                ax.axhline(y=row['macro_f1'], linestyle='--', color=model_color_map[m], alpha=0.6, linewidth=2)
                ax.text(plot_df['n'].max() * 0.85, row['macro_f1'] + 0.005,
                         f"{m} full: {row['macro_f1']:.3f}", fontsize=10, color=model_color_map[m], fontweight='bold')
        
        # Handle the 0-shot plot smoothly
        ax.set_xscale('symlog', linthresh=10)
        ax.set_xticks([0, 50, 100, 200])
        ax.set_xticklabels(['0\n(Zero-shot)', '50', '100', '200'])
        
        plt.xlabel("Number of target samples (N)", fontweight='bold')
        plt.ylabel("Macro F1 Score", fontweight='bold')
        plt.title(f"Cross-lingual Recovery: {domain.capitalize()} (EN → {target.upper()})", fontsize=16, fontweight='bold', pad=15)
        
        plt.legend(title="Models", title_fontsize='12', fontsize='11', loc='lower right', frameon=True, shadow=True)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        
        out_path = fig_dir / f"recovery_{domain}_{target}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {out_path}")

def plot_gap_matrix(df: pd.DataFrame, fig_dir: Path):
    """Heatmap of zero-shot macro F1 across models × target languages."""
    zero = df[df['setting'] == 's1'].copy()
    if zero.empty:
        print("No zero-shot data.")
        return
    
    # Pivot: rows = (model, domain), columns = target languages
    pivot = zero.pivot_table(index=['model', 'domain'], columns='target', values='macro_f1')
    if pivot.empty:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=1.5,
                cbar_kws={'label': 'Macro F1 Score', 'shrink': 0.8},
                annot_kws={"size": 13, "weight": "bold"})
    
    plt.title("Zero-shot Cross-lingual Performance (Higher is Better)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Model & Domain", fontweight='bold')
    plt.xlabel("Target Language", fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    plt.tight_layout()
    
    out_path = fig_dir / "gap_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")

def plot_error_taxonomy(errors_dir: Path, fig_dir: Path, top_k: int = 10):
    """Generates an aesthetic Bar Chart and Donut Chart of error types."""
    err_files = list(errors_dir.rglob("errors_*.jsonl"))
    if not err_files:
        print("No error files.")
        return
    counts = defaultdict(int)
    total = 0
    for f in err_files:
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                err = json.loads(line)
                etype = ErrorAnalyzer.tag_error(err)
                counts[etype] += 1
                total += 1
    if total == 0:
        return
    df_err = pd.DataFrame([{'type': k, 'count': v, 'pct': v/total*100} for k, v in counts.items()])
    df_err = df_err.sort_values('count', ascending=False).head(top_k)
    
    # 1. Bar Chart
    plt.figure(figsize=(12, 7), dpi=200)
    ax = sns.barplot(data=df_err, y='type', x='pct', hue='type', palette='magma', legend=False)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2.,
                    f'{width:.1f}%', ha="left", va="center", fontweight='bold', fontsize=12)
            
    plt.ylabel("Error Type", fontweight='bold')
    plt.xlabel("Percentage of Errors (%)", fontweight='bold')
    plt.title(f"Error Taxonomy Analysis (Total Errors: {total})", fontsize=16, fontweight='bold', pad=15)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    out_path_bar = fig_dir / "error_taxonomy_bar.png"
    plt.savefig(out_path_bar, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path_bar}")
    
    # 2. Donut Chart
    plt.figure(figsize=(8, 8), dpi=200)
    colors = sns.color_palette('magma', len(df_err))
    plt.pie(df_err['pct'], labels=df_err['type'], autopct='%1.1f%%', startangle=140, 
            colors=colors, textprops={'fontsize': 11, 'weight': 'bold'}, 
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'})
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Error Distribution", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out_path_donut = fig_dir / "error_taxonomy_donut.png"
    plt.savefig(out_path_donut, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path_donut}")

def plot_macro_f1_comparison(df: pd.DataFrame, fig_dir: Path):
    """Bar chart comparing S1 (Zero-shot) vs S3 (Full-data) for each target."""
    comp_df = df[df['setting'].isin(['s1', 's3'])].copy()
    if comp_df.empty:
        return
    
    agg_df = comp_df.groupby(['model', 'domain', 'target', 'setting'])['macro_f1'].mean().reset_index()
    agg_df['setting'] = agg_df['setting'].map({'s1': 'Zero-shot (S1)', 's3': 'Full-data (S3)'})
    
    for domain, sub in agg_df.groupby('domain'):
        g = sns.catplot(
            data=sub, kind="bar",
            x="model", y="macro_f1", hue="setting", col="target",
            palette="muted", height=5, aspect=1.2, legend_out=False
        )
        g.set_axis_labels("Models", "Macro F1 Score", fontweight='bold')
        g.set_titles("Target: {col_name}", size=14, weight='bold')
        g.despine(left=True)
        
        # Style adjustments
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"Performance Comparison: Zero-shot vs Full-data ({domain.capitalize()})", fontsize=16, fontweight='bold')
        
        out_path = fig_dir / f"f1_comparison_{domain}.png"
        g.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {out_path}")

def plot_training_history(history: dict, fig_dir: Path, model_name: str, setting: str):
    """
    Plots the training loss and validation F1 convergence curve.
    history is a dict with 'train_loss' and 'val_f1' lists.
    """
    if not history or not history.get('train_loss'):
        print(f"No history found for {model_name} {setting}")
        return
        
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, history['train_loss'], color=color, marker='o', linewidth=2.5, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Validation F1 Score', color=color, fontweight='bold')  
    ax2.plot(epochs, history['val_f1'], color=color, marker='s', linewidth=2.5, label='Validation F1')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Training Dynamics: {model_name} ({setting})", fontsize=16, fontweight='bold', pad=15)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True)
    
    out_path = fig_dir / f"training_curve_{model_name}_{setting}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {out_path}")