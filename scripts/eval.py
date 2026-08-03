#!/usr/bin/env python
"""
Aggregate results from JSON files and generate all evaluation figures.
Usage:
    python scripts/eval.py
    python scripts/eval.py --results_dir outputs/results --fig_dir outputs/figures
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import json
from pathlib import Path

import pandas as pd

from src.evaluation.visualization import (
    plot_recovery_curves, 
    plot_gap_matrix, 
    plot_error_taxonomy, 
    plot_macro_f1_comparison,
    plot_training_history
)

def generate_statistical_summary(df: pd.DataFrame, results_dir: Path):
    from scipy import stats
    import numpy as np

    groups = df.groupby(['domain', 'setting', 'target', 'n'])
    stats_records = []
    
    for (domain, setting, target, n), group in groups:
        ag_can_scores = group[group['model'] == 'ag_can']['macro_f1'].values
        
        for model in group['model'].unique():
            model_scores = group[group['model'] == model]['macro_f1'].values
            
            mean_score = np.mean(model_scores) if len(model_scores) > 0 else 0.0
            std_score = np.std(model_scores) if len(model_scores) > 0 else 0.0
            
            score_str = f"{mean_score*100:.1f} ± {std_score*100:.1f}"
            
            p_value = 1.0
            if model != 'ag_can' and len(ag_can_scores) > 1 and len(model_scores) > 1:
                _, p_value = stats.ttest_ind(ag_can_scores, model_scores, equal_var=False)
                # If AG-CAN is significantly better than this model, append * to AG-CAN's row later
                
            stats_records.append({
                "domain": domain,
                "setting": setting,
                "target": target,
                "n": n,
                "model": model,
                "mean_f1": mean_score * 100,
                "std_f1": std_score * 100,
                "formatted_score": score_str,
                "p_value_vs_agcan": p_value
            })
            
    stats_df = pd.DataFrame(stats_records)
    
    # Mark AG-CAN with * if it significantly beats the best baseline
    for (domain, setting, target, n), group in stats_df.groupby(['domain', 'setting', 'target', 'n']):
        baselines = group[group['model'] != 'ag_can']
        ag_can_row = group[group['model'] == 'ag_can']
        if not baselines.empty and not ag_can_row.empty:
            best_baseline_p = baselines['p_value_vs_agcan'].max()  # least significant difference
            if best_baseline_p < 0.05:
                idx = ag_can_row.index[0]
                stats_df.at[idx, 'formatted_score'] = stats_df.at[idx, 'formatted_score'] + "*"

    if not stats_df.empty:
        # Create pivot table for easy copy-pasting
        pivot = stats_df.pivot_table(
            index=['domain', 'setting', 'target', 'n'], 
            columns='model', 
            values='formatted_score',
            aggfunc='first'
        )
        out_path = results_dir / "statistical_summary.csv"
        pivot.to_csv(out_path)
        print(f"Statistical summary saved to {out_path}")


def aggregate_results(results_dir: Path, fig_dir: Path) -> pd.DataFrame:
    """Load all JSON result files and return a single DataFrame."""
    records = []
    for f in results_dir.rglob("*.json"):
        parts = f.stem.split("_")
        if len(parts) < 6:
            continue
        seed_str = parts[-1]
        n_str = parts[-2]
        target = parts[-3]
        setting = parts[-4]
        domain = parts[-5]
        model = "_".join(parts[:-5])
        n = int(n_str) if setting == "s2" else 0
        with open(f, 'r') as fp:
            metrics = json.load(fp)
        
        records.append({
            "model": model,
            "domain": domain,
            "setting": setting,
            "target": target,
            "n": n,
            "samples": n,
            "seed": int(seed_str),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_positive": metrics.get("f1_positive", 0.0),
            "f1_negative": metrics.get("f1_negative", 0.0),
            "f1_neutral": metrics.get("f1_neutral", 0.0),
        })
        
        # Plot training curve if history is present in the metrics JSON
        if "history" in metrics:
            plot_training_history(metrics["history"], fig_dir, model, f"{setting}_{domain}_{target}_seed{seed_str}")
        elif "train_loss" in metrics and "val_f1" in metrics:
            history = {"train_loss": metrics["train_loss"], "val_f1": metrics["val_f1"]}
            plot_training_history(history, fig_dir, model, f"{setting}_{domain}_{target}_seed{seed_str}")
            
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--fig_dir", default="outputs/figures")
    parser.add_argument("--errors_dir", default="outputs/errors")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate_results(results_dir, fig_dir)
    if df.empty:
        print(f"No result files found in {results_dir}. Run experiments first.")
        return

    # Save aggregated summary CSV
    summary_path = results_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Aggregated {len(df)} records -> {summary_path}")

    # Generate Statistical Significance Summary
    generate_statistical_summary(df, results_dir)

    # Generate all plots
    plot_recovery_curves(df, fig_dir)
    plot_gap_matrix(df, fig_dir)
    plot_error_taxonomy(Path(args.errors_dir), fig_dir)
    plot_macro_f1_comparison(df, fig_dir)

    print(f"All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
