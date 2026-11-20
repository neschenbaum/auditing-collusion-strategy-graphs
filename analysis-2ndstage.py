import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from env import CalvanoEnv
from analysis_utils import (
    analyze_run, 
    calculate_theoretical_profit, 
    plot_centrality_analysis, 
    simplify_category,
    THEORETICAL_BERTRAND_P, 
    THEORETICAL_MONOPOLY_P
)

def list_experiment_dirs(base_dir="runs"):
    if not os.path.exists(base_dir):
        return []
    
    experiments = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and '-' in name:
            run_count = len(glob.glob(os.path.join(path, "run_*.csv")))
            mtime = os.path.getmtime(path)
            experiments.append((path, name, run_count, mtime))
    
    experiments.sort(key=lambda x: x[3], reverse=True)
    return experiments

def get_latest_experiment_dir(base_dir="runs"):
    experiments = list_experiment_dirs(base_dir)
    if experiments:
        return experiments[0][0], experiments[0][1]
    return None, None

def analyze_rematch_run_logic(runs_dir, pi_bertrand, pi_monopoly, base_results_df):
    """
    Main logic for Phase 2 and 3 analysis.
    """
    rematch_dir = os.path.join(runs_dir, "rematch")
    if not os.path.exists(rematch_dir):
        print(f"Error: No rematch data found in {rematch_dir}")
        return

    print("Analyzing Rematch Results (Stages 2 & 3)...")
    
    summary_path = os.path.join(rematch_dir, "rematch_summary.csv")
    if not os.path.exists(summary_path):
        print("No rematch summary found.")
        return
        
    df_rematch = pd.read_csv(summary_path)
    
    # Calculate Deltas for Phases 2 and 3
    denom = pi_monopoly - pi_bertrand
    df_rematch["phase2_delta"] = ((df_rematch["phase2_mean_p1_profit"] + df_rematch["phase2_mean_p2_profit"]) / 2 - pi_bertrand) / denom
    df_rematch["phase3_delta"] = ((df_rematch["phase3_mean_p1_profit"] + df_rematch["phase3_mean_p2_profit"]) / 2 - pi_bertrand) / denom
    
    # Merge with Phase 1 Deltas if available
    if not base_results_df.empty and "run_id" in base_results_df.columns:
        # Check if run_id is compatible (string vs int type issues)
        base_results_df["run_id"] = base_results_df["run_id"].astype(str)
        run_deltas = base_results_df.set_index("run_id")["avg_delta"].to_dict()
        
        df_rematch["phase1_p1_delta"] = df_rematch["run_p1"].astype(str).map(lambda x: run_deltas.get(x, np.nan))
        df_rematch["phase1_p2_delta"] = df_rematch["run_p2"].astype(str).map(lambda x: run_deltas.get(x, np.nan))
        df_rematch["phase1_avg_delta"] = (df_rematch["phase1_p1_delta"] + df_rematch["phase1_p2_delta"]) / 2
    else:
        print("Warning: Base results missing or malformed. Phase 1 deltas will be NaN.")
        df_rematch["phase1_avg_delta"] = np.nan
        
    # Save basic comparison
    analysis_dir = os.path.join(runs_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    res_path = os.path.join(analysis_dir, "rematch_comparison.csv")
    cols = ["pair", "run_p1", "run_p2", "phase1_avg_delta", "phase2_delta", "phase3_delta"]
    # Handle missing columns gracefully
    cols = [c for c in cols if c in df_rematch.columns]
    df_rematch[cols].to_csv(res_path, index=False)
    print(f"Basic comparison saved to '{res_path}'")
    
    # --- Detailed Metrics ---
    print("Calculating deep metrics for Phase 2 and 3...")
    phase2_dir = os.path.join(rematch_dir, "phase2")
    phase3_dir = os.path.join(rematch_dir, "phase3")
    
    metrics_list = []
    env = CalvanoEnv()
    
    for _, row in df_rematch.iterrows():
        pair_name = row["pair"]
        run_p1 = row["run_p1"]
        run_p2 = row["run_p2"]
        
        # Phase 2: Uses ORIGINAL Q-tables
        p2_csv = os.path.join(phase2_dir, f"{pair_name}.csv")
        p1_q_ph1 = os.path.join(runs_dir, f"run_{run_p1}_qtable_p0.npy")
        p2_q_ph1 = os.path.join(runs_dir, f"run_{run_p2}_qtable_p1.npy")
        
        if os.path.exists(p2_csv) and os.path.exists(p1_q_ph1) and os.path.exists(p2_q_ph1):
            try:
                res = analyze_run(p2_csv, p1_q_ph1, p2_q_ph1, env, pi_bertrand, pi_monopoly, run_id_override=pair_name)
                res["pair"] = pair_name
                res["phase"] = "Phase 2"
                metrics_list.append(res)
            except Exception as e:
                print(f"Phase 2 error {pair_name}: {e}")
                
        # Phase 3: Uses NEW Q-tables
        p3_csv = os.path.join(phase3_dir, f"{pair_name}.csv")
        p1_q_ph3 = os.path.join(phase3_dir, f"{pair_name}_qtable_p0.npy")
        p2_q_ph3 = os.path.join(phase3_dir, f"{pair_name}_qtable_p1.npy")
        
        if os.path.exists(p3_csv) and os.path.exists(p1_q_ph3) and os.path.exists(p2_q_ph3):
            try:
                res = analyze_run(p3_csv, p1_q_ph3, p2_q_ph3, env, pi_bertrand, pi_monopoly, run_id_override=pair_name)
                res["pair"] = pair_name
                res["phase"] = "Phase 3"
                metrics_list.append(res)
            except Exception as e:
                print(f"Phase 3 error {pair_name}: {e}")
                
    if not metrics_list:
        print("No detailed metrics calculated.")
        return
        
    df_metrics = pd.DataFrame(metrics_list)
    detailed_path = os.path.join(analysis_dir, "rematch_detailed_results.csv")
    df_metrics.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to '{detailed_path}'")
    
    # --- PLOTTING ---
    # Centrality
    plot_centrality_analysis(df_metrics[df_metrics["phase"] == "Phase 2"], "Phase 2 (Test)", "phase2", analysis_dir)
    plot_centrality_analysis(df_metrics[df_metrics["phase"] == "Phase 3"], "Phase 3 (Relearn)", "phase3", analysis_dir)
    
    # Nash Analysis (Comparison)
    plot_nash_comparison(df_metrics, analysis_dir)
    
    # Delta Comparison
    plot_delta_comparison(df_rematch, analysis_dir)

def plot_nash_comparison(df_metrics, analysis_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df_p2 = df_metrics[df_metrics["phase"] == "Phase 2"]
    df_p3 = df_metrics[df_metrics["phase"] == "Phase 3"]
    
    # Basin Size
    axes[0, 0].scatter(df_p2["avg_delta"], df_p2["nash_basin_size"], 
                       alpha=0.6, s=40, c='#1f77b4', edgecolors='black', label='Phase 2')
    axes[0, 0].set_xlabel(r"Collusion Index $\Delta$")
    axes[0, 0].set_ylabel("Nash Basin Size (Count of States)")
    axes[0, 0].set_title("Phase 2: Nash Basin Size")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(df_p3["avg_delta"], df_p3["nash_basin_size"], 
                       alpha=0.6, s=40, c='#2ca02c', edgecolors='black', label='Phase 3')
    axes[0, 1].set_xlabel(r"Collusion Index $\Delta$")
    axes[0, 1].set_ylabel("Nash Basin Size (Count of States)")
    axes[0, 1].set_title("Phase 3: Nash Basin Size")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dynamic Transition Steps
    df_p2_reach = df_p2[df_p2["nash_to_attractor_steps"] >= 0]
    df_p3_reach = df_p3[df_p3["nash_to_attractor_steps"] >= 0]
    
    if not df_p2_reach.empty:
        axes[1, 0].scatter(df_p2_reach["avg_delta"], df_p2_reach["nash_to_attractor_steps"], 
                           alpha=0.5, s=40, c='#1f77b4', edgecolors='black', label='Phase 2')
    if not df_p3_reach.empty:
        axes[1, 0].scatter(df_p3_reach["avg_delta"], df_p3_reach["nash_to_attractor_steps"], 
                           alpha=0.5, s=40, c='#2ca02c', edgecolors='black', label='Phase 3')
                           
    axes[1, 0].set_xlabel(r"Collusion Index $\Delta$")
    axes[1, 0].set_ylabel("Steps from Nash to Attractor")
    axes[1, 0].set_title("Transition Time from Nash")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Prevalence
    has_nash_p2 = (df_p2["has_nash_attractor"].sum() / len(df_p2) * 100) if len(df_p2) > 0 else 0
    has_nash_p3 = (df_p3["has_nash_attractor"].sum() / len(df_p3) * 100) if len(df_p3) > 0 else 0
    
    axes[1, 1].bar(['Phase 2', 'Phase 3'], [has_nash_p2, has_nash_p3], 
                  color=['#1f77b4', '#2ca02c'], edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel("% of Runs with Nash Attractor")
    axes[1, 1].set_title("Prevalence of Nash Attractors")
    
    for i, val in enumerate([has_nash_p2, has_nash_p3]):
        axes[1, 1].text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "nash_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Nash comparison plots saved.")

def plot_delta_comparison(df_rematch, analysis_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phase 1 vs Phase 2
    axes[0].scatter(df_rematch["phase1_avg_delta"], df_rematch["phase2_delta"], 
                   alpha=0.6, s=50, c='#1f77b4', edgecolors='black')
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0].set_xlabel("Phase 1 Delta")
    axes[0].set_ylabel("Phase 2 Delta")
    axes[0].set_title("Collusion Breakdown (Ph1 vs Ph2)")
    axes[0].grid(True, alpha=0.3)
    
    # Phase 1 vs Phase 3
    axes[1].scatter(df_rematch["phase1_avg_delta"], df_rematch["phase3_delta"], 
                   alpha=0.6, s=50, c='#2ca02c', edgecolors='black')
    axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1].set_xlabel("Phase 1 Delta")
    axes[1].set_ylabel("Phase 3 Delta")
    axes[1].set_title("Collusion Recovery (Ph1 vs Ph3)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "delta_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Delta comparison plots saved.")

def main():
    parser = argparse.ArgumentParser(description="Analyze Calvano Rematch runs (Stage 2 & 3)")
    parser.add_argument("--dir", type=str, default=None, 
                       help="Experiment directory to analyze. If not specified, uses latest.")
    parser.add_argument("--list", action="store_true", 
                       help="List all available experiment directories")
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiment_dirs()
        if not experiments:
            print("No experiment directories found.")
        else:
            for path, name, run_count, mtime in experiments:
                print(f"  {name}: {run_count} runs")
        return
    
    if args.dir:
        if os.path.isabs(args.dir):
            runs_dir = args.dir
        else:
            runs_dir = os.path.join("runs", args.dir)
    else:
        runs_dir, exp_name = get_latest_experiment_dir()
        if runs_dir is None:
            print("No runs found.")
            return

    env = CalvanoEnv()
    pi_bertrand = calculate_theoretical_profit(env, THEORETICAL_BERTRAND_P, THEORETICAL_BERTRAND_P)
    pi_monopoly = calculate_theoretical_profit(env, THEORETICAL_MONOPOLY_P, THEORETICAL_MONOPOLY_P)
    
    # Try to load Stage 1 results for Delta comparison
    analysis_dir = os.path.join(runs_dir, "analysis")
    stage1_res_path = os.path.join(analysis_dir, "analysis_results.csv")
    
    if os.path.exists(stage1_res_path):
        base_results_df = pd.read_csv(stage1_res_path)
    else:
        print("Warning: Stage 1 analysis results not found. Please run analysis-1ststage.py first.")
        base_results_df = pd.DataFrame()
        
    analyze_rematch_run_logic(runs_dir, pi_bertrand, pi_monopoly, base_results_df)

if __name__ == "__main__":
    main()
