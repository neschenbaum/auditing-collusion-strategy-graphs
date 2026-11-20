import os
import glob
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import json
from env import CalvanoEnv, CalvanoAdjPriceGridEnv, CalvanoWideGridEnv
from analysis_utils import (
    analyze_run, 
    calculate_theoretical_profit, 
    plot_centrality_analysis,
    simplify_category,
    THEORETICAL_BERTRAND_P, 
    THEORETICAL_MONOPOLY_P
)

def list_experiment_dirs(base_dir="runs"):
    """List all experiment directories with their run counts."""
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

def plot_training_analysis(df_res, analysis_dir, env):
    """Generates consolidated plots for the training stage."""
    # Ensure plot_category exists
    if 'plot_category' not in df_res.columns:
        df_res['plot_category'] = df_res['category'].apply(simplify_category)
        
    categories = df_res["plot_category"].unique()
    
    color_map = {
        'sym1': '#2E86AB', 'asym1': '#A23B72', 
        'Short Cycle (≤3)': '#F18F01', 'Long Cycle (>3)': '#C73E1D', 'Other': '#6C757D'
    }
    point_colors = [color_map.get(cat, '#6C757D') for cat in df_res["plot_category"]]
    
    # Legend elements
    legend_elements = [Patch(facecolor=color_map[cat], label=cat) 
                      for cat in color_map.keys() if cat in categories]

    # --- MAIN COMPOSITE PLOT ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    # Plot 1: Closeness Centrality vs Delta
    axes[0, 0].scatter(df_res["avg_delta"], df_res["attractor_closeness"], 
                       c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[0, 0].set_xlabel(r"Collusion Index $\Delta$")
    axes[0, 0].set_ylabel("Attractor Mean Closeness (Avg over Cycle)")
    axes[0, 0].set_title("Closeness Centrality vs Collusion Index")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: In-degree Centrality vs Delta
    axes[0, 1].scatter(df_res["avg_delta"], df_res["attractor_indegree_centrality"], 
                       c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[0, 1].set_xlabel(r"Collusion Index $\Delta$")
    axes[0, 1].set_ylabel("Attractor Mean In-Degree (Avg over Cycle)")
    axes[0, 1].set_title("In-Degree Centrality vs Collusion Index")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Path Length vs Delta (Moved up)
    axes[1, 0].scatter(df_res["avg_delta"], df_res["avg_path_length"], 
                       c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[1, 0].set_xlabel(r"Collusion Index $\Delta$")
    axes[1, 0].set_ylabel("Mean Transient Length (Avg Steps to Convergence)")
    axes[1, 0].set_title("Path Length vs Collusion Index")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Num Attractors vs Delta (Moved up)
    y_jitter = np.random.normal(0, 0.1, size=len(df_res))
    axes[1, 1].scatter(df_res["avg_delta"], df_res["num_attractors"] + y_jitter, 
                       c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[1, 1].set_xlabel(r"Collusion Index $\Delta$")
    axes[1, 1].set_ylabel("Count of Disjoint Attractors (Jittered)")
    axes[1, 1].set_title("Num Attractors vs Collusion")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Dynamic Transition Steps from Nash (Moved from Nash plots)
    # Filter out unreachable cases (-1)
    df_reachable = df_res[df_res["nash_to_attractor_steps"] >= 0]
    
    if not df_reachable.empty:
        # Re-map colors for filtered data
        colors_reachable = [color_map.get(cat, '#6C757D') for cat in df_reachable["plot_category"]]
        
        axes[2, 0].scatter(df_reachable["avg_delta"], df_reachable["nash_to_attractor_steps"], 
                          c=colors_reachable, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
        axes[2, 0].set_xlabel(r"Collusion Index $\Delta$")
        axes[2, 0].set_ylabel("Steps from Nash to Attractor")
        axes[2, 0].set_title("Transition Time from Nash Equilibrium")
    else:
        axes[2, 0].text(0.5, 0.5, "Nash State cannot reach Attractors", 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title("Transition Time from Nash Equilibrium")
    
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Attractor Price Heatmap
    all_p1 = []
    all_p2 = []
    
    for idx, row in df_res.iterrows():
        prices = row.get('attractor_prices', [])
        if isinstance(prices, str):
            try:
                import ast
                prices = ast.literal_eval(prices)
            except:
                prices = []
        
        for p in prices:
            all_p1.append(p[0])
            all_p2.append(p[1])
            
    if all_p1:
        grid = env.price_grid
        step = grid[1] - grid[0]
        bins = np.linspace(grid[0] - step/2, grid[-1] + step/2, len(grid) + 1)
        
        h = axes[2, 1].hist2d(all_p1, all_p2, bins=bins, cmap='viridis', cmin=1)
        cbar = fig.colorbar(h[3], ax=axes[2, 1])
        cbar.set_label('Frequency (Count of States)')

    axes[2, 1].axvline(THEORETICAL_BERTRAND_P, color='red', linestyle=':', alpha=0.8, label='Bertrand')
    axes[2, 1].axhline(THEORETICAL_BERTRAND_P, color='red', linestyle=':', alpha=0.8)
    axes[2, 1].axvline(THEORETICAL_MONOPOLY_P, color='white', linestyle='--', alpha=0.8, label='Monopoly')
    axes[2, 1].axhline(THEORETICAL_MONOPOLY_P, color='white', linestyle='--', alpha=0.8)
    
    axes[2, 1].set_xlabel("Player 1 Price")
    axes[2, 1].set_ylabel("Player 2 Price")
    axes[2, 1].set_title("Attractor State Density (Heatmap)")
    axes[2, 1].grid(False)
    
    # Shared legend
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plot_path = os.path.join(analysis_dir, "centrality_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Main analysis plot saved to '{plot_path}'")

def plot_path_complexity_analysis(df_res, analysis_dir):
    """Generates plots for path variance and bottleneck analysis."""
    if 'plot_category' not in df_res.columns:
        df_res['plot_category'] = df_res['category'].apply(simplify_category)
        
    categories = df_res["plot_category"].unique()
    
    color_map = {
        'sym1': '#2E86AB', 'asym1': '#A23B72', 
        'Short Cycle (≤3)': '#F18F01', 'Long Cycle (>3)': '#C73E1D', 'Other': '#6C757D'
    }
    point_colors = [color_map.get(cat, '#6C757D') for cat in df_res["plot_category"]]
    
    # Legend elements
    legend_elements = [Patch(facecolor=color_map[cat], label=cat) 
                      for cat in color_map.keys() if cat in categories]

    # --- PATH COMPLEXITY PLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Plot 1: Path Length Variance vs Delta
    axes[0].scatter(df_res["avg_delta"], df_res["path_variance"],
                   c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[0].set_xlabel(r"Collusion Index $\Delta$")
    axes[0].set_ylabel("Path Length Variance")
    axes[0].set_title("Path Length Variance vs Collusion Index")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Number of Bottleneck States vs Delta
    y_jitter = np.random.normal(0, 0.3, size=len(df_res))
    axes[1].scatter(df_res["avg_delta"], df_res["num_bottlenecks"] + y_jitter,
                   c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[1].set_xlabel(r"Collusion Index $\Delta$")
    axes[1].set_ylabel("Number of Bottleneck States (Jittered)")
    axes[1].set_title("Bottleneck States vs Collusion Index")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Max Betweenness Centrality vs Delta
    axes[2].scatter(df_res["avg_delta"], df_res["max_betweenness"],
                   c=point_colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    axes[2].set_xlabel(r"Collusion Index $\Delta$")
    axes[2].set_ylabel("Max Betweenness Centrality")
    axes[2].set_title("Max Betweenness Centrality vs Collusion Index")
    axes[2].grid(True, alpha=0.3)

    # Shared legend
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plot_path = os.path.join(analysis_dir, "path_complexity_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Path complexity plot saved to '{plot_path}'")


def plot_bottleneck_prices(df_res, analysis_dir):
    """Scatter plot of bottleneck mean price vs attractor mean price, coloured by CI."""
    df_bn = df_res[df_res["bottleneck_mean_price"].notna()].copy()
    if df_bn.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(df_bn["bottleneck_mean_price"], df_bn["attractor_mean_price"],
                         c=df_bn["avg_delta"], cmap="RdYlGn", alpha=0.6, s=20,
                         edgecolors='black', linewidths=0.3, vmin=0, vmax=1)
    fig.colorbar(sc, ax=axes[0], label=r"Collusion Index $\Delta$")
    m0, b0 = np.polyfit(df_bn["bottleneck_mean_price"], df_bn["attractor_mean_price"], 1)
    x0 = np.linspace(df_bn["bottleneck_mean_price"].min(), df_bn["bottleneck_mean_price"].max(), 100)
    axes[0].plot(x0, m0 * x0 + b0, 'k-', linewidth=1.5, label=f'fit (slope={m0:.2f})')
    axes[0].axvline(THEORETICAL_BERTRAND_P, color='red', linestyle=':', alpha=0.6, label='Nash')
    axes[0].axhline(THEORETICAL_BERTRAND_P, color='red', linestyle=':', alpha=0.6)
    axes[0].axvline(THEORETICAL_MONOPOLY_P, color='green', linestyle=':', alpha=0.6, label='Monopoly')
    axes[0].axhline(THEORETICAL_MONOPOLY_P, color='green', linestyle=':', alpha=0.6)
    axes[0].set_xlabel("Bottleneck Mean Price")
    axes[0].set_ylabel("Attractor Mean Price")
    axes[0].set_title("Bottleneck vs Attractor Prices")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    price_gap = df_bn["attractor_mean_price"] - df_bn["bottleneck_mean_price"]
    axes[1].scatter(df_bn["avg_delta"], price_gap,
                   c=df_bn["avg_delta"], cmap="RdYlGn", alpha=0.6, s=20,
                   edgecolors='black', linewidths=0.3, vmin=0, vmax=1)
    m1, b1 = np.polyfit(df_bn["avg_delta"], price_gap, 1)
    x1 = np.linspace(df_bn["avg_delta"].min(), df_bn["avg_delta"].max(), 100)
    axes[1].plot(x1, m1 * x1 + b1, 'k-', linewidth=1.5, label=f'fit (slope={m1:.2f})')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    axes[1].set_xlabel(r"Collusion Index $\Delta$")
    axes[1].set_ylabel("Attractor Price − Bottleneck Price")
    axes[1].set_title("Price Gap (Attractor − Bottleneck) vs Collusion Index")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(analysis_dir, "bottleneck_prices.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bottleneck price plot saved to '{plot_path}'")


def main():
    parser = argparse.ArgumentParser(description="Analyze Calvano Replication runs (Stage 1)")
    parser.add_argument("--dir", type=str, default=None, 
                       help="Experiment directory to analyze. If not specified, uses latest.")
    parser.add_argument("--list", action="store_true", 
                       help="List all available experiment directories")
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiment_dirs()
        if not experiments:
            print("No experiment directories found in runs/")
        else:
            print("Available experiment directories:")
            for path, name, run_count, mtime in experiments:
                from datetime import datetime
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {name}: {run_count} runs (last modified: {mtime_str})")
        return
    
    # Select Directory
    if args.dir:
        if os.path.isabs(args.dir):
            runs_dir = args.dir
        else:
            runs_dir = os.path.join("runs", args.dir)
        exp_name = os.path.basename(runs_dir)
        if not os.path.exists(runs_dir):
            print(f"Error: Directory '{runs_dir}' does not exist.")
            return
    else:
        runs_dir, exp_name = get_latest_experiment_dir()
        if runs_dir is None:
            if os.path.exists("runs") and glob.glob("runs/run_*.csv"):
                runs_dir = "runs"
                exp_name = "legacy"
            else:
                print("No runs found.")
                return

    print("Starting Stage 1 Analysis...")
    print(f"Analyzing: {exp_name}")
    print(f"Directory: {runs_dir}")
    
    # Check config for environment type
    config_path = os.path.join(runs_dir, "config.json")
    env_class = CalvanoEnv  # Default
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("name") in ("CalvanoAdjPriceEnv", "CalvanoAdjPriceGridEnv"):
                    env_class = CalvanoAdjPriceGridEnv
                    print("Detected CalvanoAdjPriceGridEnv from config.")
                elif config.get("name") == "CalvanoWideGridEnv":
                    env_class = CalvanoWideGridEnv
                    print("Detected CalvanoWideGridEnv from config.")
        except Exception as e:
            print(f"Warning: Could not read config.json: {e}")
    
    env = env_class()
    pi_bertrand = calculate_theoretical_profit(env, THEORETICAL_BERTRAND_P, THEORETICAL_BERTRAND_P)
    pi_monopoly = calculate_theoretical_profit(env, THEORETICAL_MONOPOLY_P, THEORETICAL_MONOPOLY_P)
    
    run_files = sorted(glob.glob(os.path.join(runs_dir, "run_*.csv")))
    print(f"Found {len(run_files)} runs to analyze")
    
    if not run_files:
        print("No run files to analyze.")
        return
    
    results = []
    
    for i, run_file in enumerate(run_files):
        run_id = os.path.basename(run_file).split('_')[1].split('.')[0]
        q0 = os.path.join(runs_dir, f"run_{run_id}_qtable_p0.npy")
        q1 = os.path.join(runs_dir, f"run_{run_id}_qtable_p1.npy")
        
        if os.path.exists(q0) and os.path.exists(q1):
            try:
                res = analyze_run(run_file, q0, q1, env, pi_bertrand, pi_monopoly)
                results.append(res)
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(run_files)} runs...")
            except Exception as e:
                print(f"Error processing run {run_id}: {e}")
                
    if not results:
        print("No results generated.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Save Results
    analysis_dir = os.path.join(runs_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    csv_path = os.path.join(analysis_dir, "analysis_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nAnalysis saved to '{csv_path}'")
    
    # Summary Props
    print("\n=== SUMMARY STATISTICS ===")
    print(df_res['category'].value_counts().to_string())
    print("\nAverage Delta by Category:")
    print(df_res.groupby('category')['avg_delta'].agg(['mean', 'std', 'count']).to_string())
    
    # Summary stats for new metrics
    print("\n=== PATH COMPLEXITY METRICS ===")
    print(f"Path Variance - Mean: {df_res['path_variance'].mean():.2f}, Std: {df_res['path_variance'].std():.2f}")
    print(f"Path Std Dev - Mean: {df_res['path_std'].mean():.2f}, Std: {df_res['path_std'].std():.2f}")
    print(f"Num Bottlenecks - Mean: {df_res['num_bottlenecks'].mean():.2f}, Std: {df_res['num_bottlenecks'].std():.2f}")
    print(f"Mean Betweenness - Mean: {df_res['mean_betweenness'].mean():.4f}, Std: {df_res['mean_betweenness'].std():.4f}")
    print(f"Max Betweenness - Mean: {df_res['max_betweenness'].mean():.4f}, Std: {df_res['max_betweenness'].std():.4f}")
    
    # Correlations
    print("\n=== CORRELATIONS WITH COLLUSION INDEX ===")
    print(f"Path Variance vs Delta: {df_res[['path_variance', 'avg_delta']].corr().iloc[0,1]:.3f}")
    print(f"Num Bottlenecks vs Delta: {df_res[['num_bottlenecks', 'avg_delta']].corr().iloc[0,1]:.3f}")
    print(f"Max Betweenness vs Delta: {df_res[['max_betweenness', 'avg_delta']].corr().iloc[0,1]:.3f}")

    print("\n=== BOTTLENECK PRICE CHECK ===")
    has_bottleneck = df_res["bottleneck_mean_price"].notna()
    df_bn = df_res[has_bottleneck]
    pct_below = df_bn["bottleneck_below_attractor"].mean() * 100
    print(f"Runs with a transient bottleneck: {has_bottleneck.sum()}/{len(df_res)}")
    print(f"Bottleneck price below attractor price: {pct_below:.1f}%")
    price_diff = df_bn["attractor_mean_price"] - df_bn["bottleneck_mean_price"]
    print(f"Mean price gap (attractor - bottleneck): {price_diff.mean():.4f} (std {price_diff.std():.4f})")
    print(f"By collusion category (high CI = avg_delta > 0.6):")
    for label, mask in [("High CI (>0.6)", df_bn["avg_delta"] > 0.6), ("Low CI (<=0.6)", df_bn["avg_delta"] <= 0.6)]:
        sub = df_bn[mask]
        if len(sub):
            print(f"  {label}: {sub['bottleneck_below_attractor'].mean()*100:.1f}% below, mean gap {(sub['attractor_mean_price']-sub['bottleneck_mean_price']).mean():.4f}")
    
    # Plotting
    plot_training_analysis(df_res, analysis_dir, env)
    plot_path_complexity_analysis(df_res, analysis_dir)
    plot_bottleneck_prices(df_res, analysis_dir)

if __name__ == "__main__":
    main()

