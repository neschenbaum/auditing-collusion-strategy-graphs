"""
Summarize Calvano gamma-sweep analyses.

Expected workflow:
  1. Train each gamma with main.py train --gamma ...
  2. Analyze each run directory with analysis-1ststage.py --dir ...
  3. Run this script to combine all analyzed gamma directories.

The script scans runs/{env}-*/config.json and the matching
analysis/analysis_results.csv files, then writes:
  - gamma_sweep_runs.csv          row-level combined data
  - gamma_sweep_by_gamma.csv      per-gamma summary
  - gamma_sweep_correlations.csv  pooled and within-gamma correlations
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from paper_common import METRIC_KEYS, corr_column, corr_table_tex, write_table

# Canonical marker set/order shared across all paper scenarios.
METRICS = METRIC_KEYS


def corr(x, y):
    if len(x) < 2 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y)))


def load_runs(base_dir, env_name):
    frames = []
    skipped = []

    for exp_dir in sorted(Path(base_dir).glob(f"{env_name}-*")):
        config_path = exp_dir / "config.json"
        analysis_path = exp_dir / "analysis" / "analysis_results.csv"

        if not config_path.exists():
            skipped.append((exp_dir.name, "missing config.json"))
            continue
        if not analysis_path.exists():
            skipped.append((exp_dir.name, "missing analysis/analysis_results.csv"))
            continue

        with config_path.open("r") as f:
            config = json.load(f)
        if config.get("name") != env_name:
            skipped.append((exp_dir.name, f"config name is {config.get('name')}"))
            continue

        df = pd.read_csv(analysis_path)
        if "avg_delta" not in df.columns:
            skipped.append((exp_dir.name, "analysis missing avg_delta"))
            continue

        df = df.copy()
        df["gamma"] = float(config["gamma"])
        df["exp_name"] = exp_dir.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), skipped

    return pd.concat(frames, ignore_index=True), skipped


def summarize_by_gamma(df):
    agg = {
        "run_id": "count",
        "avg_delta": ["mean", "std", "min", "max"],
    }
    for metric in METRICS:
        if metric in df.columns:
            agg[metric] = ["mean", "std"]

    out = df.groupby("gamma").agg(agg)
    out.columns = ["_".join([str(x) for x in col if x]) for col in out.columns]
    out = out.rename(columns={"run_id_count": "n_runs"}).reset_index()
    return out


def summarize_correlations(df):
    records = []

    groups = [("pooled", np.nan, df)]
    groups.extend((f"gamma={g:.3g}", g, sub) for g, sub in df.groupby("gamma"))

    for group_name, gamma, sub in groups:
        for metric in METRICS:
            if metric not in sub.columns:
                continue
            records.append({
                "group": group_name,
                "gamma": gamma,
                "metric": metric,
                "n": len(sub),
                "rho_vs_delta": corr(sub[metric], sub["avg_delta"]),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Summarize Calvano gamma sweep analyses")
    parser.add_argument("--base-dir", default="runs", help="Directory containing run folders")
    parser.add_argument("--env", default="CalvanoAdjPriceGridEnv", help="Environment/run prefix")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Where to write summary CSVs")
    args = parser.parse_args()

    df, skipped = load_runs(args.base_dir, args.env)
    if df.empty:
        print("No analyzed gamma runs found.")
        for name, reason in skipped:
            print(f"  skipped {name}: {reason}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_path = out_dir / "gamma_sweep_runs.csv"
    by_gamma_path = out_dir / "gamma_sweep_by_gamma.csv"
    corr_path = out_dir / "gamma_sweep_correlations.csv"

    df.to_csv(runs_path, index=False)
    by_gamma = summarize_by_gamma(df)
    corrs = summarize_correlations(df)
    by_gamma.to_csv(by_gamma_path, index=False)
    corrs.to_csv(corr_path, index=False)

    print(f"Loaded {len(df)} analyzed runs from {df['exp_name'].nunique()} directories.")
    print("\n=== Per-gamma delta summary ===")
    cols = ["gamma", "n_runs", "avg_delta_mean", "avg_delta_std", "avg_delta_min", "avg_delta_max"]
    print(by_gamma[cols].round(3).to_string(index=False))

    print("\n=== Pooled metric correlations with delta ===")
    pooled = corrs[corrs["group"] == "pooled"][["metric", "rho_vs_delta"]]
    print(pooled.round(3).to_string(index=False))

    if skipped:
        print("\nSkipped directories:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    print(f"\nsaved {runs_path}")
    print(f"saved {by_gamma_path}")
    print(f"saved {corr_path}")

    # Appendix LaTeX table: per-gamma + pooled correlations
    gammas = sorted(df["gamma"].unique())
    headers, rho_cols, delta_bars, ns = [], [], [], []
    for g in gammas:
        sub = df[df["gamma"] == g]
        headers.append(rf"$\gamma{{=}}{g:.3g}$")
        rho_cols.append(corr_column(sub, delta_col="avg_delta"))
        delta_bars.append(sub["avg_delta"].mean())
        ns.append(len(sub))
    headers.append("Pooled")
    rho_cols.append(corr_column(df, delta_col="avg_delta"))
    delta_bars.append(df["avg_delta"].mean())
    ns.append(len(df))

    tex = corr_table_tex(
        headers, rho_cols, delta_bars=delta_bars, ns=ns,
        caption=(r"Pearson correlations $\rho(\cdot,\Delta)$ between strategy "
                 r"graph metrics and the Collusion Index for Calvano Q-learning "
                 r"runs by discount factor $\gamma$ (adjusted 15-price grid "
                 r"containing $p^N$ and $p^M$ exactly)."),
        label="tab:calvano-gamma")
    write_table("appendix_calvano_gamma.tex", tex)


if __name__ == "__main__":
    main()

