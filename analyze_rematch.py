"""
Compute graph-theoretic metrics for rematched (Phase 2) pairs.

For each pair (run_p1, run_p2), build the joint strategy graph using agent
0's Q-table from run_p1 and agent 1's Q-table from run_p2, and run the same
analysis pipeline as Phase 1 on the Phase-2 trajectory CSV.

Output: <training_dir>/rematch/phase2_analysis_results.csv with one row per
pair, plus a printout of correlations with Phase-2 CI.
"""

import os
import re
import sys
import glob
import json
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from env import CalvanoEnv, CalvanoAdjPriceGridEnv, NoisyObservationEnv
from analysis_utils import (
    analyze_run,
    calculate_theoretical_profit,
    THEORETICAL_BERTRAND_P,
    THEORETICAL_MONOPOLY_P,
)


def parse_pair_id(filename):
    m = re.match(r"pair_(\d+)_(\d+)\.csv$", os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_env(training_dir):
    cfg_path = os.path.join(training_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        name = cfg.get("name", "Calvano")
        if name in ("CalvanoAdjPriceEnv", "CalvanoAdjPriceGridEnv"):
            return CalvanoAdjPriceGridEnv()
        if name == "NoisyObservationEnv":
            return NoisyObservationEnv(noise_std=cfg.get("noise_std", 0.0))
    return CalvanoEnv()


def analyze_one(args):
    pair_csv, q_p0_path, q_p1_path, training_dir, pi_N, pi_M, run_p1, run_p2 = args
    env = build_env(training_dir)
    try:
        res = analyze_run(pair_csv, q_p0_path, q_p1_path, env, pi_N, pi_M,
                          run_id_override=f"{run_p1}_{run_p2}")
        res["run_p1"] = run_p1
        res["run_p2"] = run_p2
        return res
    except Exception as e:
        return {"run_id": f"{run_p1}_{run_p2}", "run_p1": run_p1, "run_p2": run_p2, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True,
                        help="Phase-1 directory that contains a rematch/ subdir")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not os.path.isabs(args.training_dir) and not os.path.exists(args.training_dir):
        args.training_dir = os.path.join("runs", args.training_dir)

    rematch_dir = os.path.join(args.training_dir, "rematch")
    phase2_dir = os.path.join(rematch_dir, "phase2")
    if not os.path.isdir(phase2_dir):
        print(f"No phase2 directory at {phase2_dir}")
        return

    env = build_env(args.training_dir)
    pi_N = calculate_theoretical_profit(env, THEORETICAL_BERTRAND_P, THEORETICAL_BERTRAND_P)
    pi_M = calculate_theoretical_profit(env, THEORETICAL_MONOPOLY_P, THEORETICAL_MONOPOLY_P)

    pair_files = sorted(glob.glob(os.path.join(phase2_dir, "pair_*.csv")))
    print(f"Found {len(pair_files)} rematch pairs in {phase2_dir}")

    tasks = []
    for f in pair_files:
        parsed = parse_pair_id(f)
        if parsed is None:
            continue
        p1, p2 = parsed
        q0 = os.path.join(args.training_dir, f"run_{p1}_qtable_p0.npy")
        q1 = os.path.join(args.training_dir, f"run_{p2}_qtable_p1.npy")
        if not (os.path.exists(q0) and os.path.exists(q1)):
            continue
        tasks.append((f, q0, q1, args.training_dir, pi_N, pi_M, p1, p2))

    print(f"Analyzing {len(tasks)} pairs with {args.workers} workers ...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(analyze_one, tasks)):
            results.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                print(f"  {i+1}/{len(tasks)} done")

    df = pd.DataFrame(results)
    out_path = os.path.join(rematch_dir, "phase2_analysis_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    if "avg_delta" not in df.columns:
        print("No avg_delta column; skipping correlations.")
        return

    print("\n=== Correlations with Phase-2 CI ===")
    metrics = [
        "max_betweenness",
        "attractor_indegree_centrality",
        "num_attractors",
        "basin_fraction",
        "attractor_closeness",
        "avg_path_length",
    ]
    for met in metrics:
        if met in df.columns:
            r = df[[met, "avg_delta"]].corr().iloc[0, 1]
            print(f"  {met:<32}  {r:+.3f}")

    print()
    print(f"Phase 2 CI: mean={df['avg_delta'].mean():.3f}, std={df['avg_delta'].std():.3f}")
    print(f"Phase 2 bottleneck fraction: mean={df['max_betweenness'].mean():.3f}")


if __name__ == "__main__":
    main()
