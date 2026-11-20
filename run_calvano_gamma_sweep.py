"""
Run and analyze a Calvano discount-factor sweep.

This is a thin orchestration wrapper around existing scripts:
  - main.py train
  - analysis-1ststage.py
  - summarize_calvano_gamma_sweep.py

Example:
  python run_calvano_gamma_sweep.py --runs 200 --workers 12

Use --dry-run to print the exact commands without running them.
"""

import argparse
import subprocess
import sys

import main as calvano_main


DEFAULT_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]


def run_cmd(cmd, dry_run=False):
    print(" ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def experiment_name(env, gamma, iterations):
    config = calvano_main.EXPERIMENT_CONFIG.copy()
    config["name"] = env
    config["gamma"] = gamma
    config["iterations"] = iterations
    _, exp_name = calvano_main.get_experiment_dir(config)
    return exp_name


def main():
    parser = argparse.ArgumentParser(description="Run Calvano gamma sweep")
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=calvano_main.EXPERIMENT_CONFIG["iterations"])
    parser.add_argument("--env", default=calvano_main.EXPERIMENT_CONFIG["name"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    workers = args.workers

    for gamma in args.gammas:
        exp_name = experiment_name(args.env, gamma, args.iterations)

        if not args.skip_train:
            cmd = [
                py, "main.py", "train",
                "--env", args.env,
                "--gamma", str(gamma),
                "--runs", str(args.runs),
                "--iterations", str(args.iterations),
            ]
            if workers is not None:
                cmd.extend(["--workers", str(workers)])
            run_cmd(cmd, dry_run=args.dry_run)

        if not args.skip_analysis:
            run_cmd([py, "analysis-1ststage.py", "--dir", exp_name], dry_run=args.dry_run)

    if not args.skip_summary:
        run_cmd([py, "summarize_calvano_gamma_sweep.py", "--env", args.env], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
