"""
Appendix robustness tables for the Calvano Q-learning results.

  - Baseline discount factor (gamma=0.95) pooled with the low-discount runs
    (gamma=0.1): a two-point competitive-vs-collusive panel.
  - Wider price grid (twenty-one prices, T=2e6, gamma=0.95).

Both reuse the shared markers and the largest-WCC / profit-Delta conventions of
paper_common, so the columns line up with the main tables (including the
normalized betweenness).

Output: paper_content/tables/appendix_calvano_robustness.tex
"""
import os
import sys
import glob

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from paper_common import (
    add_betweenness_norm, corr_column, corr_table_tex, metrics_with_norm,
    write_table,
)


def _cycle_length(df):
    """period = realized attractor length (states in attractor_prices); the
    stored play-trace period disagrees with the graph attractor used for the
    normalized betweenness."""
    if "attractor_prices" in df.columns:
        s = df["attractor_prices"].astype(str)
        df["period"] = ((s.str.count(r"\)\s*,\s*\(") + 1)
                        .where(s.str.contains(r"\("), 0).astype(int))
    return df


def main():
    metrics = metrics_with_norm()
    keys = [k for k, _ in metrics]

    cal = pd.read_csv(os.path.join(ROOT, "analysis_outputs", "gamma_sweep_runs.csv")
                      ).rename(columns={"avg_delta": "delta"})
    _cycle_length(cal)
    add_betweenness_norm(cal)
    pool = cal[cal["gamma"].isin([0.1, 0.95])]

    wpath = glob.glob(os.path.join(
        ROOT, "runs", "CalvanoWideGridEnv-*", "analysis", "analysis_results.csv"))[0]
    wide = pd.read_csv(wpath).rename(columns={"avg_delta": "delta"})
    _cycle_length(wide)
    add_betweenness_norm(wide)

    headers = [r"Baseline\,+\,low-$\gamma$", "Wider grid"]
    rho_cols = [corr_column(pool, keys=keys), corr_column(wide, keys=keys)]
    delta_bars = [pool["delta"].mean(), wide["delta"].mean()]
    ns = [len(pool), len(wide)]

    tex = corr_table_tex(
        headers, rho_cols, delta_bars=delta_bars, ns=ns, metrics=metrics,
        caption=(r"Pearson correlations $\rho(\cdot,\Delta)$ for two further "
                 r"variations of the Calvano Q-learning environment: the "
                 r"baseline discount factor $\gamma=0.95$ pooled with the "
                 r"low-discount runs $\gamma=0.1$, and a wider grid of "
                 r"twenty-one prices trained for $2\times10^6$ periods at "
                 r"$\gamma=0.95$. Markers computed on the largest-WCC attractor."),
        label="tab:calvano-robustness")
    write_table("appendix_calvano_robustness.tex", tex)

    print(f"baseline+low-gamma: n={len(pool)}, Delta-bar={pool['delta'].mean():.3f}")
    print(f"wider grid:         n={len(wide)}, Delta-bar={wide['delta'].mean():.3f}")


if __name__ == "__main__":
    main()
