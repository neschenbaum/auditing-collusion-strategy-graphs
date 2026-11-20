"""
Shared definitions for paper table/figure generation.

Every scenario script (Calvano gamma sweep, Janusz theory equilibria, DecQ)
writes a standardized per-run/per-equilibrium results CSV with a `delta`
column plus the six canonical marker columns below, and emits its own appendix
LaTeX table via `corr_table_tex`.  `make_paper_content.py` assembles the
main-body tables and figures from those CSVs.

All LaTeX output goes to paper_content/tables and paper_content/figures.
"""

import os

import numpy as np
import pandas as pd

# Canonical marker order and LaTeX labels (matches the slide tables).
METRICS = [
    ("max_betweenness", r"Max.\ betweenness centrality"),
    ("attractor_indegree_centrality", r"In-degree centrality"),
    ("num_attractors", r"Number of attractors"),
    ("basin_fraction", r"Basin fraction"),
    ("attractor_closeness", r"Closeness centrality"),
    ("avg_path_length", r"Average path length"),
]
METRIC_KEYS = [k for k, _ in METRICS]

# Scale-free companion to raw betweenness: the funnel load of the busiest
# transient node divided by the maximum possible given the largest-WCC size,
# b_max / (|basin| - cycle_length).  Comparable across grids of different size.
BETWEENNESS_NORM = ("betweenness_norm", r"Max.\ betweenness (norm.)")


def metrics_with_norm():
    """METRICS with the normalized betweenness inserted after the raw one."""
    out = []
    for m in METRICS:
        out.append(m)
        if m[0] == "max_betweenness":
            out.append(BETWEENNESS_NORM)
    return out

TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "paper_content", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "paper_content", "figures")


def corr(df, col, delta_col="delta"):
    """Pearson correlation of a marker column with the collusion index."""
    if col not in df.columns:
        return float("nan")
    x, y = df[col], df[delta_col]
    if len(df) < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(df[[col, delta_col]].corr().iloc[0, 1])


def corr_column(df, delta_col="delta", keys=None):
    """dict metric_key -> rho for one dataset."""
    keys = METRIC_KEYS if keys is None else keys
    return {k: corr(df, k, delta_col) for k in keys}


def add_betweenness_norm(df):
    """Add b_max / (|basin| - cycle_length), the scale-free betweenness.

    Requires max_betweenness, basin_size and period (cycle length) columns;
    a no-op otherwise.  Undefined (NaN) when the basin has no transients.
    """
    if {"max_betweenness", "basin_size", "period"} <= set(df.columns):
        transients = df["basin_size"].astype(float) - df["period"].astype(float)
        df["betweenness_norm"] = np.where(
            transients > 0, df["max_betweenness"] / transients, np.nan)
    return df


def _fmt_rho(v, bold=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = r"--"
        return s
    s = f"{v:+.3f}"
    return rf"$\mathbf{{{s}}}$" if bold else rf"${s}$"


def corr_table_tex(col_headers, rho_cols, delta_bars=None, ns=None,
                   caption="", label="", bold_cols=(), metrics=None):
    """
    Build a booktabs correlation table.

    Args:
        col_headers: list of column header strings (LaTeX-safe)
        rho_cols:    list of dicts metric_key -> rho, one per column
        delta_bars:  optional list of mean Delta per column
        ns:          optional list of N per column
        bold_cols:   indices of columns whose rho values are bolded
    """
    ncol = len(col_headers)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    if ncol > 4:
        lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l" + "c" * ncol + "}")
    lines.append(r"\toprule")
    lines.append(rf" & \multicolumn{{{ncol}}}{{c}}{{$\rho(\cdot, \Delta)$}} \\")
    lines.append(rf"\cmidrule(lr){{2-{ncol + 1}}}")
    lines.append("Metric & " + " & ".join(col_headers) + r" \\")
    lines.append(r"\midrule")
    for key, lbl in (METRICS if metrics is None else metrics):
        cells = [_fmt_rho(rc.get(key), bold=(i in bold_cols))
                 for i, rc in enumerate(rho_cols)]
        lines.append(lbl + " & " + " & ".join(cells) + r" \\")
    if delta_bars is not None or ns is not None:
        lines.append(r"\midrule")
    if delta_bars is not None:
        cells = [rf"${v:.3f}$" if v == v else "--" for v in delta_bars]
        lines.append(r"$\bar\Delta$ & " + " & ".join(cells) + r" \\")
    if ns is not None:
        lines.append(r"$N$ & " + " & ".join(str(n) for n in ns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(filename, tex):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"saved {path}")
    return path
