# Auditing algorithmic collusion from strategy graphs

Simulation and analysis code for the Q-learning results in Eschenbaum and Meylahn (2026), Auditing Algorithmic Collusion from Strategy Graphs, arXiv:2608.07098.

The agents are the Q-learning algorithm of Calvano et al. (2020). The agent and the environment port the Haskell implementation in open-games-learning by Philipp Zahn.


## Files

* `env.py`, `agent.py`: pricing environments and Q-learning agent
* `main.py`: training runs and rematching of independently trained policies (with `rematch.py`)
* `run_calvano_gamma_sweep.py`: runs the discount-factor sweep, from training to the summary tables
* `analysis_utils.py`, `categorize_convergence.py`: strategy graphs and graph metrics
* `analysis-1ststage.py`: analysis of the training runs
* `analyze_rematch.py`, `analysis-2ndstage.py`: analysis of the rematched pairs
* `summarize_calvano_gamma_sweep.py`, `robustness_tables.py`, `paper_common.py`: tables for the paper
* `verify_grid.py`: checks the price grid
