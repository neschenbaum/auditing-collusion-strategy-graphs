"""
Rematch-Relearn experiment module.
Supports pairing trained agents and running test/relearn phases.

Phase 2 (Test): Frozen Q-tables, no exploration (epsilon=0), no learning
Phase 3 (Relearn): Resume learning with new opponent, epsilon reset to 1.0
"""

import os
import glob
import numpy as np
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
from env import CalvanoEnv, CalvanoAdjPriceGridEnv, NoisyObservationEnv
import json
from agent import QLearningAgent


def get_available_runs(exp_dir):
    """
    Get list of available run IDs from an experiment directory.
    
    Returns:
        List of run IDs (integers)
    """
    csv_files = glob.glob(os.path.join(exp_dir, "run_*.csv"))
    run_ids = []
    for f in csv_files:
        basename = os.path.basename(f)
        # Extract run ID from "run_X.csv"
        try:
            run_id = int(basename.split("_")[1].split(".")[0])
            # Check Q-tables exist
            q0 = os.path.join(exp_dir, f"run_{run_id}_qtable_p0.npy")
            q1 = os.path.join(exp_dir, f"run_{run_id}_qtable_p1.npy")
            if os.path.exists(q0) and os.path.exists(q1):
                run_ids.append(run_id)
        except (IndexError, ValueError):
            continue
    return sorted(run_ids)


def generate_cross_run_pairs(run_ids, num_pairs=None):
    """
    Generate random cross-run pairings for rematch.
    
    Args:
        run_ids: List of available run IDs
        num_pairs: Number of pairs to generate (default: len(run_ids))
    
    Returns:
        List of (run_id_p1, run_id_p2) tuples where p1 != p2
    """
    if len(run_ids) < 2:
        raise ValueError("Need at least 2 runs for cross-run rematching")
    
    if num_pairs is None:
        num_pairs = len(run_ids)
    
    pairs = []
    for _ in range(num_pairs):
        # Pick two different runs
        r1, r2 = random.sample(run_ids, 2)
        pairs.append((r1, r2))
    
    return pairs


def load_trained_agent(exp_dir, run_id, player_id, env):
    """
    Load a trained agent from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        run_id: Run ID to load
        player_id: 0 or 1
        env: CalvanoEnv instance (for num_actions)
    
    Returns:
        QLearningAgent with loaded Q-table
    """
    q_path = os.path.join(exp_dir, f"run_{run_id}_qtable_p{player_id}.npy")
    agent = QLearningAgent.from_trained(q_path, env.num_actions, player_id)
    return agent


def run_rematch_phase(agent1, agent2, env, iterations, mode='test', track_window=100):
    """
    Run a rematch phase between two trained agents.
    
    Args:
        agent1, agent2: QLearningAgent instances with loaded Q-tables
        env: CalvanoEnv instance
        iterations: Number of iterations to run
        mode: 'test' (no learning, no exploration) or 'relearn' (resume learning)
        track_window: Number of final iterations to record
    
    Returns:
        DataFrame with phase results (last track_window iterations)
    """
    # Configure agents based on mode
    if mode == 'test':
        agent1.freeze()
        agent2.freeze()
    elif mode == 'relearn':
        agent1.unfreeze(reset_epsilon=1.0)
        agent2.unfreeze(reset_epsilon=1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Reset environment with random initial state
    obs = env.reset()
    
    # Storage for tracking window
    history = []
    
    for i in range(iterations):
        # Actions (explore only if in relearn mode)
        a1 = agent1.act(obs[0], explore=(mode == 'relearn'))
        a2 = agent2.act(obs[1], explore=(mode == 'relearn'))
        actions = {0: a1, 1: a2}
        
        # Step
        next_obs, rewards, _, _ = env.step(actions)
        
        # Update (only has effect if learn_enabled=True)
        agent1.update(obs[0], a1, rewards[0], next_obs[0])
        agent2.update(obs[1], a2, rewards[1], next_obs[1])
        
        obs = next_obs
        
        # Capture metrics for convergence window
        if i >= iterations - track_window:
            history.append({
                "iteration": i,
                "p1_idx": a1,
                "p2_idx": a2,
                "p1_price": env.price_grid[a1],
                "p2_price": env.price_grid[a2],
                "p1_profit": rewards[0],
                "p2_profit": rewards[1]
            })
    
    return pd.DataFrame(history), agent1, agent2


def run_single_rematch(args):
    """
    Worker function for parallel rematch execution.
    
    Args is tuple: (exp_dir, run_id_p1, run_id_p2, test_iters, relearn_iters, 
                    output_dir, track_window)
    """
    (exp_dir, run_id_p1, run_id_p2, test_iters, relearn_iters, 
     output_dir, track_window) = args
    
    # Load config to determine environment
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        env_type = config.get("env_type", "CalvanoEnv")
        if env_type == "NoisyObservationEnv":
            noise_std = config.get("noise_std", 0.0)
            env = NoisyObservationEnv(noise_std=noise_std)
        elif env_type == "CalvanoAdjPriceGridEnv":
            env = CalvanoAdjPriceGridEnv()
        else:
            env = CalvanoEnv()
    else:
        # Fallback for old experiments without config
        print(f"Warning: No config.json found in {exp_dir}, defaulting to CalvanoEnv")
        env = CalvanoEnv()

    pair_name = f"pair_{run_id_p1}_{run_id_p2}"
    
    # Load agents (make copies for each phase)
    agent1_base = load_trained_agent(exp_dir, run_id_p1, 0, env)
    agent2_base = load_trained_agent(exp_dir, run_id_p2, 1, env)
    
    results = {"pair": pair_name, "run_p1": run_id_p1, "run_p2": run_id_p2}
    
    # === Phase 2: Test (no learning) ===
    if test_iters > 0:
        agent1_test = agent1_base.clone()
        agent2_test = agent2_base.clone()
        
        df_test, _, _ = run_rematch_phase(
            agent1_test, agent2_test, env, 
            test_iters, mode='test', track_window=track_window
        )
        
        # Save Phase 2 results
        phase2_dir = os.path.join(output_dir, "phase2")
        os.makedirs(phase2_dir, exist_ok=True)
        df_test.to_csv(os.path.join(phase2_dir, f"{pair_name}.csv"), index=False)
        
        # Compute Phase 2 metrics
        results["phase2_mean_p1_profit"] = df_test["p1_profit"].mean()
        results["phase2_mean_p2_profit"] = df_test["p2_profit"].mean()
    
    # === Phase 3: Relearn ===
    if relearn_iters > 0:
        agent1_relearn = agent1_base.clone()
        agent2_relearn = agent2_base.clone()
        
        df_relearn, agent1_final, agent2_final = run_rematch_phase(
            agent1_relearn, agent2_relearn, env,
            relearn_iters, mode='relearn', track_window=track_window
        )
        
        # Save Phase 3 results
        phase3_dir = os.path.join(output_dir, "phase3")
        os.makedirs(phase3_dir, exist_ok=True)
        df_relearn.to_csv(os.path.join(phase3_dir, f"{pair_name}.csv"), index=False)
        
        # Save final Q-tables
        agent1_final.save_q_table(os.path.join(phase3_dir, f"{pair_name}_qtable_p0.npy"))
        agent2_final.save_q_table(os.path.join(phase3_dir, f"{pair_name}_qtable_p1.npy"))
        
        # Compute Phase 3 metrics
        results["phase3_mean_p1_profit"] = df_relearn["p1_profit"].mean()
        results["phase3_mean_p2_profit"] = df_relearn["p2_profit"].mean()
    
    return results


def run_rematch_experiment(training_dir, num_pairs=None, test_iters=1000, 
                           relearn_iters=100000, track_window=100, workers=None):
    """
    Run complete rematch experiment with cross-run pairings.
    
    Args:
        training_dir: Path to training experiment directory
        num_pairs: Number of cross-run pairs (default: number of runs)
        test_iters: Iterations for Phase 2 (test, no learning)
        relearn_iters: Iterations for Phase 3 (relearn)
        track_window: Final iterations to track
        workers: Number of parallel workers
    
    Returns:
        DataFrame with summary results for all pairs
    """
    if workers is None:
        workers = os.cpu_count()
    
    # Get available runs
    run_ids = get_available_runs(training_dir)
    if len(run_ids) < 2:
        raise ValueError(f"Need at least 2 training runs, found {len(run_ids)}")
    
    print(f"Found {len(run_ids)} training runs: {run_ids}")
    
    # Generate cross-run pairs
    pairs = generate_cross_run_pairs(run_ids, num_pairs)
    print(f"Generated {len(pairs)} cross-run pairs")
    
    # Create output directory
    output_dir = os.path.join(training_dir, "rematch")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rematch config
    import json
    config = {
        "training_dir": training_dir,
        "pairs": pairs,
        "test_iters": test_iters,
        "relearn_iters": relearn_iters,
        "track_window": track_window
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare arguments for parallel execution
    run_args = [
        (training_dir, p1, p2, test_iters, relearn_iters, output_dir, track_window)
        for p1, p2 in pairs
    ]
    
    # Run in parallel
    print(f"Running {len(pairs)} rematch experiments with {workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_rematch, arg) for arg in run_args]
        
        for i, future in enumerate(futures):
            res = future.result()
            results.append(res)
            print(f"Completed pair {i+1}/{len(pairs)}: {res['pair']}")
    
    # Save summary
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "rematch_summary.csv"), index=False)
    
    print(f"\nRematch experiment complete. Results saved to '{output_dir}'")
    
    return df_results


if __name__ == "__main__":
    # Simple test
    import argparse
    parser = argparse.ArgumentParser(description="Run rematch experiment")
    parser.add_argument("training_dir", help="Path to training experiment directory")
    parser.add_argument("--pairs", type=int, default=None, help="Number of pairs")
    parser.add_argument("--test-iters", type=int, default=1000)
    parser.add_argument("--relearn-iters", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    run_rematch_experiment(
        args.training_dir,
        num_pairs=args.pairs,
        test_iters=args.test_iters,
        relearn_iters=args.relearn_iters,
        workers=args.workers
    )
