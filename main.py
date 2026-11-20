import numpy as np
import pandas as pd
import os
import time
import argparse
import hashlib
import json
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from env import CalvanoEnv, CalvanoAdjPriceGridEnv, CalvanoWideGridEnv
from agent import QLearningAgent

# Experiment configuration
EXPERIMENT_CONFIG = {
    "name": "CalvanoAdjPriceGridEnv",
    "iterations": 1000000,
    "track_window": 100,
    # Environment parameters
    "a0": 0.0,
    "a1": 2.0,
    "a2": 2.0,
    "mu": 0.25,
    "c1": 1.0,
    "c2": 1.0,
    "p_bertrand": 1.47,
    "p_monopoly": 1.92,
    "xi": 0.1,
    "m": 14,
    # Agent parameters
    "alpha": 0.15,
    "gamma": 0.95,
    "beta": 0.000004,
}

def get_experiment_hash(config):
    """
    Generate a short hash from the experiment configuration.
    This ensures runs with identical parameters go to the same directory.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]

def get_experiment_dir(config, base_dir="runs"):
    """
    Get or create the experiment directory with format: {name}-{hash}
    Example: Calvano-a1b2c3d4
    """
    exp_hash = get_experiment_hash(config)
    exp_name = f"{config['name']}-{exp_hash}"
    exp_dir = os.path.join(base_dir, exp_name)
    return exp_dir, exp_name

def save_experiment_config(config, exp_dir):
    """Save the experiment configuration to the directory."""
    config_path = os.path.join(exp_dir, "config.json")
    # Add timestamp
    config_with_meta = {
        **config,
        "created_at": datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(config_with_meta, f, indent=2)

def run_single_experiment(args):
    """
    Executes a single full learning run and saves raw data.
    Args is a tuple: (run_id, iterations, output_dir, track_window, env_name, gamma)

    Note: env_name and gamma are passed explicitly (not read from the global
    EXPERIMENT_CONFIG) because on Windows ProcessPoolExecutor uses spawn, which
    re-imports this module and would only see the module-level config defaults.
    """
    run_id, iterations, output_dir, track_window, env_name, gamma = args

    # Initialize fresh environment and agents
    if env_name == "CalvanoAdjPriceGridEnv":
        env = CalvanoAdjPriceGridEnv()
    elif env_name == "CalvanoWideGridEnv":
        env = CalvanoWideGridEnv()
    else:
        env = CalvanoEnv()
    agent1 = QLearningAgent(env.num_actions, player_id=0, env=env, gamma=gamma)
    agent2 = QLearningAgent(env.num_actions, player_id=1, env=env, gamma=gamma)
    
    # Run Learning Phase
    obs = env.reset()
    
    # Storage for last N rounds
    history = []
    
    start_time = time.time()
    
    for i in range(iterations):
        # Actions
        a1 = agent1.act(obs[0], explore=True)
        a2 = agent2.act(obs[1], explore=True)
        actions = {0: a1, 1: a2}
        
        # Step
        next_obs, rewards, _, _ = env.step(actions)
        
        # Update
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

    duration = time.time() - start_time
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, f"run_{run_id}.csv")
    df_hist.to_csv(csv_path, index=False)
    
    # Save Q-Tables
    np.save(os.path.join(output_dir, f"run_{run_id}_qtable_p0.npy"), agent1.q_table)
    np.save(os.path.join(output_dir, f"run_{run_id}_qtable_p1.npy"), agent2.q_table)
    
    return {
        "run_id": run_id,
        "duration_sec": duration
    }

def get_latest_experiment_dir(base_dir="runs"):
    """
    Find the most recently modified experiment subdirectory.
    Returns (path, name) or (None, None) if no subdirectories exist.
    """
    if not os.path.exists(base_dir):
        return None, None
    
    subdirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            # Check if it has our naming pattern (Name-hash)
            if '-' in name:
                mtime = os.path.getmtime(path)
                subdirs.append((path, name, mtime))
    
    if not subdirs:
        return None, None
    
    # Sort by modification time, most recent first
    subdirs.sort(key=lambda x: x[2], reverse=True)
    return subdirs[0][0], subdirs[0][1]

def main():
    parser = argparse.ArgumentParser(description="Calvano Replication Experiments")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training experiments')
    train_parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    train_parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    train_parser.add_argument("--iterations", type=int, default=EXPERIMENT_CONFIG["iterations"],
                             help="Number of iterations per run")
    train_parser.add_argument("--gamma", type=float, default=None,
                             help="Discount factor (default: config value)")
    train_parser.add_argument("--env", type=str, default=None,
                             choices=["Calvano", "CalvanoAdjPriceGridEnv", "CalvanoWideGridEnv"],
                             help="Environment / experiment name (default: config value)")
    
    # Rematch command
    rematch_parser = subparsers.add_parser('rematch', help='Run rematch experiments')
    rematch_parser.add_argument("--training-dir", type=str, default=None,
                               help="Training experiment directory (default: latest)")
    rematch_parser.add_argument("--pairs", type=int, default=None,
                               help="Number of cross-run pairs (default: number of runs)")
    rematch_parser.add_argument("--test-iters", type=int, default=1000,
                               help="Iterations for test phase (Phase 2)")
    rematch_parser.add_argument("--relearn-iters", type=int, default=100000,
                               help="Iterations for relearn phase (Phase 3)")
    rematch_parser.add_argument("--workers", type=int, default=os.cpu_count(),
                               help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Default to train if no command specified (backwards compatibility)
    if args.command is None or args.command == 'train':
        run_training(args)
    elif args.command == 'rematch':
        run_rematch(args)


def run_training(args):
    """Run training experiments (Phase 1)."""
    # Update config with command-line overrides
    config = EXPERIMENT_CONFIG.copy()
    if hasattr(args, 'iterations') and args.iterations:
        config["iterations"] = args.iterations
    if getattr(args, 'env', None):
        config["name"] = args.env
    if getattr(args, 'gamma', None) is not None:
        config["gamma"] = args.gamma
    
    # Get experiment directory
    exp_dir, exp_name = get_experiment_dir(config)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config (only if it doesn't exist yet)
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        save_experiment_config(config, exp_dir)
    
    runs = getattr(args, 'runs', 1)
    workers = getattr(args, 'workers', os.cpu_count())
    
    # Smart Execution: Check for existing runs to avoid redundant computation
    existing_runs = len(glob.glob(os.path.join(exp_dir, "run_*.csv")))
    if existing_runs >= runs:
        print(f"Experiment: {exp_name}")
        print(f"Directory: {exp_dir}")
        print(f"Found {existing_runs} existing runs (requested {runs}). Skipping training.")
        return

    print(f"Experiment: {exp_name}")
    print(f"Output directory: {exp_dir}")
    print(f"Starting simulation with {runs} runs using {workers} workers...")
    print(f"Iterations per run: {config['iterations']}")
    
    start_total = time.time()
    
    # Prepare arguments for parallel execution
    run_args = [
        (i, config["iterations"], exp_dir, config["track_window"],
         config["name"], config["gamma"])
        for i in range(runs)
    ]
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_experiment, arg) for arg in run_args]
        
        for i, future in enumerate(futures):
            res = future.result()
            print(f"Run {res['run_id']} finished ({res['duration_sec']:.2f}s)")
            
    total_time = time.time() - start_total
    print(f"\nAll runs completed in {total_time:.2f}s")
    print(f"Raw data saved to '{exp_dir}/'")


def run_rematch(args):
    """Run rematch experiments (Phase 2 and 3)."""
    from rematch import run_rematch_experiment
    
    # Determine training directory
    if args.training_dir:
        training_dir = args.training_dir
        if not os.path.isabs(training_dir):
            training_dir = os.path.join("runs", training_dir)
    else:
        # Use latest experiment
        training_dir, exp_name = get_latest_experiment_dir()
        if training_dir is None:
            print("No training experiments found. Run 'python main.py train' first.")
            return
        print(f"Using latest experiment: {exp_name}")
    
    if not os.path.exists(training_dir):
        print(f"Error: Training directory '{training_dir}' does not exist.")
        return
    
    run_rematch_experiment(
        training_dir,
        num_pairs=args.pairs,
        test_iters=args.test_iters,
        relearn_iters=args.relearn_iters,
        workers=args.workers
    )


if __name__ == "__main__":
    main()

