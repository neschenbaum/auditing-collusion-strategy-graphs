import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import deque
from env import CalvanoEnv
from categorize_convergence import categorize_from_play_data, simplify_category

# Constants matching main.py
THEORETICAL_BERTRAND_P = 1.47
THEORETICAL_MONOPOLY_P = 1.92

def calculate_theoretical_profit(env, p1, p2):
    """Calculates profit for symmetric costs given generic prices p1, p2."""
    exp_1 = np.exp((env.a1 - p1) / env.mu)
    exp_2 = np.exp((env.a2 - p2) / env.mu)
    exp_0 = np.exp(env.a0 / env.mu)
    denominator = exp_1 + exp_2 + exp_0
    
    demand_1 = exp_1 / denominator
    profit_1 = (p1 - env.c1) * demand_1
    return profit_1

def get_nash_index(env):
    """
    Find the price grid index closest to the Bertrand (Nash) price.
    """
    idx = (np.abs(env.price_grid - env.p_bertrand)).argmin()
    return idx

def get_policy_action(q_table, state_idx):
    """Returns the greedy action for a given state tuple (p1_idx, p2_idx)."""
    p1, p2 = state_idx
    return np.argmax(q_table[p1, p2, :])

def build_policy_graph(env, q_p0, q_p1):
    """
    Constructs the full state transition graph based on greedy policies.
    Returns: NetworkX DiGraph
    """
    G = nx.DiGraph()
    num_p = env.num_actions
    
    # Iterate over all possible states
    states = [(p1, p2) for p1 in range(num_p) for p2 in range(num_p)]
    
    for state in states:
        a0 = get_policy_action(q_p0, state)
        a1 = get_policy_action(q_p1, state)
        next_state = (a0, a1)
        G.add_edge(state, next_state)
        
    return G

def find_attractors(G):
    """
    Finds all attracting components (cycles and fixed points) in the graph.
    Returns a list of sets, where each set contains the nodes of an attractor.
    """
    return list(nx.attracting_components(G))

def largest_wcc_attractor(G, tie_break=None):
    """
    Attractor of the largest weakly-connected component (largest basin).

    This is the paper's convention: all markers are computed on the largest
    WCC, whose unique cycle is "the attractor".  It is the outcome that governs
    the expected price under the joint policy and coincides with the realized
    attractor in converged runs.

    Args:
        G: policy graph (functional relation)
        tie_break: optional function attractor_set -> sortable value used to
                   break basin-size ties (e.g. mean collusion index)

    Returns:
        (attractor_nodes: set, num_attractors: int)
    """
    atts = find_attractors(G)

    def key(a):
        size, _ = analyze_basin(G, set(a))
        return (size, tie_break(a)) if tie_break is not None else size

    best = max(atts, key=key)
    return set(best), len(atts)

def analyze_basin(G, attractor_nodes):
    """
    Analyzes the basin of attraction for a specific set of attractor nodes.
    Returns:
        basin_size (int): Number of nodes in the basin (inclusive).
        avg_path_length (float): Average steps to reach the attractor from within the basin.
    """
    G_rev = G.reverse()
    queue = deque([(n, 0) for n in attractor_nodes])
    visited_bfs = set(attractor_nodes)
    total_path_len = 0
    
    while queue:
        node, dist = queue.popleft()
        total_path_len += dist
        
        for neighbor in G_rev.neighbors(node):
            if neighbor not in visited_bfs:
                visited_bfs.add(neighbor)
                queue.append((neighbor, dist + 1))
                
    basin_size = len(visited_bfs)
    avg_path_length = total_path_len / basin_size if basin_size > 0 else 0
    
    return basin_size, avg_path_length

def compute_path_variance(G, attractor_nodes):
    """
    Computes the variance of path lengths to the attractor across all states in its basin.
    
    Returns:
        path_variance (float): Variance of path lengths
        path_std (float): Standard deviation of path lengths
    """
    G_rev = G.reverse()
    queue = deque([(n, 0) for n in attractor_nodes])
    visited_bfs = set(attractor_nodes)
    path_lengths = []
    
    while queue:
        node, dist = queue.popleft()
        path_lengths.append(dist)
        
        for neighbor in G_rev.neighbors(node):
            if neighbor not in visited_bfs:
                visited_bfs.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    if len(path_lengths) > 1:
        path_variance = np.var(path_lengths)
        path_std = np.std(path_lengths)
    else:
        path_variance = 0.0
        path_std = 0.0
    
    return path_variance, path_std

def count_bottleneck_states(G, attractor_nodes, threshold_percentile=75):
    """
    Counts the number of bottleneck states in the transient graph leading to the attractor.
    A bottleneck is defined as a state with betweenness centrality above a threshold.
    
    Args:
        G: Full policy graph
        attractor_nodes: Set of nodes in the attractor
        threshold_percentile: Percentile above which a state is considered a bottleneck
    
    Returns:
        num_bottlenecks (int): Count of bottleneck states
        mean_betweenness (float): Mean betweenness of transient states
    """
    # Find the basin (all states that lead to this attractor)
    G_rev = G.reverse()
    queue = deque(attractor_nodes)
    visited_bfs = set(attractor_nodes)
    
    while queue:
        node = queue.popleft()
        for neighbor in G_rev.neighbors(node):
            if neighbor not in visited_bfs:
                visited_bfs.add(neighbor)
                queue.append(neighbor)
    
    # Transient states = basin - attractor
    transient_states = visited_bfs - set(attractor_nodes)
    
    if len(transient_states) == 0:
        return 0, 0.0
    
    # Compute betweenness centrality for the subgraph of basin
    basin_subgraph = G.subgraph(visited_bfs)
    
    try:
        betweenness = nx.betweenness_centrality(basin_subgraph)
        
        # Extract betweenness for transient states only
        transient_betweenness = [betweenness[node] for node in transient_states]
        
        if len(transient_betweenness) > 0:
            threshold_value = np.percentile(transient_betweenness, threshold_percentile)
            num_bottlenecks = sum(1 for b in transient_betweenness if b >= threshold_value)
            mean_betweenness = np.mean(transient_betweenness)
        else:
            num_bottlenecks = 0
            mean_betweenness = 0.0
            
    except Exception as e:
        # In case betweenness computation fails
        num_bottlenecks = 0
        mean_betweenness = 0.0
    
    return num_bottlenecks, mean_betweenness

def max_betweenness(G, attractor_nodes):
    """
    Largest betweenness centrality among transient nodes (formal definition):

        b(v) = |{ u in V : the path from u to the attractor passes through v }|

    Returns ( max_{v not in A} b(v), argmax_node ).

    In a functional relation paths are unique (out-degree 1), so b(v) is the
    size of v's in-subtree toward the attractor (v itself included) -- the
    number of nodes that funnel through v on the way to the attractor.  The
    returned value is the raw count for the busiest transient node.
    """
    G_rev = G.reverse()

    # BFS on reversed graph: discovers basin and gives reverse-topological order
    topo = []
    visited = set(attractor_nodes)
    queue = deque(list(attractor_nodes))
    while queue:
        node = queue.popleft()
        for pred in G_rev.neighbors(node):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
                topo.append(pred)

    transient = visited - set(attractor_nodes)
    if not transient:
        return 0.0, None

    subtree_size = {v: 1 for v in transient}

    # Process leaves-first (reversed BFS order), accumulate toward attractor
    for node in reversed(topo):
        succ = next(iter(G.neighbors(node)))  # unique out-edge
        if succ in subtree_size:
            subtree_size[succ] += subtree_size[node]

    bottleneck_node = max(subtree_size, key=subtree_size.__getitem__)
    return subtree_size[bottleneck_node], bottleneck_node


def analyze_run(run_path, q_p0_path, q_p1_path, env, pi_bertrand, pi_monopoly, run_id_override=None):
    """
    Analyzes a single run with graph-theoretic metrics.
    """
    if run_id_override:
        run_id = run_id_override
    else:
        run_id = os.path.basename(run_path).split('_')[1].split('.')[0]
    
    # 1. Load data
    df = pd.read_csv(run_path)
    
    # 2. Delta Calculation
    mean_profits = df[['p1_profit', 'p2_profit']].mean()
    p0_delta = (mean_profits["p1_profit"] - pi_bertrand) / (pi_monopoly - pi_bertrand)
    p1_delta = (mean_profits["p2_profit"] - pi_bertrand) / (pi_monopoly - pi_bertrand)
    avg_delta = (p0_delta + p1_delta) / 2
    
    # 3. Categorize from play data (Legacy check)
    category, period, is_symmetric, p1_cycle, p2_cycle = categorize_from_play_data(df, env)
    
    # 4. Build policy graph
    q_p0 = np.load(q_p0_path)
    q_p1 = np.load(q_p1_path)
    G = build_policy_graph(env, q_p0, q_p1)
    
    # 5. Find realized attractor by tracing from last state
    last_state_csv = df.iloc[-1]
    last_node = (int(last_state_csv["p1_idx"]), int(last_state_csv["p2_idx"]))
    
    # Trace forward until we hit a cycle
    curr = last_node
    visited_trace = set()
    
    # Safety limit - though policy graph is finite, we should be careful
    steps = 0
    max_steps = env.num_actions * env.num_actions + 10
    
    while curr not in visited_trace and steps < max_steps:
        visited_trace.add(curr)
        successors = list(G.successors(curr))
        if not successors:
            break
        curr = successors[0]
        steps += 1
        
    # Now extract the cycle
    cycle_start = curr
    realized_cycle = [cycle_start]
    
    if list(G.successors(curr)): # If not a dead end
        curr = list(G.successors(curr))[0]
        steps = 0
        while curr != cycle_start and steps < max_steps:
            realized_cycle.append(curr)
            successors = list(G.successors(curr))
            if not successors: 
                break
            curr = successors[0]
            steps += 1
            
    realized_attractor_nodes = set(realized_cycle)
    
    # 6. Count ALL attractors in the graph
    attractor_components = find_attractors(G)
    num_attractors = len(attractor_components)
    
    # 7. Basin analysis for realized attractor
    realized_basin_size, avg_path_length = analyze_basin(G, realized_attractor_nodes)
    total_states = env.num_actions * env.num_actions
    basin_fraction = realized_basin_size / total_states
    
    # 8. NEW: Path variance computation
    path_variance, path_std = compute_path_variance(G, realized_attractor_nodes)
    
    # 9. Bottleneck states computation
    num_bottlenecks, mean_betweenness = count_bottleneck_states(G, realized_attractor_nodes)
    max_betweenness_val, bottleneck_node = max_betweenness(G, realized_attractor_nodes)

    # Compare bottleneck prices to attractor prices
    attractor_mean_price = np.mean([
        (env.price_grid[s[0]] + env.price_grid[s[1]]) / 2
        for s in realized_cycle
    ])
    if bottleneck_node is not None:
        bottleneck_mean_price = (env.price_grid[bottleneck_node[0]] + env.price_grid[bottleneck_node[1]]) / 2
        bottleneck_below_attractor = bool(bottleneck_mean_price < attractor_mean_price)
    else:
        bottleneck_mean_price = float('nan')
        bottleneck_below_attractor = False
    
    # 10. Graph Centrality Metrics
    try:
        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        attractor_closeness = np.mean([closeness.get(n, 0) for n in realized_cycle])
        
        # In-degree centrality
        indegree_centrality = nx.in_degree_centrality(G)
        attractor_indegree_centrality = np.mean([indegree_centrality.get(n, 0) for n in realized_cycle])
    except:
        attractor_closeness = 0.0
        attractor_indegree_centrality = 0.0

    # 11. Raw In-degree
    indegrees = np.array([G.in_degree(n) for n in realized_cycle])
    avg_indegree = np.mean(indegrees)
    
    # 12. Nash Equilibrium Analysis
    nash_idx = get_nash_index(env)
    nash_state = (nash_idx, nash_idx)
    
    has_nash_attractor = False
    nash_attractor_mean_indegree = 0
    nash_basin_size = 0
    min_dist_to_nash = float('inf')
    nash_indegree = G.in_degree(nash_state) or 0
    
    for attractor in attractor_components:
        attractor_nodes = set(attractor)
        
        # Check if Nash is in this attractor
        if nash_state in attractor_nodes:
            has_nash_attractor = True
            nash_basin_size, _ = analyze_basin(G, attractor_nodes)
            nash_attractor_mean_indegree = np.mean([G.in_degree(n) for n in attractor_nodes])
        
        # Find minimum distance to any Nash-like attractor (Static Grid Distance)
        for node in attractor_nodes:
            p1_idx, p2_idx = node
            dist = max(abs(p1_idx - nash_idx), abs(p2_idx - nash_idx))
            if dist < min_dist_to_nash:
                min_dist_to_nash = dist
    
    # 13. Dynamic Distance (Nash -> Attractor)
    try:
        if nx.has_path(G, nash_state, realized_cycle[0]):
            dists = []
            for target in realized_cycle:
                try:
                    d = nx.shortest_path_length(G, nash_state, target)
                    dists.append(d)
                except nx.NetworkXNoPath:
                    pass
            
            nash_to_attractor_steps = min(dists) if dists else -1
        else:
            nash_to_attractor_steps = -1
            
    except Exception:
        nash_to_attractor_steps = -1

    # 14. Nash vs Attractor comparison
    nash_vs_attractor_indegree_ratio = avg_indegree / (nash_indegree + 1e-6)
    
    # 15. Get prices
    attractor_prices = [(env.price_grid[s[0]], env.price_grid[s[1]]) for s in realized_cycle]

    return {
        "run_id": run_id,
        "avg_delta": avg_delta,
        "p0_delta": p0_delta,
        "p1_delta": p1_delta,
        "category": category,
        "period": period,
        "is_symmetric": is_symmetric,
        "attractor_closeness": attractor_closeness,
        "attractor_indegree_centrality": attractor_indegree_centrality,
        "avg_indegree": avg_indegree,
        "basin_size": realized_basin_size,
        "basin_fraction": basin_fraction,
        "num_attractors": num_attractors,
        "avg_path_length": avg_path_length,
        "path_variance": path_variance,
        "path_std": path_std,
        "num_bottlenecks": num_bottlenecks,
        "mean_betweenness": mean_betweenness,
        "max_betweenness": max_betweenness_val,
        "bottleneck_mean_price": bottleneck_mean_price,
        "attractor_mean_price": attractor_mean_price,
        "bottleneck_below_attractor": bottleneck_below_attractor,
        "attractor_prices": attractor_prices,
        "has_nash_attractor": has_nash_attractor,
        "nash_attractor_mean_indegree": nash_attractor_mean_indegree,
        "nash_basin_size": nash_basin_size,
        "nash_indegree": nash_indegree,
        "nash_vs_attractor_indegree_ratio": nash_vs_attractor_indegree_ratio,
        "min_dist_to_nash": min_dist_to_nash,
        "nash_to_attractor_steps": nash_to_attractor_steps
    }

def plot_centrality_analysis(df, phase_name, filename_prefix, analysis_dir):
    """Create dual plots for closeness and in-degree centrality"""
    if df.empty: 
        return
    df = df.copy()
    if 'plot_category' not in df.columns:
        df['plot_category'] = df['category'].apply(simplify_category)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    color_map = {
        'sym1': '#2E86AB', 'asym1': '#A23B72', 
        'Short Cycle (≤3)': '#F18F01', 'Long Cycle (>3)': '#C73E1D', 'Other': '#6C757D'
    }
    
    # Handle missing categories in color map by defaulting
    point_colors = [color_map.get(cat, '#6C757D') for cat in df['plot_category']]
    
    # Plot 1: Closeness Centrality
    axes[0].scatter(df["avg_delta"], df["attractor_closeness"], 
               c=point_colors, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    axes[0].set_xlabel(r"Collusion Index $\Delta = \frac{\bar{\pi} - \pi_N}{\pi_M - \pi_N}$")
    axes[0].set_ylabel("Attractor Mean Closeness (Avg over Cycle)")
    axes[0].set_title(f"{phase_name}: Closeness vs Collusion")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: In-degree Centrality
    axes[1].scatter(df["avg_delta"], df["attractor_indegree_centrality"], 
               c=point_colors, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    axes[1].set_xlabel(r"Collusion Index $\Delta = \frac{\bar{\pi} - \pi_N}{\pi_M - \pi_N}$")
    axes[1].set_ylabel("Attractor Mean In-Degree Centrality (Avg over Cycle)")
    axes[1].set_title(f"{phase_name}: In-Degree Centrality vs Collusion")
    axes[1].grid(True, alpha=0.3)
    
    import matplotlib.patches as mpatches
    legend_elements = [mpatches.Patch(facecolor=color_map[cat], label=cat, edgecolor='black')
                      for cat in color_map.keys() if cat in df['plot_category'].unique()]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(analysis_dir, f"{filename_prefix}_centrality.png"), 
               dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{filename_prefix}_centrality.png'")
    plt.close()
