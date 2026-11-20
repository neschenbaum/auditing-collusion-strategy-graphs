import numpy as np
from collections import Counter

# Constants
CONVERGENCE_THRESHOLD = 0.80  # 80% threshold
MIN_ITERATIONS = 2  # Exclude random start iterations

def find_repeated_sequences(actions, max_seq_length=15):
    """
    Efficiently find repeated sequences in action array.
    Returns (sequences, seq_len) tuple where sequences that appear frequently (>7 times) 
    and explain ≥80% of actions. 
    """
    n = len(actions)
    
    for seq_len in range(1, min(max_seq_length + 1, n // 2 + 1)):
        # Create sliding windows efficiently with numpy
        if seq_len == 1:
            # Special case: just count individual actions
            counts = Counter(actions)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= CONVERGENCE_THRESHOLD * n:
                # Return as (result, seq_len) tuple
                return [(most_common[0], most_common[0], most_common[1])], 1
        else:
            # Use stride tricks for efficient sliding window
            windows = np.lib.stride_tricks.sliding_window_view(actions, seq_len)
            
            # Convert to tuples for hashing
            sequences = [tuple(w) for w in windows]
            counts = Counter(sequences)
            
            # Filter sequences that appear > 7 times
            frequent_seqs = {seq: count for seq, count in counts.items() if count > 7}
            
            if not frequent_seqs:
                continue
                
            # Check if these sequences explain ≥80% of actions
            total_coverage = sum(frequent_seqs.values()) * seq_len
            if total_coverage >= CONVERGENCE_THRESHOLD * n:
                # Check if counts are similar (within 2) - indicates true cycle
                seq_counts = list(frequent_seqs.values())
                if len(seq_counts) > 1 and max(seq_counts) - min(seq_counts) <= 2:
                    # Return the sequences as transitions (for cycle detection)
                    transitions = []
                    for seq, count in frequent_seqs.items():
                        for i in range(len(seq) - 1):
                            transitions.append((seq[i], seq[i+1], count))
                    return transitions, seq_len
                elif len(seq_counts) == 1:
                    # Single repeated sequence
                    seq = list(frequent_seqs.keys())[0]
                    transitions = [(seq[i], seq[i+1], frequent_seqs[seq]) 
                                  for i in range(len(seq) - 1)]
                    if len(transitions) == 0:  # Length 1 sequence
                        return [(seq[0], seq[0], frequent_seqs[seq])], 1
                    return transitions, seq_len
    
    return None, 0

def categorize_from_play_data(df, env):
    """
    Categorize equilibrium type from actual play data.
    Returns: (category, period, is_symmetric, p1_cycle, p2_cycle)
    
    Categories:
    - "sym1": Both players converge to same single action
    - "asym1": Players converge to different single actions
    - "cyc_N": Players cycle with period N
    - "other": No clear convergence pattern
    """
    # Exclude early random iterations
    df_conv = df[df['iteration'] > MIN_ITERATIONS].copy()
    
    if len(df_conv) == 0:
        return "other", 1, False, None, None
    
    # Get action indices for both players
    p1_actions = df_conv['p1_idx'].values
    p2_actions = df_conv['p2_idx'].values
    n_iters = len(df_conv)
    
    # Count action frequencies
    p1_counts = Counter(p1_actions)
    p2_counts = Counter(p2_actions)
    p1_most_common = p1_counts.most_common(1)[0] if p1_counts else (None, 0)
    p2_most_common = p2_counts.most_common(1)[0] if p2_counts else (None, 0)
    
    # Check for single-action convergence (sym1 or asym1)
    p1_converged = p1_most_common[1] >= CONVERGENCE_THRESHOLD * n_iters
    p2_converged = p2_most_common[1] >= CONVERGENCE_THRESHOLD * n_iters
    
    if p1_converged and p2_converged:
        if p1_most_common[0] == p2_most_common[0]:
            return "sym1", 1, True, [p1_most_common[0]], [p2_most_common[0]]
        else:
            return "asym1", 1, False, [p1_most_common[0]], [p2_most_common[0]]
    
    # Look for cycles in player 1
    p1_result, p1_seq_len = find_repeated_sequences(p1_actions)
    
    if p1_result is not None:
        # Check if player 2 also cycles
        p2_result, p2_seq_len = find_repeated_sequences(p2_actions)
        
        if p2_result is not None and p1_seq_len == p2_seq_len:
            # Both cycle with same period
            if isinstance(p1_result, list) and isinstance(p1_result[0], tuple) and len(p1_result[0]) == 3:
                period = len(p1_result)
                p1_cycle = [t[0] for t in p1_result] + [p1_result[-1][1]]
                p2_cycle = [t[0] for t in p2_result] + [p2_result[-1][1]]
            else:
                period = 1
                p1_cycle = [p1_result[0][0]]
                p2_cycle = [p2_result[0][0]] if isinstance(p2_result[0], tuple) else [p2_result[0]]
            
            is_symmetric = p1_cycle == p2_cycle
            return f"cyc_{period}", period, is_symmetric, p1_cycle, p2_cycle
        else:
            # Only p1 cycles
            if isinstance(p1_result, list) and len(p1_result) > 0:
                period = len(p1_result) if isinstance(p1_result[0], tuple) and len(p1_result[0]) == 3 else 1
            else:
                period = 1
            return f"cyc_{period}", period, False, None, None
    
    # Try player 2 if player 1 didn't cycle
    p2_result, p2_seq_len = find_repeated_sequences(p2_actions)
    
    if p2_result is not None:
        if isinstance(p2_result, list) and len(p2_result) > 0:
            period = len(p2_result) if isinstance(p2_result[0], tuple) and len(p2_result[0]) == 3 else 1
        else:
            period = 1
        return f"cyc_{period}", period, False, None, None
    
    # No clear pattern
    return "other", 0, False, None, None

def simplify_category(cat):
    """Simplify category names for plotting."""
    if cat in ['sym1', 'asym1']:
        return cat
    elif cat.startswith('cyc_'):
        try:
            period = int(cat.split('_')[1].split()[0])
            if period <= 3:
                return 'Short Cycle (≤3)'
            else:
                return 'Long Cycle (>3)'
        except:
            return 'Other'
    else:
        return 'Other'