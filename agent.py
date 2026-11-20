import numpy as np
import copy

class QLearningAgent:
    """
    Q-Learning agent matching the Haskell CalvanoReplication implementation.
    - Q-table initialization: Uses expected profit estimates (lsValues in Haskell)
    - Exploration decay: Uses exponential formula e^(-β) per iteration
    - Q-learning update: Standard Bellman equation
    
    Supports freeze/unfreeze for rematch experiments:
    - freeze(): Disable exploration and learning (test mode)
    - unfreeze(): Re-enable exploration and learning (relearn mode)
    """
    def __init__(self, num_actions, player_id, env=None, gamma=0.95):
        self.player_id = player_id
        self.num_actions = num_actions

        # Parameters
        self.alpha = 0.15      # Learning Rate
        self.gamma = gamma     # Discount Factor
        self.beta = 0.000004   # Exploration Decay
        self.epsilon = 1.0     # Initial Exploration Rate
        
        # Control flags for rematch phases
        self.explore_enabled = True   # Whether exploration is allowed
        self.learn_enabled = True     # Whether Q-table updates are allowed
        self.decay_epsilon = True     # Whether epsilon decays each step
        
        # Q-Table Configuration
        # Dimensions: [State_P1, State_P2, Action]
        # State is (p1_prev_idx, p2_prev_idx)
        self.q_table = np.zeros((num_actions, num_actions, num_actions))
        
        # Initialize Q-table with profit estimates
        if env is not None:
            self._initialize_q_table(env)
        
    def _initialize_q_table(self, env):
        """
        Initializes Q-table using expected profit estimates.
        For each action (own price), compute average profit assuming uniform 
        opponent prices, then discount by (1-gamma) to get steady-state value.
        """
        n = self.num_actions
        
        # Compute expected profit for each of our actions (price indices)
        for action_idx in range(n):
            price = env.price_grid[action_idx]
            
            # Average profit over all possible opponent prices
            total_profit = 0.0
            for opp_idx in range(n):
                opp_price = env.price_grid[opp_idx]
                
                # Compute demand (player-specific based on player_id)
                exp_1 = np.exp((env.a1 - price) / env.mu)
                exp_2 = np.exp((env.a2 - opp_price) / env.mu)
                exp_0 = np.exp(env.a0 / env.mu)
                
                if self.player_id == 0:
                    # Player 1: our price affects demand_1
                    demand = exp_1 / (exp_1 + exp_2 + exp_0)
                    profit = (price - env.c1) * demand
                else:
                    # Player 2: swap roles - our price is p2, opponent is p1
                    exp_1_swap = np.exp((env.a1 - opp_price) / env.mu)
                    exp_2_swap = np.exp((env.a2 - price) / env.mu)
                    demand = exp_2_swap / (exp_1_swap + exp_2_swap + exp_0)
                    profit = (price - env.c2) * demand
                
                total_profit += profit
            
            avg_profit = total_profit / n
            # Steady-state value = expected_profit / (1 - gamma)
            initial_value = avg_profit / (1 - self.gamma)
            
            # Set this value for all state-action pairs with this action
            self.q_table[:, :, action_idx] = initial_value
        
    def act(self, observation, explore=True):
        """
        Selects an action using epsilon-greedy policy.
        
        Matches Haskell (QLearning.hs, chooseExploreAction):
        - If random < epsilon: choose random action
        - Else: choose action with max Q-value
        
        Args:
            observation: tuple (p1_prev, p2_prev) indices
            explore: boolean, whether to use epsilon-greedy
        """
        p1_prev, p2_prev = observation
        
        # Only explore if both explore=True AND self.explore_enabled
        if explore and self.explore_enabled and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit: Argmax Q(s, :)
            # Tie-breaking: Random among best
            q_values = self.q_table[p1_prev, p2_prev, :]
            max_q = np.max(q_values)
            # Find all indices with max value
            best_actions = np.flatnonzero(q_values == max_q)
            return np.random.choice(best_actions)

    def update(self, observation, action, reward, next_observation):
        """
        Updates Q-table using Bellman equation.
        
        Matches Haskell (QLearning.hs, chooseLearnDecrExploreQTable):
        Q(s,a) <- (1-alpha)*Q(s,a) + alpha*(r + gamma*max_a' Q(s', a'))
        
        Also updates exploration rate using exponential decay:
        epsilon <- epsilon * e^(-beta)
        
        If learn_enabled is False, skips Q-table update (test mode).
        """
        # Skip learning if disabled (test/frozen mode)
        if not self.learn_enabled:
            return
            
        p1_prev, p2_prev = observation
        p1_next, p2_next = next_observation
        
        current_q = self.q_table[p1_prev, p2_prev, action]
        max_next_q = np.max(self.q_table[p1_next, p2_next, :])
        
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        
        self.q_table[p1_prev, p2_prev, action] = new_q
        
        # Decay exploration using exponential formula
        # Matches Haskell (QLearning.hs line 294):
        # updateExploreRate decreaseFactor = env % exploreRate %~ (* ((exp 1) ** (-decreaseFactor)))
        if self.decay_epsilon:
            self.epsilon = self.epsilon * np.exp(-self.beta)

    def freeze(self):
        """
        Disable learning and exploration (test mode).
        Used in Phase 2 of rematch experiments.
        """
        self.explore_enabled = False
        self.learn_enabled = False
    
    def unfreeze(self, reset_epsilon=1.0):
        """
        Re-enable learning and exploration (relearn mode).
        Used in Phase 3 of rematch experiments.

        Args:
            reset_epsilon: New epsilon value (default 1.0 for full exploration reset)
        """
        self.explore_enabled = True
        self.learn_enabled = True
        self.epsilon = reset_epsilon
        self.decay_epsilon = True

    def set_forced_exploration(self, eps_min):
        """
        Pin epsilon at a fixed positive value with no decay; keep learning on.
        Used in Lambin-style post-convergence forced exploration: this agent
        keeps deviating with probability eps_min, while continuing to update
        its Q-table.
        """
        self.epsilon = eps_min
        self.decay_epsilon = False
        self.explore_enabled = True
        self.learn_enabled = True

    def set_greedy_learner(self):
        """
        Disable exploration but keep learning on.  The agent plays greedy
        actions but updates its Q-table in response to whatever the opponent
        does (used for the non-perturbed agent in Lambin Phase 1.5).
        """
        self.epsilon = 0.0
        self.decay_epsilon = False
        self.explore_enabled = False
        self.learn_enabled = True

    def load_q_table(self, path):
        """Load a Q-table from file."""
        self.q_table = np.load(path)
    
    def save_q_table(self, path):
        """Save Q-table to file."""
        np.save(path, self.q_table)

    @classmethod
    def from_trained(cls, q_table_path, num_actions, player_id):
        """
        Create an agent from a saved Q-table.
        Does not re-initialize Q-values (uses the trained ones).
        
        Args:
            q_table_path: Path to .npy file with Q-table
            num_actions: Number of actions in action space
            player_id: Player identifier (0 or 1)
        """
        agent = cls(num_actions, player_id, env=None)  # Don't initialize Q-table
        agent.load_q_table(q_table_path)
        return agent

    def clone(self):
        """
        Deep copy of the agent for rematching phases.
        Crucial for mimicking the freeze/thaw logic of the Haskell codebase.
        """
        return copy.deepcopy(self)

