import numpy as np

class CalvanoEnv:
    """
    A Gym-like environment for the Calvano et al. Replication Game.
    Simultaneous move game with Logit Demand.
    """
    def __init__(self):
        self.a0 = 0.0
        self.a1 = 2.0
        self.a2 = 2.0
        self.mu = 0.25
        self.c1 = 1.0  # Cost player 1
        self.c2 = 1.0  # Cost player 2
        
        # Grid Parameters
        self.p_bertrand = 1.47
        self.p_monopoly = 1.92
        self.xi = 0.1
        self.m = 14  # Number of steps in grid ("pM = 14")

        # Action Space Calculation
        price_delta = self.p_monopoly - self.p_bertrand
        self.lower_bound = self.p_bertrand - self.xi * price_delta
        self.upper_bound = self.p_monopoly + self.xi * price_delta
        
        self.price_grid = np.linspace(self.lower_bound, self.upper_bound, self.m + 1)
        self.num_actions = len(self.price_grid)
        
        self.state = None

    def reset(self):
        """
        Resets the environment. 
        In Calvano et al, agents observe the *previous* prices.
        We initialize with random prices from the grid.
        Returns:
            obs: {0: p1_idx, 1: p2_idx} (Joint action indices)
        """
        # Random initial prices (indices)
        p1_idx = np.random.randint(0, self.num_actions)
        p2_idx = np.random.randint(0, self.num_actions)
        
        # State is the tuple of indices (player1_idx, player2_idx)
        self.state = (p1_idx, p2_idx)
        
        # Observation for both agents is the full state
        # In this game, both see the same thing: (p1_last, p2_last)
        return {0: self.state, 1: self.state}

    def step(self, action_dict):
        """
        Executes simultaneous moves.
        Args:
            action_dict: {0: p1_action_idx, 1: p2_action_idx}
        """
        p1_idx = action_dict[0]
        p2_idx = action_dict[1]
        
        p1_price = self.price_grid[p1_idx]
        p2_price = self.price_grid[p2_idx]
        
        # Eq 5: D_i = exp((ai - pi)/mu) / denominator
        # Denom = exp((a1-p1)/mu) + exp((a2-p2)/mu) + exp(a0/mu)
        
        exp_1 = np.exp((self.a1 - p1_price) / self.mu)
        exp_2 = np.exp((self.a2 - p2_price) / self.mu)
        exp_0 = np.exp(self.a0 / self.mu)
        
        denominator = exp_1 + exp_2 + exp_0
        
        demand_1 = exp_1 / denominator
        demand_2 = exp_2 / denominator
        
        # --- Profit ---
        # pi = (p - c) * D
        profit_1 = (p1_price - self.c1) * demand_1
        profit_2 = (p2_price - self.c2) * demand_2
        
        # Update state to current actions
        self.state = (p1_idx, p2_idx)
        
        # Observations for next round
        obs = {0: self.state, 1: self.state}
        rewards = {0: profit_1, 1: profit_2}
        dones = {"__all__": False} # We handle max_iters in runner
        infos = {}
        
        return obs, rewards, dones, infos


class CalvanoAdjPriceGridEnv(CalvanoEnv):
    """
    Identical to CalvanoEnv but with a price grid adjusted to exactly include
    the Nash (1.47) and Monopoly (1.92) prices.
    """
    def __init__(self):
        super().__init__()

        # Redefine grid to ensure inclusion of p_bertrand and p_monopoly
        # We use 12 steps between them: (1.92 - 1.47) / 12 = 0.0375 step size
        # Grid: [1.4325, 1.47, ..., 1.92, 1.9575] (15 points)

        k_steps = 12  # Steps between critical prices
        delta = (self.p_monopoly - self.p_bertrand) / k_steps

        # One buffer step in each direction matches the original density (m=14)
        p_min = self.p_bertrand - 1.0 * delta
        p_max = self.p_monopoly + 1.0 * delta

        # Re-initialize grid properties
        self.price_grid = np.linspace(p_min, p_max, self.m + 1)
        self.num_actions = len(self.price_grid)


class CalvanoWideGridEnv(CalvanoEnv):
    """
    Like CalvanoAdjPriceGridEnv (step delta = 0.0375, with Nash = 1.47 and
    Monopoly = 1.92 lying exactly on the grid) but with a WIDER price range:
    `buffer_steps` extra grid points are added below the Nash price and above
    the Monopoly price (default 4 each side).

    For buffer_steps = 4 this gives 21 prices spanning [1.32, 2.07], with
    Nash at index 4 and Monopoly at index 16.

    Purpose: test grid edge effects -- whether collusive prices overshoot the
    monopoly price or competitive prices undershoot Nash once the grid boundary
    is no longer binding (the standard grid has only one buffer step each side).
    """
    def __init__(self, buffer_steps=4):
        super().__init__()

        k_steps = 12  # interior steps between Nash and Monopoly (matches AdjPriceGrid)
        delta = (self.p_monopoly - self.p_bertrand) / k_steps
        self.buffer_steps = buffer_steps

        p_min = self.p_bertrand - buffer_steps * delta
        p_max = self.p_monopoly + buffer_steps * delta
        n_points = k_steps + 2 * buffer_steps + 1  # 21 for buffer_steps=4

        self.price_grid = np.linspace(p_min, p_max, n_points)
        self.num_actions = len(self.price_grid)


class NoisyObservationEnv(CalvanoEnv):
    """
    Environment where agents observe both prices, but with added Gaussian noise.

    This tests whether reducing signal quality (while maintaining full observability)
    affects the robustness of learned collusive strategies.

    Key hypothesis: Noise may prevent overfitting to opponent-specific strategies,
    potentially improving robustness when agents are rematched.

    Inherits from CalvanoEnv (baseline) to keep the noise feature independent
    of grid adjustment choices.
    """
    def __init__(self, noise_std=0.1):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise added to price observations.
                       Noise is in price units (not index units).
                       Example values:
                       - 0.0: No noise (equivalent to baseline)
                       - 0.05: Low noise (~1-2 price steps)
                       - 0.10: Medium noise (~2-3 price steps)
                       - 0.20: High noise (~5-6 price steps)
        """
        super().__init__()
        self.noise_std = noise_std

    def _add_observation_noise(self, price_idx):
        """
        Add Gaussian noise to a price observation.

        Process:
        1. Convert index to actual price
        2. Add Gaussian noise to price
        3. Clip to valid price range
        4. Map back to nearest grid index

        Args:
            price_idx: True price index (0 to num_actions-1)

        Returns:
            Noisy price index (0 to num_actions-1)
        """
        if self.noise_std == 0:
            return price_idx

        # Convert index to price
        true_price = self.price_grid[price_idx]

        # Add Gaussian noise
        noisy_price = true_price + np.random.normal(0, self.noise_std)

        # Clip to valid price range
        noisy_price = np.clip(
            noisy_price,
            self.price_grid[0],
            self.price_grid[-1]
        )

        # Find nearest grid point
        noisy_idx = np.argmin(np.abs(self.price_grid - noisy_price))

        return noisy_idx

    def reset(self):
        """
        Reset environment with noisy observations.
        """
        # Random initial prices (indices)
        p1_idx = np.random.randint(0, self.num_actions)
        p2_idx = np.random.randint(0, self.num_actions)

        # State is the true tuple of indices
        self.state = (p1_idx, p2_idx)

        # Add noise to observations
        p1_obs = self._add_observation_noise(p1_idx)
        p2_obs = self._add_observation_noise(p2_idx)

        # Both agents observe noisy versions of both prices
        obs = {0: (p1_obs, p2_obs), 1: (p1_obs, p2_obs)}

        return obs

    def step(self, action_dict):
        """
        Execute step with noisy observations.

        Note: Rewards are based on TRUE prices (agents know their realized profits),
        but observations for next state are noisy.
        """
        p1_idx = action_dict[0]
        p2_idx = action_dict[1]

        p1_price = self.price_grid[p1_idx]
        p2_price = self.price_grid[p2_idx]

        # Calculate demand and profits (based on true prices)
        exp_1 = np.exp((self.a1 - p1_price) / self.mu)
        exp_2 = np.exp((self.a2 - p2_price) / self.mu)
        exp_0 = np.exp(self.a0 / self.mu)

        denominator = exp_1 + exp_2 + exp_0

        demand_1 = exp_1 / denominator
        demand_2 = exp_2 / denominator

        profit_1 = (p1_price - self.c1) * demand_1
        profit_2 = (p2_price - self.c2) * demand_2

        # Update state to current actions (true state)
        self.state = (p1_idx, p2_idx)

        # Add noise to observations
        p1_obs = self._add_observation_noise(p1_idx)
        p2_obs = self._add_observation_noise(p2_idx)

        # Both agents observe noisy versions of both prices
        obs = {0: (p1_obs, p2_obs), 1: (p1_obs, p2_obs)}
        rewards = {0: profit_1, 1: profit_2}
        dones = {"__all__": False}
        infos = {}

        return obs, rewards, dones, infos


class PartialObservabilityEnv(CalvanoEnv):
    """
    Environment with partial observability: agents observe own price perfectly
    but opponent's price with added Gaussian noise.

    This creates a smooth spectrum between:
    - opponent_noise_std = 0: Full observability (equivalent to CalvanoEnv baseline)
    - opponent_noise_std → ∞: Effectively own-price-only observation

    Key difference from NoisyObservationEnv:
    - NoisyObservationEnv adds noise to BOTH prices
    - PartialObservabilityEnv adds noise ONLY to opponent's price

    This design matches the Haskell OwnPriceObservation paradigm where agents
    have perfect knowledge of their own actions but imperfect information about
    the opponent.
    """
    def __init__(self, opponent_noise_std=0.1):
        """
        Args:
            opponent_noise_std: Standard deviation of Gaussian noise added to
                               opponent's price observation (in price units).
                               Example values:
                               - 0.0: Perfect observability (equivalent to baseline)
                               - 0.05: Low noise (~1-2 price steps)
                               - 0.10: Medium noise (~2-3 price steps)
                               - 0.20: High noise (~5-6 price steps)
                               - 1.0+: Very high noise, opponent info nearly useless
        """
        super().__init__()
        self.opponent_noise_std = opponent_noise_std

    def _add_opponent_noise(self, opponent_price_idx):
        """
        Add Gaussian noise to opponent's price observation.

        Process:
        1. Convert index to actual price
        2. Add Gaussian noise to price
        3. Clip to valid price range
        4. Map back to nearest grid index

        Args:
            opponent_price_idx: True opponent price index (0 to num_actions-1)

        Returns:
            Noisy opponent price index (0 to num_actions-1)
        """
        if self.opponent_noise_std == 0:
            return opponent_price_idx

        # Convert index to price
        true_price = self.price_grid[opponent_price_idx]

        # Add Gaussian noise
        noisy_price = true_price + np.random.normal(0, self.opponent_noise_std)

        # Clip to valid price range
        noisy_price = np.clip(
            noisy_price,
            self.price_grid[0],
            self.price_grid[-1]
        )

        # Find nearest grid point
        noisy_idx = np.argmin(np.abs(self.price_grid - noisy_price))

        return noisy_idx

    def reset(self):
        """
        Reset environment with partial observability.
        Each agent observes their own price perfectly, opponent's price with noise.
        """
        # Random initial prices (indices)
        p1_idx = np.random.randint(0, self.num_actions)
        p2_idx = np.random.randint(0, self.num_actions)

        # State is the true tuple of indices
        self.state = (p1_idx, p2_idx)

        # Player 1 observes: (own_price=p1, noisy_opponent=p2+noise)
        # Player 2 observes: (own_price=p2, noisy_opponent=p1+noise)
        p1_sees_p2 = self._add_opponent_noise(p2_idx)
        p2_sees_p1 = self._add_opponent_noise(p1_idx)

        # Observation format: (own_price, opponent_price)
        # Agent 0: sees (p1, noisy_p2)
        # Agent 1: sees (p2, noisy_p1) - note order is (own, opponent)
        obs = {
            0: (p1_idx, p1_sees_p2),
            1: (p2_idx, p2_sees_p1)
        }

        return obs

    def step(self, action_dict):
        """
        Execute step with partial observability.

        Rewards are based on TRUE prices (agents know their realized profits),
        but observations for next state include noise on opponent's price.
        """
        p1_idx = action_dict[0]
        p2_idx = action_dict[1]

        p1_price = self.price_grid[p1_idx]
        p2_price = self.price_grid[p2_idx]

        # Calculate demand and profits (based on true prices)
        exp_1 = np.exp((self.a1 - p1_price) / self.mu)
        exp_2 = np.exp((self.a2 - p2_price) / self.mu)
        exp_0 = np.exp(self.a0 / self.mu)

        denominator = exp_1 + exp_2 + exp_0

        demand_1 = exp_1 / denominator
        demand_2 = exp_2 / denominator

        profit_1 = (p1_price - self.c1) * demand_1
        profit_2 = (p2_price - self.c2) * demand_2

        # Update state to current actions (true state)
        self.state = (p1_idx, p2_idx)

        # Add noise to opponent observations only
        p1_sees_p2 = self._add_opponent_noise(p2_idx)
        p2_sees_p1 = self._add_opponent_noise(p1_idx)

        # Each agent observes (own_price, noisy_opponent_price)
        obs = {
            0: (p1_idx, p1_sees_p2),
            1: (p2_idx, p2_sees_p1)
        }
        rewards = {0: profit_1, 1: profit_2}
        dones = {"__all__": False}
        infos = {}

        return obs, rewards, dones, infos
