"""
Tabular Q-learning agent for the TrafficEnvironment.

States: (q_N, q_S, q_E, q_W, phase, emergency_lane) — tuples as returned by env.get_state().
Actions: 0 = keep current phase, 1 = switch phase.
"""
from __future__ import annotations

import numpy as np


class QLearningAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.
    Compatible with TrafficEnvironment (states as tuples, actions 0 or 1).
    """

    # Action space: 0 = keep phase, 1 = switch phase
    N_ACTIONS = 2

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        """
        Initialize the Q-learning agent.

        Parameters
        ----------
        alpha : float
            Learning rate for Q-updates.
        gamma : float
            Discount factor for future rewards.
        epsilon : float
            Probability of taking a random action (exploration).
        seed : int | None
            Random seed for reproducibility (optional).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Q-table: keys are (state, action), values are floats
        # state = (q_N, q_S, q_E, q_W, phase)
        self._q_table: dict[tuple, float] = {}

    def get_q_value(self, state: tuple, action: int) -> float:
        """
        Return Q(s, a). If (state, action) is not in the table, return 0.0.

        Parameters
        ----------
        state : tuple
            (q_N, q_S, q_E, q_W, phase)
        action : int
            0 or 1

        Returns
        -------
        float
        """
        key = (state, action)
        return self._q_table.get(key, 0.0)

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon, return a random action (0 or 1).
        Otherwise, return the action with the highest Q-value for this state.

        Parameters
        ----------
        state : tuple
            (q_N, q_S, q_E, q_W, phase)

        Returns
        -------
        int
            Action: 0 (keep phase) or 1 (switch phase).
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.N_ACTIONS))
        return self.get_best_action(state)

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ) -> None:
        """
        Perform one Q-learning update:

            Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

        Parameters
        ----------
        state : tuple
            Current state (q_N, q_S, q_E, q_W, phase).
        action : int
            Action taken (0 or 1).
        reward : float
            Reward received after taking action in state.
        next_state : tuple
            Resulting state (q_N, q_S, q_E, q_W, phase).
        """
        q_sa = self.get_q_value(state, action)
        max_q_next = max(
            self.get_q_value(next_state, a) for a in range(self.N_ACTIONS)
        )
        td_target = reward + self.gamma * max_q_next
        new_q = q_sa + self.alpha * (td_target - q_sa)
        self._q_table[(state, action)] = new_q

    def get_best_action(self, state: tuple) -> int:
        """
        Return the action with the highest Q-value for the given state.
        Ties are broken arbitrarily (first max wins).

        Parameters
        ----------
        state : tuple
            (q_N, q_S, q_E, q_W, phase)

        Returns
        -------
        int
            Action: 0 or 1.
        """
        q0 = self.get_q_value(state, 0)
        q1 = self.get_q_value(state, 1)
        return 1 if q1 > q0 else 0
