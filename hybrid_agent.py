"""
Hybrid traffic control agent: Q-learning + emergency preemption.

Architecture:
    1. State compression — raw environment state (6-tuple) is compressed
       into direction totals: (ns_total//2, ew_total//2, phase, em_axis).
       This reduces the state space from ~146k to ~726, making tabular
       Q-learning feasible while preserving the decision-relevant info.

    2. Q-learning core — learns when to keep or switch the traffic phase
       based on relative queue pressure in each direction.

    3. Emergency override — deterministic rule: if an emergency vehicle
       is waiting at a red light, immediately switch to give it green.
       This handles the rare-but-critical emergency case that Q-learning
       struggles to learn from sparse penalties.

This design reflects real-world practice where adaptive signal controllers
combine learned policies with safety-critical overrides.
"""
from __future__ import annotations

import numpy as np
from q_learning_agent import QLearningAgent
from traffic_environment import TrafficEnvironment


def compress_state(state: tuple) -> tuple:
    """
    Compress the raw 6-tuple environment state into a compact
    representation suitable for tabular Q-learning.

    Raw state:  (q_N, q_S, q_E, q_W, phase, emergency_lane)
    Compressed: (ns_bucket, ew_bucket, phase, emergency_axis)

    - ns_bucket: (q_N + q_S) // 2, capped at 10  (0-10, 11 values)
    - ew_bucket: (q_E + q_W) // 2, capped at 10  (0-10, 11 values)
    - phase: 0 or 1                                (2 values)
    - emergency_axis: 0=none, 1=NS, 2=EW           (3 values)

    Total: 11 * 11 * 2 * 3 = 726 possible states.
    """
    q_N, q_S, q_E, q_W, phase = state[0], state[1], state[2], state[3], state[4]
    em = state[5] if len(state) > 5 else 0

    ns_bucket = min((q_N + q_S) // 2, 10)
    ew_bucket = min((q_E + q_W) // 2, 10)

    if em in (1, 2):
        em_axis = 1  # emergency on North-South axis
    elif em in (3, 4):
        em_axis = 2  # emergency on East-West axis
    else:
        em_axis = 0  # no emergency

    return (ns_bucket, ew_bucket, phase, em_axis)


class HybridAgent:
    """
    Hybrid agent combining Q-learning with emergency preemption.

    The agent uses a trained Q-learning policy for normal traffic control
    and a deterministic override for emergency vehicle handling.
    """

    def __init__(self, q_agent: QLearningAgent):
        self.q_agent = q_agent

    def choose_action(self, state: tuple) -> int:
        """
        Select an action given the raw environment state.

        1. If an emergency vehicle is at a red light → switch immediately.
        2. Otherwise → use Q-learning on the compressed state.

        Parameters
        ----------
        state : tuple
            Raw environment state (q_N, q_S, q_E, q_W, phase, emergency_lane).

        Returns
        -------
        int
            Action: 0 (keep phase) or 1 (switch phase).
        """
        em = state[5] if len(state) > 5 else 0
        phase = state[4]

        # Emergency override: give green to the emergency lane
        if em in (1, 2) and phase == 1:  # emergency on NS, but EW is green
            return 1  # switch to NS green
        if em in (3, 4) and phase == 0:  # emergency on EW, but NS is green
            return 1  # switch to EW green

        # Normal traffic: use Q-learning on compressed state
        return self.q_agent.choose_action(compress_state(state))


def train_hybrid(
    arrival_rate_ns: float = 0.4,
    arrival_rate_ew: float = 0.4,
    emergency_prob: float = 0.05,
    n_episodes: int = 10000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    seed: int = 42,
) -> HybridAgent:
    """
    Train a HybridAgent on the traffic environment.

    Q-learning is trained WITH emergency events in the environment so
    the compressed state (which includes emergency_axis) gets proper
    Q-value updates.  The emergency override still handles deployment.

    Parameters
    ----------
    arrival_rate_ns, arrival_rate_ew : float
        Poisson mean arrival rates for NS and EW lanes.
    emergency_prob : float
        Probability of emergency vehicle per timestep during training.
    n_episodes : int
        Number of training episodes.
    max_steps : int
        Steps per episode.
    alpha, gamma : float
        Q-learning hyperparameters.
    epsilon_start, epsilon_end : float
        Linear epsilon decay schedule.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    HybridAgent
        The trained hybrid agent.
    """
    q_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed)
    rng = np.random.default_rng(seed + 1000)

    for ep in range(n_episodes):
        # Linear epsilon decay
        frac = ep / max(n_episodes - 1, 1)
        q_agent.epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        # Each episode uses a different random seed for diverse experience
        ep_seed = int(rng.integers(0, 100_000))
        env = TrafficEnvironment(
            arrival_rate_ns=arrival_rate_ns,
            arrival_rate_ew=arrival_rate_ew,
            emergency_prob=emergency_prob,
            max_steps=max_steps,
            seed=ep_seed,
        )

        state = compress_state(env.reset())
        done = False
        while not done:
            action = q_agent.choose_action(state)
            raw_next, reward, done = env.step(action)
            next_state = compress_state(raw_next)
            q_agent.update(state, action, reward, next_state)
            state = next_state

    # Disable exploration for deployment
    q_agent.epsilon = 0.0
    return HybridAgent(q_agent)
