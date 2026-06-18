from __future__ import annotations

import numpy as np


class TrafficEnvironment:
    """
    Simple 4-way intersection environment for tabular RL.

    Lanes: North (N), South (S), East (E), West (W)
    Phases:
        0 -> North-South green
        1 -> East-West green

    State (discrete):
        (q_N, q_S, q_E, q_W, phase, emergency_lane)
        where queues are integers in [0, max_queue]
        and emergency_lane encodes:
            0 = no emergency vehicle
            1 = emergency in North lane
            2 = emergency in South lane
            3 = emergency in East lane
            4 = emergency in West lane
    """

    # Mapping from emergency_lane code to queue index
    _EMERGENCY_LANE_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}

    def __init__(
        self,
        arrival_rate_ns: float = 0.7,
        arrival_rate_ew: float = 0.7,
        max_queue: int = 10,
        max_steps: int = 200,
        seed: int | None = None,
        depart_per_lane: int = 1,
        emergency_prob: float = 0.01,
        emergency_penalty: float = 50.0,
    ):
        """
        Initialize the environment.

        Parameters
        ----------
        arrival_rate_ns : float
            Poisson mean for arrivals on N and S lanes (per timestep).
        arrival_rate_ew : float
            Poisson mean for arrivals on E and W lanes (per timestep).
        max_queue : int
            Maximum queue length per lane (anything above is clipped).
        max_steps : int
            Episode length in timesteps.
        seed : int | None
            Random seed for reproducibility (optional).
        depart_per_lane : int
            Max number of cars that can depart per green lane per timestep.
        emergency_prob : float
            Probability that an emergency vehicle appears each timestep
            (only when no emergency vehicle is already present).
        emergency_penalty : float
            Extra penalty applied when an emergency vehicle is waiting at red.
        """
        self.arrival_rate_ns = arrival_rate_ns
        self.arrival_rate_ew = arrival_rate_ew
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.depart_per_lane = depart_per_lane

        # --- Emergency vehicle parameters ---
        self.emergency_prob = emergency_prob
        self.emergency_penalty = emergency_penalty

        self.rng = np.random.default_rng(seed)

        # Internal state
        self.queues = None       # [q_N, q_S, q_E, q_W]
        self.phase = None        # 0 or 1
        self.timestep = None
        self.emergency_lane = None  # 0 = none, 1 = N, 2 = S, 3 = E, 4 = W

        # Tracking metrics
        self.total_waiting_time = None
        self.total_cars_passed = None
        self.emergency_wait_steps = None   # steps the current emergency waited
        self.total_emergency_waits = None  # cumulative emergency wait steps

        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns
        -------
        state : tuple[int, int, int, int, int, int]
            (q_N, q_S, q_E, q_W, phase, emergency_lane)
        """
        self.queues = np.zeros(4, dtype=int)  # N, S, E, W
        self.phase = 0  # start with NS green by default
        self.timestep = 0

        # --- Reset emergency state ---
        self.emergency_lane = 0  # no emergency vehicle at start

        self.total_waiting_time = 0
        self.total_cars_passed = 0
        self.emergency_wait_steps = 0
        self.total_emergency_waits = 0

        return self.get_state()

    def get_state(self):
        """
        Return current discrete state representation.

        Returns
        -------
        state : tuple[int, int, int, int, int, int]
            (q_N, q_S, q_E, q_W, phase, emergency_lane)
        """
        q_N, q_S, q_E, q_W = self.queues
        return (
            int(q_N), int(q_S), int(q_E), int(q_W),
            int(self.phase),
            int(self.emergency_lane),
        )

    def _arrivals(self):
        """
        Sample new vehicle arrivals based on Poisson processes.
        """
        # N, S share arrival_rate_ns; E, W share arrival_rate_ew
        arrivals_ns = self.rng.poisson(self.arrival_rate_ns, size=2)  # N, S
        arrivals_ew = self.rng.poisson(self.arrival_rate_ew, size=2)  # E, W
        arrivals = np.concatenate([arrivals_ns, arrivals_ew])

        # Update queues with arrivals, respecting max_queue
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

    def _maybe_spawn_emergency(self):
        """
        With small probability, spawn an emergency vehicle in a random lane.
        Only spawns if no emergency vehicle is currently present.
        The emergency vehicle is added to that lane's queue (counts as a car).
        """
        if self.emergency_lane != 0:
            return  # already have an active emergency vehicle

        if self.rng.random() < self.emergency_prob:
            # Pick a random lane: 1=N, 2=S, 3=E, 4=W
            self.emergency_lane = int(self.rng.integers(1, 5))
            # Add the emergency vehicle to the lane's queue
            idx = self._EMERGENCY_LANE_TO_INDEX[self.emergency_lane]
            self.queues[idx] = min(self.queues[idx] + 1, self.max_queue)
            self.emergency_wait_steps = 0

    def _is_emergency_lane_green(self) -> bool:
        """
        Check whether the lane containing the emergency vehicle has a green signal.
        """
        if self.emergency_lane == 0:
            return False
        # N(1), S(2) are green in phase 0; E(3), W(4) are green in phase 1
        if self.emergency_lane in (1, 2) and self.phase == 0:
            return True
        if self.emergency_lane in (3, 4) and self.phase == 1:
            return True
        return False

    def _departures(self):
        """
        Let vehicles depart from lanes that have green.
        Emergency vehicles depart immediately when their lane is green.
        """
        # Phase 0: N/S green, E/W red
        # Phase 1: E/W green, N/S red
        if self.phase == 0:
            green_indices = [0, 1]  # N, S
        else:
            green_indices = [2, 3]  # E, W

        departed = 0
        for idx in green_indices:
            # Number of cars that can depart this timestep on this lane
            leaving = min(self.depart_per_lane, self.queues[idx])
            self.queues[idx] -= leaving
            departed += leaving

        self.total_cars_passed += departed

        # --- Clear emergency vehicle if its lane was green ---
        if self.emergency_lane != 0 and self._is_emergency_lane_green():
            self.emergency_lane = 0
            self.emergency_wait_steps = 0

    def step(self, action: int):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            0 -> keep current phase
            1 -> switch phase

        Returns
        -------
        next_state : tuple[int, int, int, int, int, int]
        reward : float
        done : bool
        """
        # 1. Apply action (possibly change phase)
        if action == 1:
            self.phase = 1 - self.phase  # toggle between 0 and 1
        elif action != 0:
            raise ValueError("Action must be 0 (keep) or 1 (switch).")

        # 2. Arrivals (regular traffic)
        self._arrivals()

        # 3. Possibly spawn an emergency vehicle
        self._maybe_spawn_emergency()

        # 4. Departures on green lanes (also clears emergency if green)
        self._departures()

        # 5. Compute reward and update tracking
        total_queue_length = int(self.queues.sum())
        # Waiting time approximation: all queued vehicles waited this step
        self.total_waiting_time += total_queue_length

        reward = -float(total_queue_length)

        # --- Emergency penalty: if emergency vehicle is still waiting at red ---
        if self.emergency_lane != 0:
            self.emergency_wait_steps += 1
            self.total_emergency_waits += 1
            reward -= self.emergency_penalty

        # 6. Advance time and check termination
        self.timestep += 1
        done = self.timestep >= self.max_steps

        next_state = self.get_state()
        return next_state, reward, done
