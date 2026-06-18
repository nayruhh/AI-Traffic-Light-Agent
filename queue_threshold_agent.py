class QueueThresholdAgent:
    """
    Baseline traffic light controller that switches phases based on
    the relative queue lengths on the two approaches (NS vs EW).

    Assumes the environment state is a tuple:
        (q_N, q_S, q_E, q_W, phase)

    and that `phase` encodes the current green direction as:
        0 -> North-South (NS) green
        1 -> East-West (EW) green
    """

    def __init__(self) -> None:
        # No tunable parameters for this simple baseline.
        # All decisions are made directly from the current state.
        pass

    def choose_action(self, state) -> int:
        """
        Decide whether to keep the current phase or switch based on
        the current queue lengths.

        Args:
            state: A tuple (q_N, q_S, q_E, q_W, phase, emergency_lane) where
                q_N, q_S, q_E, q_W are queue lengths for each approach,
                phase is the current green direction (0=NS, 1=EW),
                and emergency_lane indicates an active emergency vehicle
                (0=none, 1=N, 2=S, 3=E, 4=W).

        Returns:
            0 to keep the current phase,
            1 to switch the phase.
        """
        q_N, q_S, q_E, q_W, phase = state[0], state[1], state[2], state[3], state[4]
        # emergency_lane is state[5] when present (compatible with old 5-tuple too)
        emergency_lane = state[5] if len(state) > 5 else 0

        # --- Emergency vehicle priority: switch to give it green if needed ---
        if emergency_lane in (1, 2) and phase == 1:  # emergency on NS, but EW green
            return 1  # switch to NS green
        if emergency_lane in (3, 4) and phase == 0:  # emergency on EW, but NS green
            return 1  # switch to EW green
        # If emergency already has green, keep current phase
        if emergency_lane != 0:
            return 0

        # Compute total queues for each direction pair
        ns_queue = q_N + q_S
        ew_queue = q_E + q_W

        # If NS side is more congested but EW is currently green,
        # switch to give priority to NS.
        if ns_queue > ew_queue and phase == 1:  # currently EW green
            return 1

        # If EW side is more congested but NS is currently green,
        # switch to give priority to EW.
        if ew_queue > ns_queue and phase == 0:  # currently NS green
            return 1

        # Otherwise, keep the current phase.
        return 0

