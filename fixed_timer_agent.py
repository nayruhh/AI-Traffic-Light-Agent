class FixedTimerAgent:
    """
    Simple baseline controller that switches the traffic light phase
    after a fixed number of environment timesteps.
    """

    def __init__(self, switch_interval: int = 10) -> None:
        """
        Args:
            switch_interval: Number of timesteps to wait before switching
                the traffic light phase.
        """
        self.switch_interval = switch_interval
        self.step_counter = 0
        self.current_phase = 0

    def choose_action(self, state) -> int:
        """
        Decide whether to keep the current phase or switch.

        Args:
            state: Environment state (unused for fixed-timer baseline).

        Returns:
            0 to keep the current phase,
            1 to switch the phase.
        """
        self.step_counter += 1

        if self.step_counter >= self.switch_interval:
            # Time to switch phase
            self.step_counter = 0
            # Optionally track phase internally if needed by the environment
            self.current_phase = 1 - self.current_phase
            return 1

        # Keep current phase
        return 0

