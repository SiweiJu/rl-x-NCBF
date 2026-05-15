import numpy as np


class BoosterDRPerturbation:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["perturbation"]
        self.kick_robots = config.get("kick_robots", True)
        self.kick_min_vel = config.get("kick_min_vel", 0.0)
        self.kick_max_vel = config.get("kick_max_vel", 0.4)
        velocity_range = config.get("velocity_range")
        self.velocity_range = None if velocity_range is None else np.array([
            velocity_range.get(axis, [0.0, 0.0])
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ], dtype=float)


    def sample(self):
        if not self.kick_robots:
            return

        if self.velocity_range is not None:
            kick = self.env.np_rng.uniform(
                low=self.velocity_range[:, 0],
                high=self.velocity_range[:, 1],
            )
            self.env.internal_state["data"].qvel[:6] = kick
            return

        kick = self.env.np_rng.uniform(low=-1.0, high=1.0, size=(3,)) * self.kick_max_vel
        kick = np.sign(kick) * np.clip(np.abs(kick), a_min=self.kick_min_vel, a_max=self.kick_max_vel)
        self.env.internal_state["data"].qvel[:3] += kick
