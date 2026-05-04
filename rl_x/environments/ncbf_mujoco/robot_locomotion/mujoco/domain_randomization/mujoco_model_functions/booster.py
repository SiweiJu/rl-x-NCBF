import numpy as np


class BoosterDRMuJoCoModel:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["mujoco_model"]

        self.randomize_gravity = config.get("randomize_gravity", True)
        self.gravity_range = np.array(config.get("gravity_range", [9.51, 10.11]))

        self.randomize_model_geom_friction_tangential = config.get("randomize_model_geom_friction_tangential", False)
        self.model_geom_friction_tangential_range = np.array(config.get("model_geom_friction_tangential_range", [0.5, 1.5]))
        self.randomize_model_geom_friction_torsional = config.get("randomize_model_geom_friction_torsional", False)
        self.model_geom_friction_torsional_range = np.array(config.get("model_geom_friction_torsional_range", [0.1, 0.3]))
        self.randomize_model_geom_friction_rolling = config.get("randomize_model_geom_friction_rolling", False)
        self.model_geom_friction_rolling_range = np.array(config.get("model_geom_friction_rolling_range", [0.00008, 0.00012]))

        self.randomize_floor_geom_friction_tangential = config.get("randomize_floor_geom_friction_tangential", True)
        self.floor_geom_friction_tangential_range = np.array(config.get("floor_geom_friction_tangential_range", [0.5, 1.5]))
        self.randomize_floor_geom_friction_torsional = config.get("randomize_floor_geom_friction_torsional", True)
        self.floor_geom_friction_torsional_range = np.array(config.get("floor_geom_friction_torsional_range", [0.1, 0.3]))
        self.randomize_floor_geom_friction_rolling = config.get("randomize_floor_geom_friction_rolling", True)
        self.floor_geom_friction_rolling_range = np.array(config.get("floor_geom_friction_rolling_range", [0.00008, 0.00012]))

        self.randomize_geom_damping = config.get("randomize_geom_damping", False)
        self.geom_damping_range = np.array(config.get("geom_damping_range", [100.0, 500.0]))
        self.randomize_geom_stiffness = config.get("randomize_geom_stiffness", False)
        self.geom_stiffness_range = np.array(config.get("geom_stiffness_range", [100000.0, 300000.0]))

        self.default_geom_friction = env.initial_mj_model.geom_friction.copy()
        self.default_geom_solref = env.initial_mj_model.geom_solref.copy()
        self.default_gravity = env.initial_mj_model.opt.gravity.copy()
        self.floor_geom_id = env.floor_geom_id


    @staticmethod
    def _lerp(value_range, interpolation):
        return value_range[0] + (value_range[1] - value_range[0]) * interpolation


    def sample(self):
        model = self.env.internal_state["mj_model"]
        geom_friction = self.default_geom_friction.copy()

        if (
            self.randomize_model_geom_friction_tangential or
            self.randomize_model_geom_friction_torsional or
            self.randomize_model_geom_friction_rolling
        ):
            interpolation = self.env.np_rng.uniform(size=(self.default_geom_friction.shape[0],))
            tangential = self.default_geom_friction[:, 0].copy()
            torsional = self.default_geom_friction[:, 1].copy()
            rolling = self.default_geom_friction[:, 2].copy()
            if self.randomize_model_geom_friction_tangential:
                tangential = self._lerp(self.model_geom_friction_tangential_range, interpolation)
            if self.randomize_model_geom_friction_torsional:
                torsional = self._lerp(self.model_geom_friction_torsional_range, interpolation)
            if self.randomize_model_geom_friction_rolling:
                rolling = self._lerp(self.model_geom_friction_rolling_range, interpolation)
            model_geom_friction = np.stack([tangential, torsional, rolling], axis=1)
            geom_ids = np.arange(self.default_geom_friction.shape[0])
            geom_friction[geom_ids != self.floor_geom_id] = model_geom_friction[geom_ids != self.floor_geom_id]

        if (
            self.randomize_floor_geom_friction_tangential or
            self.randomize_floor_geom_friction_torsional or
            self.randomize_floor_geom_friction_rolling
        ):
            interpolation = self.env.np_rng.uniform()
            floor_friction = self.default_geom_friction[self.floor_geom_id].copy()
            tangential = floor_friction[0]
            torsional = floor_friction[1]
            rolling = floor_friction[2]
            if self.randomize_floor_geom_friction_tangential:
                tangential = self._lerp(self.floor_geom_friction_tangential_range, interpolation)
            if self.randomize_floor_geom_friction_torsional:
                torsional = self._lerp(self.floor_geom_friction_torsional_range, interpolation)
            if self.randomize_floor_geom_friction_rolling:
                rolling = self._lerp(self.floor_geom_friction_rolling_range, interpolation)
            geom_friction[self.floor_geom_id] = np.array([tangential, torsional, rolling])

        geom_solref = self.default_geom_solref.copy()
        if self.randomize_geom_stiffness or self.randomize_geom_damping:
            stiffness_interpolation = self.env.np_rng.uniform(size=(self.default_geom_solref.shape[0],))
            damping_interpolation = self.env.np_rng.uniform(size=(self.default_geom_solref.shape[0],))
            if self.randomize_geom_stiffness:
                geom_solref[:, 0] = -self._lerp(self.geom_stiffness_range, stiffness_interpolation)
            if self.randomize_geom_damping:
                geom_solref[:, 1] = -self._lerp(self.geom_damping_range, damping_interpolation)

        model.geom_friction[:] = geom_friction
        model.geom_solref[:] = geom_solref
        if self.randomize_gravity:
            model.opt.gravity[:] = np.array([0.0, 0.0, -self._lerp(self.gravity_range, self.env.np_rng.uniform())])
        else:
            model.opt.gravity[:] = self.default_gravity
