import jax
import jax.numpy as jnp


class BoosterDRMuJoCoModel:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["mujoco_model"]

        self.randomize_gravity = config.get("randomize_gravity", True)
        self.gravity_range = jnp.array(config.get("gravity_range", [9.51, 10.11]))

        self.randomize_model_geom_friction_tangential = config.get("randomize_model_geom_friction_tangential", False)
        self.model_geom_friction_tangential_range = jnp.array(config.get("model_geom_friction_tangential_range", [0.5, 1.5]))
        self.randomize_model_geom_friction_torsional = config.get("randomize_model_geom_friction_torsional", False)
        self.model_geom_friction_torsional_range = jnp.array(config.get("model_geom_friction_torsional_range", [0.1, 0.3]))
        self.randomize_model_geom_friction_rolling = config.get("randomize_model_geom_friction_rolling", False)
        self.model_geom_friction_rolling_range = jnp.array(config.get("model_geom_friction_rolling_range", [0.00008, 0.00012]))

        self.randomize_floor_geom_friction_tangential = config.get("randomize_floor_geom_friction_tangential", True)
        self.floor_geom_friction_tangential_range = jnp.array(config.get("floor_geom_friction_tangential_range", [0.5, 1.5]))
        self.randomize_floor_geom_friction_torsional = config.get("randomize_floor_geom_friction_torsional", True)
        self.floor_geom_friction_torsional_range = jnp.array(config.get("floor_geom_friction_torsional_range", [0.1, 0.3]))
        self.randomize_floor_geom_friction_rolling = config.get("randomize_floor_geom_friction_rolling", True)
        self.floor_geom_friction_rolling_range = jnp.array(config.get("floor_geom_friction_rolling_range", [0.00008, 0.00012]))

        self.randomize_geom_damping = config.get("randomize_geom_damping", False)
        self.geom_damping_range = jnp.array(config.get("geom_damping_range", [100.0, 500.0]))
        self.randomize_geom_stiffness = config.get("randomize_geom_stiffness", False)
        self.geom_stiffness_range = jnp.array(config.get("geom_stiffness_range", [100000.0, 300000.0]))

        self.default_geom_friction = env.initial_mjx_model.geom_friction
        self.default_geom_solref = env.initial_mjx_model.geom_solref
        self.default_gravity = env.initial_mjx_model.opt.gravity
        self.floor_geom_id = env.floor_geom_id
        self.geom_ids = jnp.arange(env.initial_mj_model.ngeom)


    @staticmethod
    def _lerp(value_range, interpolation):
        return value_range[0] + (value_range[1] - value_range[0]) * interpolation


    def sample(self, internal_state, mjx_model, should_randomize, key):
        gravity_key, model_friction_key, floor_friction_key, damping_key, stiffness_key = jax.random.split(key, 5)

        geom_friction = self.default_geom_friction

        if (
            self.randomize_model_geom_friction_tangential or
            self.randomize_model_geom_friction_torsional or
            self.randomize_model_geom_friction_rolling
        ):
            interpolation = jax.random.uniform(model_friction_key, shape=(self.default_geom_friction.shape[0],))
            tangential = self.default_geom_friction[:, 0]
            torsional = self.default_geom_friction[:, 1]
            rolling = self.default_geom_friction[:, 2]
            if self.randomize_model_geom_friction_tangential:
                tangential = self._lerp(self.model_geom_friction_tangential_range, interpolation)
            if self.randomize_model_geom_friction_torsional:
                torsional = self._lerp(self.model_geom_friction_torsional_range, interpolation)
            if self.randomize_model_geom_friction_rolling:
                rolling = self._lerp(self.model_geom_friction_rolling_range, interpolation)
            model_geom_friction = jnp.stack([tangential, torsional, rolling], axis=1)
            geom_friction = jnp.where(
                (self.geom_ids[:, None] == self.floor_geom_id),
                geom_friction,
                model_geom_friction,
            )

        if (
            self.randomize_floor_geom_friction_tangential or
            self.randomize_floor_geom_friction_torsional or
            self.randomize_floor_geom_friction_rolling
        ):
            interpolation = jax.random.uniform(floor_friction_key)
            floor_friction = self.default_geom_friction[self.floor_geom_id]
            tangential = floor_friction[0]
            torsional = floor_friction[1]
            rolling = floor_friction[2]
            if self.randomize_floor_geom_friction_tangential:
                tangential = self._lerp(self.floor_geom_friction_tangential_range, interpolation)
            if self.randomize_floor_geom_friction_torsional:
                torsional = self._lerp(self.floor_geom_friction_torsional_range, interpolation)
            if self.randomize_floor_geom_friction_rolling:
                rolling = self._lerp(self.floor_geom_friction_rolling_range, interpolation)
            geom_friction = geom_friction.at[self.floor_geom_id].set(jnp.array([tangential, torsional, rolling]))

        geom_solref = self.default_geom_solref
        if self.randomize_geom_stiffness or self.randomize_geom_damping:
            stiffness_interpolation = jax.random.uniform(stiffness_key, shape=(self.default_geom_solref.shape[0],))
            damping_interpolation = jax.random.uniform(damping_key, shape=(self.default_geom_solref.shape[0],))
            stiffness = self.default_geom_solref[:, 0]
            damping = self.default_geom_solref[:, 1]
            if self.randomize_geom_stiffness:
                stiffness = -self._lerp(self.geom_stiffness_range, stiffness_interpolation)
            if self.randomize_geom_damping:
                damping = -self._lerp(self.geom_damping_range, damping_interpolation)
            geom_solref = geom_solref.at[:, 0].set(stiffness)
            geom_solref = geom_solref.at[:, 1].set(damping)

        gravity_z = -self._lerp(self.gravity_range, jax.random.uniform(gravity_key))
        opt_gravity = jnp.where(
            self.randomize_gravity,
            jnp.array([0.0, 0.0, gravity_z]),
            self.default_gravity,
        )

        new_mjx_model = mjx_model.tree_replace({
            "geom_friction": geom_friction,
            "geom_solref": geom_solref,
            "opt.gravity": opt_gravity,
        })
        return jax.lax.cond(should_randomize, lambda _: new_mjx_model, lambda _: mjx_model, None)
