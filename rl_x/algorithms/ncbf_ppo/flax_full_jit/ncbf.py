from copy import deepcopy
from functools import partial
from typing import Sequence, Callable, Optional, Tuple, Dict
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from mujoco import mjx

Array = jnp.ndarray


def get_ncbf(config, env):
    # TODO: make different types of ncbf
    ncbf_type = config.algorithm.ncbf.type
    n_ncbf_ensemble = config.algorithm.ncbf.n_ensemble
    use_safety_layer = config.algorithm.ncbf.use_safety_layer
    ncbf_observation_indices = getattr(env, "ncbf_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    gamma_c = config.algorithm.ncbf.gamma_c

    act_low = jnp.array(env.single_action_space.low)
    act_high = jnp.array(env.single_action_space.high)


    NCBF = [NCBF_FFNN(config.algorithm.ncbf.nr_hidden_units, ncbf_observation_indices) for _ in range(n_ncbf_ensemble)]

    dynamics_step_function = get_dynamics_step_function_mjx(env)

    safety_layer_function = make_get_safe_action(NCBF[0].apply, dynamics_step_function, env.dynamics_observation_indices,
                                                 use_safety_layer, act_low, act_high, gamma_c=gamma_c)

    # dummy onnly clipping
    dummy_safety_layer_function = lambda action_raw, obs_t, phi: (jnp.clip(action_raw, act_low, act_high), jnp.array(False), jnp.array(0.0))

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function_for_batch = safety_layer_function
    else:
        safety_layer_function_for_batch = dummy_safety_layer_function

    # Vectorize over env axis 0: (N_env, act_dim), (N_env, obs_dim), phi -> (N_env, act_dim), (N_env, info_struct)
    batched_get_safe_action = jax.jit(
        jax.vmap(
            safety_layer_function_for_batch,
            in_axes=(0, 0, None),  # action_raw[env], obs_t[env], same phi for items in the batch
            out_axes=(0, 0, 0)  # batched u_safe, constraint_active, delta_u
        )
    )
    return NCBF, ensemble_forward_pass, batched_get_safe_action, safety_layer_function


@jax.jit
def ensemble_forward_pass(train_states, input):
    """
    one step forward pass through an ensemble of networks, 1 input
    """

    apply_fn = train_states[0].apply_fn

    params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in train_states])
    # Vectorized apply over ensemble, then take mean
    predictions = jax.vmap(lambda p: apply_fn(p, input))(params_stack)

    def aggregate_predictions(preds):
        # return jnp.mean(preds, axis=0), jnp.std(preds, axis=0)
        def cvar_soft(losses, alpha=0.90):
            eta = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
            tail = jnp.maximum(losses - eta, 0.0)
            return eta + jnp.mean(tail) / (1 - alpha)

        def cvar_topk(losses, alpha=0.80):
            # losses: (E,) or (E, ...) , larger = worse
            E = losses.shape[0]
            k = jnp.maximum(1, jnp.int32(jnp.ceil((1.0 - alpha) * E)))
            tail = jnp.sort(losses, axis=0)[-k:, ...]
            return jnp.mean(tail, axis=0)

        risks = 1 - preds
        cvar = 1 - cvar_topk(risks)
        return cvar, jnp.std(preds), preds

    return aggregate_predictions(predictions.squeeze())


class NCBF_FFNN(nn.Module):
    nr_hidden_units: int
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.observation_indices]
        # Two hidden layers
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        # Scalar CBF output h(x)
        h1 = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # clip the output to be in [0, 1]
        # h1 = nn.sigmoid(h1)
        # to get the gradient without the sigmoid, we do not use sigmoid here, add a sigmoid function when preedicting

        h1 = jnp.squeeze(h1, -1)  # shape ()
        return h1


def make_get_safe_action(
    ncbf_apply: Callable[[dict, Array], Array],   # h_phi(obs)
    system_forward_dynamics_function: Callable[[Array, Array], Array],
    state_from_obs_id: Array,
    use_safety_layer: bool,
    act_low: Array,
    act_high: Array,
    *,
    gamma_c: float = 0.0,
    eta_cbf: float = 1.0,      # \tilde alpha(s) = eta_cbf * s
    lambda_s: float = 1e3,     # slack penalty
):
    """
    Returns a JIT-able safety layer:
        get_safe_action(action_raw, x_t, t, contact, phi) -> (u_safe, info)

    Args:
      ncbf_apply: flax apply function h_phi(obs)
      system_forward_dynamics_function: x_next = f(x, u, t, contact) (your MJX WBD step)
      state_from_obs_id: indices to get dynamics state from observation [pos(3), quat(4), joint_pos, joint_vel, contact (4)]
        use_safety_layer: whether to use safety layer (if False, returns raw action, but still calculates constraint violations)
      gamma_c: conservative margin
      eta_cbf: class-K gain
      lambda_s: slack penalty (large -> hard projection)
      obs_from_state: optional mapping if your NCBF uses an observation (not full state).
                      If None, we assume NCBF takes the raw state x_t / x_next directly.

    Notes:
      - Uses linearization:
          h_phi(x_{t+1}(u)) ≈ h_phi(x_{t+1}(u0)) + a^T (u - u0)
        where a = d/du h_phi(x_{t+1}(u))|_{u0}
      - No explicit g(x) needed.
    """

    def alpha(s: Array) -> Array:
        return eta_cbf * s


    def a_and_c_from_linearization(obs_t: Array, u0: Array, phi: dict):
        """
        Compute:
          a = d/du h_phi(x_{t+1}(u)) | u0
          c = linearized RHS so constraint is a^T u >= c
        Args:
            obs_t: current observation (full observation matrix, need to get state from it)
        """

        def h_of_u(x_t, u):
            # derivtives needs to be take for u only, x_t fixed
            def h(u):
                x_next = system_forward_dynamics_function(x_t, u)
                obs_next = x_next[3:]
                return ncbf_apply(phi, obs_next)  # scalar-ish
            return h(u)

        x_t = obs_t[state_from_obs_id][:-4]  # get dynamics state from observation, remove contact at end

        # a = ∂/∂u h(f(x,u)) at u0
        a = jax.jacrev(h_of_u, argnums=1)(x_t, u0)  # (m,)
        h_u0 = h_of_u(x_t, u0)

        ncbf_obs_t = x_t[3:]
        h_x  = ncbf_apply(phi, ncbf_obs_t)

        # Discrete-time CBF condition:
        #   h(x_{t+1}) - h(x_t) + alpha(h(x_t)-gamma_c) >= 0
        #
        # Linearize h(x_{t+1}(u)):
        #   h(x_{t+1}(u)) ≈ h_u0 + a^T (u-u0)
        #
        # => h_u0 + a^T(u-u0) - h_x + alpha(h_x-gamma_c) >= 0
        # => a^T u >= -h_u0 + h_x - alpha(h_x-gamma_c) + a^T u0  =: c_lin
        c_lin = -h_u0 + h_x - alpha(h_x - gamma_c) + jnp.dot(a, u0)
        return a, c_lin, h_x, h_u0

    @jax.jit
    def get_safe_action(
        action_raw: Array,
        obs_t: Array,
        phi: dict
    ) -> Tuple[Array, Array, Array]:
        a, c, h_x, h_u0 = a_and_c_from_linearization(obs_t, action_raw, phi)

        aTa = jnp.dot(a, a) + 1e-12
        aTu = jnp.dot(a, action_raw)

        # constraint violation amount
        delta = jnp.maximum(0.0, c - aTu)

        # closed-form QP solution (soft slack)
        gain = delta / (aTa + (1.0 / lambda_s))
        u_safe = action_raw + gain * a

        # eps_star = delta / (1.0 + lambda_s * aTa)

        # info = {
        #     # "a": a,
        #     # "c": c,
        #     # "delta": delta,
        #     # "gain": gain,
        #     # "epsilon_star": eps_star,
        #     "constraint_active": (delta > 0.0),
        #     # "h_x": h_x,
        #     # "h_u0": h_u0,
        # }
        constraint_active = jnp.array(delta > 0.0)

        u_processed = jax.lax.cond(use_safety_layer, lambda _: u_safe, lambda _: action_raw, operand=None)

        u_clipped = jnp.clip(u_processed, act_low, act_high)
        action_raw_clipped = jnp.clip(action_raw, act_low, act_high)
        delta_u = jnp.linalg.norm(u_clipped - action_raw_clipped)

        return u_clipped, constraint_active, delta_u

    return get_safe_action

def get_dynamics_step_function(env):
    def quadruped_wb_dynamics(mjx_model, contact_id, body_id, n_joints, dt, x, u, contact):
        """
        Compute the whole-body dynamics of a quadruped robot using forward dynamics and contact forces.

        Args:
            mjx_model: The MuJoCo XLA model object for the simulation.
            contact_id (list): List of contact point (foot geometry id) IDs for each leg. [FL, FR, RL, RR]
            body_id (list): List of body IDs for each leg. [FL, FR, RL, RR]
            n_joints (int): Number of joints in the quadruped.
            dt (float): Time step for the simulation.
            x (jnp.ndarray): Current state vector [position, orientation, joint positions, velocities].
            u (jnp.ndarray): Control input vector (torques for the joints).
            contact (jnp.array): Contact parameters for each foot at the current time step.

        Returns:
            jnp.ndarray: The updated state vector after applying dynamics and contact forces.
        """
        # Create a new data object for the simulation
        mjx_data = mjx.make_data(mjx_model)
        # Update the position and velocity in the data object
        mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

        # Perform forward kinematics and dynamics computations
        mjx_data = mjx.fwd_position(mjx_model, mjx_data)
        mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

        # Extract the mass matrix and bias forces
        M = mjx_data.qLD
        D = mjx_data.qfrc_bias

        # Create the torque vector, with zeros for the base and control inputs for the joints
        tau = jnp.concatenate([jnp.zeros(6), u])

        # Get the positions of the contact points on the legs
        FL_leg = mjx_data.geom_xpos[contact_id[0]]
        FR_leg = mjx_data.geom_xpos[contact_id[1]]
        RL_leg = mjx_data.geom_xpos[contact_id[2]]
        RR_leg = mjx_data.geom_xpos[contact_id[3]]

        # Compute the Jacobians for each leg
        J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
        J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
        J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
        J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

        # Concatenate the Jacobians into a single matrix
        J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
        # Concatenate the positions of the legs into a single vector
        current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
        alpha = 25
        # Compute the velocity-level constraint violation
        g_dot = J.T @ x[n_joints+7:13+2*n_joints]
        # Compute the stabilization term
        baumgarte_term = -2 * alpha * g_dot

        # Compute the inverse of the mass matrix projected onto the constraint Jacobian
        JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
        # Compute the right-hand side of the constraint force equation
        rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False), tau - D) + baumgarte_term
        # Solve for the ground reaction forces
        cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
        grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)
        # Apply the contact forces only to the legs that are in contact
        grf = jnp.concatenate([grf[:3]*contact[0], grf[3:6]*contact[1], grf[6:9]*contact[2], grf[9:12]*contact[3]])
        # Update the velocity using the computed forces
        v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt
        # Perform semi-implicit Euler integration to update the position and orientation
        p = x[:3] + v[:3] * dt
        quat = math.quat_integrate(x[3:7], v[3:6], dt)
        q = x[7:7+n_joints] + v[6:6+n_joints] * dt
        # Concatenate the updated state variables into a single vector
        x_next = jnp.concatenate([p, quat, q, v, current_leg, grf])

        return x_next

    model = deepcopy(env.initial_mj_model)
    mjx_model = mjx.put_model(model)

    contact_id = env.foot_geom_ids
    body_id = env.body_ids_of_feet
    n_joints = mjx_model.njnt
    dt = env.dt

    return jax.jit(lambda x, u, contact: quadruped_wb_dynamics(mjx_model, contact_id, body_id, n_joints, dt, x, u, contact))

def get_dynamics_step_function_mjx(env):
    model = deepcopy(env.initial_mj_model)
    mjx_model = mjx.put_model(model)
    n_joints = env.nr_actuator_joints
    nr_substeps = env.nr_substeps

    @jax.jit
    def system_dynamics(x, u):
        data = mjx.make_data(mjx_model)

        # harded coded indexing for now
        # qpos : pos(3), quat(4), joint_pos(n_joints)
        # qvel : vel(3), ang_vel(3), joint_vel(n_joints
        # pos(3) is not necessar
        qpos = data.qpos
        qpos = qpos.at[3:7 + n_joints].set(x[:4 + n_joints])
        qvel = x[7 + n_joints:]

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=u)
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(mjx_model, data), None),
            init=data,
            xs=(),
            length=nr_substeps
        )
        qpos = data.qpos
        qvel = data.qvel
        return jnp.concatenate([qpos, qvel], axis=0)

    return system_dynamics