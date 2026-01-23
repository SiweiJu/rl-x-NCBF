from copy import deepcopy
from functools import partial
from typing import Sequence, Callable, Optional, Tuple, Dict

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from mujoco import mjx

import jax.scipy as jsp

Array = jnp.ndarray


def get_ncbf(config, env):
    # TODO: make different types of ncbf
    n_ncbf_ensemble = config.algorithm.ncbf.n_enssemble
    use_safety_layer = config.algorithm.ncbf.use_safety_layer
    gamma_c = config.algorithm.ncbf.gamma_c
    eta_cbf = config.algorithm.ncbf.eta_cbf
    lambda_slack = config.algorithm.ncbf.lambda_slack
    ncbf_clipping = config.algorithm.ncbf.action_clipping

    use_robot_safety_layer = config.algorithm.ncbf.use_robust_safety_layer


    NCBF = [NCBF_FFNN(config.algorithm.ncbf.nr_hidden_units) for _ in range(n_ncbf_ensemble)]
    NCBF_apply = get_ensemble_forward_pass(NCBF[0].apply)

    dynamics_step_function = get_dynamics_step_function_mjx(env.envs[0])

    act_low = jnp.array(env.single_action_space.low)
    act_high = jnp.array(env.single_action_space.high)

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function = make_get_safe_action(
            ncbf_apply=NCBF_apply,
            system_forward_dynamics_function=dynamics_step_function,
            state_from_obs_id=env.dynamics_observation_indices,
            ncbf_obs_in_dynamics_state_id=env.ncbf_obs_in_dynamics_state_idx,
            use_safety_layer=use_safety_layer,
            act_low=act_low,
            act_high=act_high,
            gamma_c=gamma_c,
            eta_cbf=eta_cbf,
            lambda_s=lambda_slack,
            use_robust_safety_layer=use_robot_safety_layer,
            action_clipping=ncbf_clipping,
        )

    # dummy
    if ncbf_clipping:
        dummy_safety_layer_function = lambda action_raw, obs_t, phi: (jnp.clip(action_raw, act_low, act_high),
                                                                      jnp.array(False), jnp.array(0.0), jnp.array(0.0),
                                                                      jnp.array(0.0))
    else:
        dummy_safety_layer_function = lambda action_raw, obs_t, phi: (action_raw, jnp.array(False), jnp.array(0.0), jnp.array(0.0),
                                                                      jnp.array(0.0))

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function_for_batch = safety_layer_function
    else:
        safety_layer_function_for_batch = dummy_safety_layer_function

    # Vectorize over env axis 0: (N_env, act_dim), (N_env, obs_dim), phi -> (N_env, act_dim), (N_env, info_struct)
    batched_get_safe_action = jax.jit(
        jax.vmap(
            safety_layer_function_for_batch,
            in_axes=(0, 0, None),  # action_raw[env], obs_t[env], same phi for items in the batch
            out_axes=(0, 0, 0, 0, 0)  # batched u_safe, constraint_active, delta_u, h_u0
        )
    )
    return NCBF, NCBF_apply, batched_get_safe_action, safety_layer_function_for_batch, dynamics_step_function

def get_ensemble_forward_pass(apply_fn):
    alpha = 0.6
    E = 5  # number of ensemble members, hardcoded for now
    k = max(1, int(np.ceil((1.0 - alpha) * E)))
    # print("cvar consider least k:", k)

    @jax.jit
    def ensemble_forward_pass(params_stack, input):
        def single_forward(params):
            return apply_fn(params, input)

        # Use tree_leaves to iterate over actual parameter dicts
        predictions = jax.vmap(single_forward, in_axes=0, out_axes=0)(params_stack)

        def aggregate_predictions(preds):
            # return preds[0]
            # return jnp.mean(preds, axis=0)
            # # def cvar_soft(losses):
            # #     eta = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
            # #     tail = jnp.maximum(losses - eta, 0.0)
            # #     return eta + jnp.mean(tail) / (1 - alpha)
            # #
            def cvar_firstk(x):
                tail = jnp.sort(x, axis=0)[:k, ...]
                return jnp.mean(tail, axis=0)


            # def cvar_gaussian(losses, eps=1e-6):
            #     """
            #     Gaussian-fitted lower-tail CVaR at level alpha.
            #     losses: (E, ...) with larger=worse.
            #     Returns: CVaR_alpha(losses) with same shape as losses[0].
            #     """
            #     mu = jnp.mean(losses, axis=0)
            #     # Use ddof=0 for stability with small E; add eps to avoid sigma=0 issues.
            #     sigma = jnp.std(losses, axis=0) + eps
            #
            #     # z_alpha = Phi^{-1}(alpha)
            #     z = jsp.special.ndtri(alpha)
            #
            #     # phi(z) = standard normal pdf
            #     phi = jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)
            #
            #     # lower-tail CVaR (worst tail)
            #     return mu - sigma * (phi / alpha)

            #

            return cvar_firstk(preds)

        return aggregate_predictions(predictions.squeeze()), jnp.std(predictions, axis=0), predictions
    return ensemble_forward_pass



class NCBF_FFNN(nn.Module):
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        # Two hidden layers
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        # Scalar CBF output h(x)
        h = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # h = nn.sigmoid(h)
        # This output is used in the safety layer to calculate the gradient only, for prediction, use self.predict_with_sigmoid
        return jnp.squeeze(h, -1)  # shape ()


def make_get_safe_action(
    ncbf_apply: Callable[[dict, Array], Array],   # h_phi(obs)
    system_forward_dynamics_function: Callable[[Array, Array], Array],
    state_from_obs_id: Array,
    ncbf_obs_in_dynamics_state_id: Array,
    use_safety_layer: bool,
    act_low: Array,
    act_high: Array,
    gamma_c: float,
    eta_cbf: float,      # \tilde alpha(s) = eta_cbf * s
    lambda_s: float,     # slack penalty
    action_clipping: bool = False,
    lambda_h:float = 0,
    use_robust_safety_layer: bool = False,
):
    """
    Returns a JIT-able safety layer:
        get_safe_action(action_raw, x_t, t, contact, phi) -> (u_safe, info)

    Args:
      ncbf_apply: flax apply function input: apply_fn, parameters_stack, input
      system_forward_dynamics_function: x_next = f(x, u, t, contact)
      state_from_obs_id: indices to get dynamics state from observation [pos(3), quat(4), joint_pos, joint_vel]
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

    @jax.jit
    def a_and_c_from_linearization(obs_t: Array, u0: Array, phi: dict):
        """
        Compute:
          a = d/du h_phi(x_{t+1}(u)) | u0
          c = linearized RHS so constraint is a^T u >= c
        Args:
            obs_t: current observation (full observation matrix, need to get state from it)
        """

        x_t = obs_t[state_from_obs_id] # get dynamics state from observation, remove contact at end

        def h_of_u(u):
            x_next = system_forward_dynamics_function(x_t, u)
            h_input_next = x_next[ncbf_obs_in_dynamics_state_id]
            h, _, _ = ncbf_apply(phi, h_input_next)
            return h # scalar-ish # note here phis is parameter stack for the ensemble

        # for debugging
        x_next = system_forward_dynamics_function(x_t, u0)

        h_u0, lin = jax.linearize(h_of_u, u0)  # forward-mode, works with dynamic loops
        I = jnp.eye(u0.shape[0], dtype=u0.dtype)
        a = jax.vmap(lin)(I)  # (act_dim,) because output is scalar
        a = jnp.reshape(a, (-1,))

        # jax.debug.print("a norm: {na}, a: {a}", na=jnp.linalg.norm(a), a=a)

        # dx_du = jax.jacfwd(lambda u: system_forward_dynamics_function(x_t, u))(u0)  # (state_dim, act_dim)
        # dh_dx = jax.jacfwd(lambda x: ncbf_apply(phi, x[3:]))(x_t)  # (state_dim,)
        #
        # grad_dx_du_norm = jnp.linalg.norm(dx_du)
        # grad_a_norm = jnp.linalg.norm(a)
        # grad_dh_dx_norm = jnp.linalg.norm(dh_dx)
        #
        # jax.debug.print("dx/du norm: {ndx}, dh/du norm: {na}, dh/dx norm: {ndh}", ndx=grad_dx_du_norm, na=grad_a_norm, ndh=grad_dh_dx_norm)

        # h_u0 = h_of_u(u0)

        ncbf_obs_t = x_t[..., ncbf_obs_in_dynamics_state_id]
        h_x, h_x_std, _  = ncbf_apply(phi, ncbf_obs_t)

        # Discrete-time CBF condition:
        #   h(x_{t+1}) - h(x_t) + alpha(h(x_t)-gamma_c) >= 0
        #
        # Linearize h(x_{t+1}(u)):
        #   h(x_{t+1}(u)) ≈ h_u0 + a^T (u-u0)
        #
        # => h_u0 + a^T(u-u0) - h_x + alpha(h_x-gamma_c) >= 0
        # => a^T u >= -h_u0 + h_x - alpha(h_x-gamma_c) + a^T u0  =: c_lin
        c_lin = -h_u0 + h_x - alpha(h_x - gamma_c) + jnp.dot(a, u0)
        return a, c_lin, h_x, h_u0, x_next, h_x_std

    @jax.jit
    def get_safe_action(
        action_raw: Array,
        obs_t: Array,
        phis: dict
    ) -> Tuple[Array, Array, Array, Array, Array]:

        # clip action raw first
        # action_raw = jnp.clip(action_raw, act_low, act_high)
        action_raw = jax.lax.cond(action_clipping, lambda x: jnp.clip(x, act_low, act_high), lambda x: x, action_raw)

        a, c, h_x, h_u0, x_next, h_x_std = a_and_c_from_linearization(obs_t, action_raw, phis)

        aTa = jnp.dot(a, a) + 1e-12
        aTu = jnp.dot(a, action_raw)

        # closed-form QP solution (soft slack)
        delta = jnp.maximum(0.0, c - aTu)
        gain = delta / (aTa + (1.0 / lambda_s))
        u_safe = action_raw + gain * a

        # # # linear penalized
        # b = c - aTu
        # delta = jnp.maximum(0.0, b)
        #
        # A = jnp.sqrt(aTa)  # = ||a||_2   (aTa already has +1e-12 above)
        # eff = A - lambda_h  # effective ascent margin
        #
        # def g_of_u(u):
        #     x_next_u = system_forward_dynamics_function(obs_t[state_from_obs_id], u)
        #     h_input_next_u = x_next_u[ncbf_obs_in_dynamics_state_id]
        #     h_next_u, _, _ = ncbf_apply(phis, h_input_next_u)
        #     # g(u) = h(x_{t+1}(u)) - h(x_t) + alpha(h(x_t)-gamma_c)
        #     return h_next_u - h_x + alpha(h_x - gamma_c)
        #
        # def sqp_filter(u0, max_sqp_iters=10, max_ls_iters=10, beta=0.5, c1=1e-4, eps=1e-8):
        #     """
        #     SQP loop to find u close to u0 that makes g(u) >= 0.
        #     - max_sqp_iters: number of SQP outer iterations
        #     - max_ls_iters: backtracking steps per SQP iteration
        #     - beta: step shrink factor
        #     - c1: Armijo-like sufficient increase parameter
        #     """
        #
        #     def g_and_grad(g, u, eps=1e-12):
        #         # g: R^n -> R  (scalar)
        #         g0, lin = jax.linearize(g, u)  # forward-mode JVP closure
        #         I = jnp.eye(u.shape[0], dtype=u.dtype)
        #         grad = jax.vmap(lin)(I)  # shape (act_dim,)
        #         grad = jnp.reshape(grad, (-1,))
        #         return g0, grad
        #
        #     def one_sqp_iter(u):
        #         gk, ak = g_and_grad(g_of_u, u, eps=eps)
        #
        #         # If already feasible, do nothing.
        #         def feasible(_):
        #             return u, gk, jnp.array(0.0, u.dtype)
        #
        #         def infeasible(_):
        #             aTa = jnp.dot(ak, ak) + eps
        #             # min-norm QP step for linearized constraint (with slack)
        #             step = (-gk) * ak / (aTa + 1.0 / lambda_s)
        #
        #             # Backtracking line search on true g(u)
        #             def ls_body(carry, _):
        #                 u_curr, t, g_curr, accepted = carry
        #                 u_try = u + t * step
        #                 g_try = g_of_u(u_try)
        #
        #                 # Accept if sufficient increase in g (or reach feasibility)
        #                 # For negative gk, "increase" means less negative; Armijo-like:
        #                 # g(u+t*d) >= g(u) + c1 * t * (a^T d)
        #                 rhs = gk + c1 * t * jnp.dot(ak, step)
        #                 ok = (g_try >= 0.0) | (g_try >= rhs)
        #
        #                 u_next = jax.lax.select(ok & (~accepted), u_try, u_curr)
        #                 g_next = jax.lax.select(ok & (~accepted), g_try, g_curr)
        #                 acc_next = accepted | ok
        #                 t_next = jax.lax.select(acc_next, t, t * beta)
        #                 return (u_next, t_next, g_next, acc_next), None
        #
        #             # initialize LS
        #             (u_ls, t_ls, g_ls, acc), _ = jax.lax.scan(
        #                 ls_body,
        #                 (u, jnp.array(1.0, u.dtype), gk, jnp.array(False)),
        #                 xs=None,
        #                 length=max_ls_iters
        #             )
        #
        #             # If never accepted, still take a tiny step (optional); here keep u unchanged.
        #             u_out = jax.lax.select(acc, u_ls, u)
        #             g_out = jax.lax.select(acc, g_ls, gk)
        #             step_norm = jnp.linalg.norm(u_out - u)
        #             return u_out, g_out, step_norm
        #
        #         return jax.lax.cond(gk >= 0.0, feasible, infeasible, operand=None)
        #
        #     def outer_body(u, _):
        #         u_new, g_new, step_norm = one_sqp_iter(u)
        #         return u_new, (g_new, step_norm)
        #
        #     u_final, (g_traj, step_traj) = jax.lax.scan(outer_body, u0, xs=None, length=max_sqp_iters)
        #     return u_final, g_traj[-1], step_traj[-1]

        # u_safe, g_sqp, step_sqp = sqp_filter(u_safe)

        # def inactive_case(_):
        #     # constraint satisfied already
        #     u_safe = action_raw
        #     eps_star = jnp.array(0.0, dtype=action_raw.dtype)
        #     return u_safe, eps_star
        #
        # def active_case(_):
        #     # need to enforce (possibly with slack)
        #     def slack_only(_):
        #         # If ||a|| <= lambda_h, the LHS a^TΔu - lambda_h||Δu|| <= 0 for all Δu aligned with a,
        #         # so the constraint can only be satisfied via slack.
        #         u_safe = action_raw
        #         eps_star = delta
        #         return u_safe, eps_star
        #
        #     def move_along_a(_):
        #         # Closed-form solution for the SOCP with single "affine - norm" constraint
        #         # Δu = step_coeff * a
        #         denom = (1.0 + lambda_s * eff * eff) * A  # scalar
        #         step_coeff = (lambda_s * eff * delta) / denom
        #         u_safe = action_raw + step_coeff * a
        #         eps_star = delta / (1.0 + lambda_s * eff * eff)
        #         return u_safe, eps_star
        #
        #     return jax.lax.cond(eff <= 0.0, slack_only, move_along_a, operand=None)
        #
        # u_safe, eps_star = jax.lax.cond(delta <= 0.0, inactive_case, active_case, operand=None)

        # eps_star = delta / (1.0 + lambda_s * aTa)

        u_processed = jax.lax.cond(use_safety_layer, lambda _: u_safe, lambda _: action_raw, operand=None)
        u_clipped = jax.lax.cond(
            action_clipping,
            lambda x: jnp.clip(x, act_low, act_high),
            lambda x: x,
            u_processed
        )
        delta_u = jnp.linalg.norm(u_clipped - action_raw)

        # # for debugging
        # def barrier_residual(u):
        #     x_next_u = system_forward_dynamics_function(obs_t[state_from_obs_id], u)
        #     h_input_next_u = x_next_u[ncbf_obs_in_dynamics_state_id]
        #     h_next_u, _, _ = ncbf_apply(phis, h_input_next_u)
        #     return h_next_u - h_x + alpha(h_x - gamma_c), h_next_u
        # g_raw, h_next_raw = barrier_residual(action_raw)
        # g_exec, h_next_exec = barrier_residual(u_clipped)
        #
        # def proj_box(u):
        #     return jnp.clip(u, act_low, act_high)
        #
        # # recovery mode
        # def recovery(u_init, step=0.1, K=5, eps=1e-8):
        #     def barrier_residual_of_u(u):
        #         x_next_u = system_forward_dynamics_function(obs_t[state_from_obs_id], u)
        #         h_input_next_u = x_next_u[ncbf_obs_in_dynamics_state_id]
        #         h_next_u, _, _ = ncbf_apply(phis, h_input_next_u)
        #         return h_next_u - h_x + alpha(h_x - gamma_c)
        #
        #     def body(u, _):
        #         g_u0, lin = jax.linearize(barrier_residual_of_u, u)  # forward-mode, works with dynamic loops
        #         I = jnp.eye(u.shape[0], dtype=u.dtype)
        #         grad = jax.vmap(lin)(I)  # (act_dim,) because output is scalar
        #         grad = jnp.reshape(grad, (-1,))
        #
        #         # g_val, grad = jax.value_and_grad(lambda u: barrier_residual(u)[0])(u)                # normalize to avoid huge updates
        #         grad_n = grad / (jnp.linalg.norm(grad) + eps)
        #         u_new = proj_box(u + step * grad_n)
        #         return u_new, g_u0
        #
        #     u_rec, _ = jax.lax.scan(body, u_init, None, length=K)
        #     return u_rec
        #
        # u_rec = jax.lax.cond(g_exec < -0.1, lambda _: recovery(u_clipped), lambda _: u_clipped, operand=None)
        # g_rec = barrier_residual(u_rec)[0]
        # delta_rec = jnp.linalg.norm(action_raw - u_rec)
        #
        # # to debug, evaluate g using linearized h
        # def g_lin_penalized(u):
        #     h_next_lin = h_u0 + jnp.dot(a, (u - action_raw)) - lambda_h * jnp.linalg.norm(u - action_raw)
        #     return h_next_lin - h_x + alpha(h_x - gamma_c)
        #
        # # to debug, evaluate g using linearized h
        # def g_lin(u):
        #     h_next_lin = h_u0 + jnp.dot(a, (u - action_raw))
        #     return h_next_lin - h_x + alpha(h_x - gamma_c)
        #
        #
        # g_lin_raw  = g_lin(action_raw)   # should equal h_u0 - h_x + alpha_term
        # g_lin_exec = g_lin_penalized(u_clipped)
        #
        # # for debugging: check feasibility
        # u_best = jnp.where(a >= 0.0, act_high, act_low)
        # aTu_best = jnp.dot(a, u_best)
        # margin_lin = aTu_best - c  # <0 means infeasible even in linear approximation
        #
        # clip_norm = jnp.linalg.norm(u_clipped - u_processed)
        # sat_frac = jnp.mean((u_clipped <= act_low + 1e-6) | (u_clipped >= act_high - 1e-6))
        #
        # bad = (g_exec < 0.0) | (margin_lin < 0.0) | (clip_norm > 1e-3)
        #
        # def _do_print(_):
        #     jax.debug.print(
        #         "CBF dbg: g_raw={gr:.2e} g_lin_raw={glr:.2e} g_exec={ge:.2e} g_lin_exec={gle:.2e} g_rec={grec:.2e} h_x={hx:.2e} hnr={hnr:.2e} hne={hne:.2e} "
        #         "delta_u={d:.2e} delta_rec={dr:.2e} aTa={ata:.2e} eff_margin={eff:.2e}, margin={m:.2e} clip={cn:.2e} sat={sf:.2f} std={st:.2e}",
        #         gr=g_raw, glr=g_lin_raw, ge=g_exec, gle=g_lin_exec, grec=g_rec, hx=h_x, hnr=h_next_raw, hne=h_next_exec,
        #         d=delta_u, dr=delta_rec, ata=aTa, eff=eff, m=margin_lin, cn=clip_norm, sf=sat_frac, st=h_x_std,
        #         ordered=True,
        #     )
        #     return 0

        # _ = jax.lax.cond(bad, _do_print, lambda _: 0, operand=None)
        return u_clipped, delta, delta_u, x_next, h_u0

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
    nr_substeps = env.nr_substeps
    template_data = mjx.make_data(mjx_model)

    nq, nv = mjx_model.nq, mjx_model.nv

    joint_nominal_positions = jnp.array([env.internal_state["actuator_joint_nominal_positions"]])[0]
    scaling_factor = env.internal_state["scaling_factor"]

    @jax.jit
    def system_dynamics(x, u):
        # harded coded indexing for now
        # qpos : pos(3), quat(4), joint_pos(n_joints)
        # qvel : vel(3), ang_vel(3), joint_vel(n_joints
        # return: qpos and qvel for actuated joints only
        qpos = x[:nq]
        qvel = x[nq:nq+nv]

        # denormalize action
        u = joint_nominal_positions + u * scaling_factor

        data = template_data.replace(qpos=qpos, qvel=qvel, ctrl=u)
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(mjx_model, data), None),
            init=data,
            xs=(),
            length=nr_substeps
        )

        # only return joint qpos and qvel
        return jnp.concatenate([data.qpos, data.qvel], axis=0)

    return system_dynamics


def get_dynamics_step_function_mujoco(env):
    model = deepcopy(env.initial_mj_model)
    nr_substeps = env.nr_substeps
    template_data = mujoco.MjData(model)

    nq, nv = model.nq, model.nv

    joint_nominal_positions = jnp.array([env.internal_state["actuator_joint_nominal_positions"]])[0]
    scaling_factor = env.internal_state["scaling_factor"]

    def system_dynamics(x, u):
        # harded coded indexing for now
        # qpos : pos(3), quat(4), joint_pos(n_joints)
        # qvel : vel(3), ang_vel(3), joint_vel(n_joints
        # return: qpos and qvel for actuated joints only
        qpos = x[:nq]
        qvel = x[nq:nq+nv]

        # denormalize action
        u = joint_nominal_positions + u * scaling_factor

        template_data.qpos= qpos
        template_data.qvel = qvel
        template_data.ctrl = u
        mujoco.mj_forward(model, template_data)
        mujoco.mj_step(model, template_data, nr_substeps)

        # only return joint qpos and qvel
        return jnp.concatenate([template_data.qpos, template_data.qvel], axis=0)

    return system_dynamics

