from copy import deepcopy
from typing import Sequence, Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


def get_ncbf(config, env):
    # TODO: make different types of ncbf
    ncbf_type = config.algorithm.ncbf.type
    ncbf_observation_indices = getattr(env, "ncbf_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    NCBF = NCBF_FFNN(config.algorithm.nr_hidden_units, ncbf_observation_indices)
    dynamics_step_function = get_dynamics_step_function(config)
    safety_layer_function = make_get_safe_action(NCBF.apply, dynamics_step_function)

    return (NCBF, safety_layer_function)


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
        h = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return jnp.squeeze(h, -1)  # shape ()


def make_get_safe_action(
    ncbf_apply: Callable[[dict, jnp.ndarray], jnp.ndarray],  # h_phi(x)
    system_forward_dynamics_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], # x_{t+1} = f(x_t, u_t)
    *,
    # Option A: known control-affine dynamics x_{t+1} = f(x_t) + g(x_t) u_t
    f: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    g: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    # Option B: learned differentiable one-step predictor x_{t+1} = fhat(x_t, u_t)
    fhat: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    gamma_c: float = 0.0,
    eta_cbf: float = 1.0,      # \tilde alpha(s) = eta_cbf * s   Kappa function
    lambda_s: float = 1e3      # slack penalty (large -> hard projection)
):
    def alpha(s: jnp.ndarray) -> jnp.ndarray:
        # kappa function for CBF constraint
        return eta_cbf * s

    # --------- Option A: Known control-affine dynamics ---------
    if f is not None and g is not None:
        # a = (∇h(x_t))^T g(x_t)
        def compute_a_c_known(x_t: jnp.ndarray, phi: dict) -> (jnp.ndarray, jnp.ndarray):
            grad_h = jax.grad(ncbf_apply, argnums=1)(phi, x_t)                    # (n,)
            gx = g(x_t)                                      # (n, m)
            fx = f(x_t)                                      # (n,)
            a = gx.T @ grad_h                                # (m,)
            h_vals = ncbf_apply(phi, x_t)
            c = - grad_h @ fx - alpha(h_vals - gamma_c)   # scalar
            return a, c

        @jax.jit
        def get_safe_action(action_raw, x_t, phi):
            a, c = compute_a_c_known(x_t, phi)
            aTa = jnp.dot(a, a) + 1e-12                      # numeric safety
            aTu = jnp.dot(a, action_raw)
            # Use ReLU on (c - a^T u_raw) to keep identity when constraint inactive
            delta = jnp.maximum(0.0, c - aTu)
            gain = delta / (aTa + (1.0 / lambda_s))
            u_safe = action_raw + gain * a
            # epsilon* as in the piecewise expression
            eps_star = delta / (1.0 + lambda_s * aTa)
            info = {
                "a": a, "c": c,
                "delta": delta,
                "gain": gain,
                "epsilon_star": eps_star,
                "constraint_active": (delta > 0.0)
            }
            return u_safe, info

        return get_safe_action
    else:
        @jax.jit
        def get_safe_action(action_raw: jnp.ndarray, x_t: jnp.ndarray):
            # No safety layer implemented for learned dynamics yet
            return action_raw, {}
        return get_safe_action

    # TODO, decide how to handle Option B, learn overal funciton or control affine, and how to parse it

    # # --------- Option B: Learned dynamics with linearization around action_raw ---------
    # else:
    #     # We will linearize h(x_{t+1}(u)) at u0 = action_raw:
    #     #   h(x_{t+1}(u)) ≈ h(u0) + a^T (u - u0),
    #     #   where a = d/du h(x_{t+1}(u))|_{u0} = (∇_x h)(x_{t+1}) @ (∂x_{t+1}/∂u).
    #     def a_and_c_from_linearization(x_t: jnp.ndarray, u0: jnp.ndarray):
    #         def h_of_u(u):
    #             x_next = fhat(x_t, u)                       # (n,)
    #             return ncbf(x_next)                         # scalar
    #
    #         # a = ∂/∂u h(fhat(x,u)) |_{u0}
    #         a = jax.jacrev(h_of_u)(u0)                      # (m,)
    #         h_u0 = h_of_u(u0)
    #         h_x  = ncbf(x_t)
    #         # From: h(u) - h(x_t) + alpha(h(x_t)-gamma_c) >= 0
    #         # Linearized: a^T(u-u0) + h(u0) - h(x_t) + alpha(...) >= 0
    #         # => a^T u >= [ -h(u0) + h(x_t) - alpha(...) + a^T u0 ] =: c_lin
    #         c_lin = - h_u0 + h_x - alpha(h_x - gamma_c) + jnp.dot(a, u0)
    #         return a, c_lin
    #
    #     @jax.jit
    #     def get_safe_action(action_raw: jnp.ndarray, x_t: jnp.ndarray):
    #         a, c = a_and_c_from_linearization(x_t, action_raw)
    #         aTa = jnp.dot(a, a) + 1e-12
    #         aTu = jnp.dot(a, action_raw)
    #         delta = jnp.maximum(0.0, c - aTu)
    #         gain = delta / (aTa + (1.0 / lambda_s))
    #         u_safe = action_raw + gain * a
    #         eps_star = delta / (1.0 + lambda_s * aTa)
    #         info = {
    #             "a": a, "c": c,
    #             "delta": delta,
    #             "gain": gain,
    #             "epsilon_star": eps_star,
    #             "constraint_active": (delta > 0.0)
    #         }
    #         return u_safe, info
    #
    #     return get_safe_action

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

    return jax.jit(lambda x, u, t, contact: quadruped_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, contact))