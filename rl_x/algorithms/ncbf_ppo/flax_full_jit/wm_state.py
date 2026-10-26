"""World-model state, DR parameter vector xi, action delay, policy observation and analytic margins (T11).

Project plan 4.2 scheme 1 ("observation + privileged quantities"), 4.3 (delay in imagination), 4.4 method A
(x^- after physics + qvel clip, before shocks / DR resampling; x^+ after), 4.5 (analytic margins).

Everything that touches the env (`extract_state`, `extract_xi`, `realized_delay_steps`) works on ONE env (vmap it);
the functions used in imagination (`delay_step`, `make_policy_obs_fn`, `margins`, `margin_fall`, window-frame
helpers) are pure jnp, broadcast over leading batch axes, and are jit / grad safe.

Conventions (see schema.json written by `schema_dict`):
  * The state x reads the IMU quantities exactly where the env reads them for the observation and the termination:
    `data.site_xmat` / `data.site_xpos` of the "imu" site and the gyro `data.sensordata`. MuJoCo computes those at
    the start of the last physics substep, and the shock / DR step changes qpos / qvel without recomputing them.
    So in x^- and x^+ of the same step only the qpos/qvel-derived groups (joint_pos, joint_vel, base_lin_vel,
    base_xy) differ; this is exactly what the policy observation and the termination of the env see.
  * At an episode start the env does not run mjx.forward after sampling the initial state, so the IMU groups of
    the first state of an episode are the ones of the initial mjx_data (identity orientation, zero gyro). The
    first transition of each episode (episode_step == 0) is therefore not a clean physics transition: mask it.
  * Positions / yaw are stored absolute (world frame); `window_origin` + `to_window_frame` give the
    window-start frame (plan 4.2 invariance).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


SCHEMA_VERSION = "pncbf-p1-wmdata/v1"

# Policy observation scaling of `LocomotionEnv.get_observation` (normalization branch).
OBS_JOINT_POS_SCALE = 3.14
OBS_JOINT_VEL_SCALE = 100.0
OBS_ACTION_SCALE = 10.0
OBS_ANG_VEL_SCALE = 50.0
OBS_CLIP = 10.0

QVEL_LIMIT = 100.0  # env: max_qvel of the free joint; termination when |qvel[:3]| == 100 (clipped)


# ----------------------------------------------------------------------------------------------------------------
# Layouts
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Layout:
    """Ordered named groups over a flat vector."""
    groups: List[Tuple[str, int]]
    units: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def dim(self):
        return int(sum(size for _, size in self.groups))

    @property
    def slices(self):
        out, start = {}, 0
        for name, size in self.groups:
            out[name] = slice(start, start + size)
            start += size
        return out

    def get(self, vec, name):
        return vec[..., self.slices[name]]

    def to_json(self):
        return {
            "dim": self.dim,
            "groups": {name: [s.start, s.stop] for name, s in self.slices.items()},
            "units": {name: self.units.get(name, "") for name, _ in self.groups},
            "notes": {name: self.notes[name] for name, _ in self.groups if name in self.notes},
        }


def state_layout(env) -> Layout:
    nj = int(env.nr_actuator_joints)
    groups = [
        ("joint_pos", nj),
        ("joint_vel", nj),
        ("imu_ang_vel", 3),
        ("projected_gravity", 3),
        ("last_action", nj),
        ("base_lin_vel", 3),
        ("imu_height", 1),
        ("roll_pitch", 2),
        ("base_xy", 2),
        ("base_yaw", 1),
    ]
    units = {
        "joint_pos": "rad (absolute qpos of the actuated joints, not minus nominal)",
        "joint_vel": "rad/s",
        "imu_ang_vel": "rad/s, IMU frame (gyro sensordata)",
        "projected_gravity": "unit vector, IMU frame: R_imu^T (0, 0, -1)",
        "last_action": "policy action units (the chosen, undelayed action of the previous step; = obs prev-action * 10)",
        "base_lin_vel": "m/s, trunk-origin linear velocity qvel[0:3] expressed in the IMU frame",
        "imu_height": "m, IMU site height over the ground (env robot_imu_height_over_ground)",
        "roll_pitch": "rad, euler 'xyz' of the IMU orientation (env imu_orientation_euler[0:2])",
        "base_xy": "m, world frame, trunk origin qpos[0:2]",
        "base_yaw": "rad, world frame, euler 'xyz'[2] of the IMU orientation",
    }
    notes = {
        "imu_ang_vel": "from data of the last physics substep start; unchanged by shocks (see module doc)",
        "projected_gravity": "redundant with roll_pitch (g = [sin p, -cos p sin r, -cos p cos r]); kept because the policy reads it",
        "base_xy": "absolute; use to_window_frame for the window-start frame",
        "base_yaw": "absolute, wrapped to (-pi, pi]; use to_window_frame",
    }
    return Layout(groups, units, notes)


def xi_layout(env) -> Layout:
    m = env.initial_mj_model
    nb = int(m.nbody) - 1
    nj = int(m.njnt) - 1
    nf = int(len(env.foot_geom_indices))
    na = int(env.nr_actuator_joints)
    ndof = int(m.nv) - 6
    groups = [
        # mjx_model fields changed by the seen-robot / unseen-robot / mujoco-model / joint-dropout DR
        ("body_mass", nb), ("body_inertia", 3 * nb), ("body_ipos", 3 * nb), ("body_iquat", 4 * nb),
        ("body_pos", 3 * nb), ("body_quat", 4 * nb), ("imu_site_pos", 3), ("foot_geom_size", 3 * nf),
        ("jnt_axis", 3 * nj), ("torque_limit", na), ("jnt_range", 2 * nj),
        ("dof_damping", ndof), ("dof_armature", ndof), ("jnt_stiffness", nj), ("dof_frictionloss", ndof),
        ("kp", na), ("kd", na),
        ("foot_friction", 3 * nf), ("contact_solref", 2 * (nf + 1)), ("foot_solimp", 5 * nf),
        ("opt_impratio", 1), ("opt_gravity", 3), ("opt_density", 1), ("opt_viscosity", 1),
        # internal_state parameters
        ("nominal_joint_pos", na), ("joint_max_vel", na), ("joint_position_offset", na), ("action_scaling", 1),
        ("dropout_open_mask", na), ("dropout_lock_mask", na),
        ("delay_mixed", 1), ("delay_nr_steps", 1), ("curriculum_coeff", 1),
        ("nominal_imu_height", 1), ("ground_height", 1),
    ]
    units = {
        "body_mass": "kg, bodies 1..nbody-1", "body_inertia": "kg m^2 (diag, body-major)", "body_ipos": "m (CoM)",
        "body_iquat": "wxyz", "body_pos": "m", "body_quat": "wxyz", "imu_site_pos": "m",
        "foot_geom_size": "m (sphere: [0] = radius)", "jnt_axis": "unit vectors of the hinge joints",
        "torque_limit": "Nm (actuator_forcerange[:, 1])", "jnt_range": "rad, [lo, hi] per hinge joint (locked joints: nominal +- 0.001)",
        "kp": "Nm/rad (actuator_gainprm[:, 0], 0 for open-dropout joints)", "kd": "Nms/rad (-actuator_biasprm[:, 2])",
        "foot_friction": "[tangential, torsional, rolling] per foot", "contact_solref": "[timeconst, dampratio] of floor + feet",
        "foot_solimp": "5 per foot", "opt_gravity": "m/s^2 (xy randomized too)",
        "nominal_joint_pos": "rad (PD target = nominal + action * action_scaling + joint_position_offset)",
        "joint_max_vel": "rad/s (joint qvel clip after each control step)", "joint_position_offset": "rad (unseen PD target offset)",
        "action_scaling": "rad per action unit", "dropout_open_mask": "1 = actuator works, 0 = open (no torque)",
        "dropout_lock_mask": "1 = free, 0 = locked at nominal", "delay_mixed": "1 = per-step random delay in {0..max}",
        "delay_nr_steps": "fixed delay when not mixed", "curriculum_coeff": "env curriculum (1 in eval)",
        "nominal_imu_height": "m (termination threshold base)", "ground_height": "m (plane center height)",
    }
    return Layout(groups, units)


# ----------------------------------------------------------------------------------------------------------------
# Extraction from the env (single env; vmap)
# ----------------------------------------------------------------------------------------------------------------
def _imu_rotation(env, data):
    return Rotation.from_matrix(data.site_xmat[env.imu_site_id].reshape(3, 3))


def extract_state(env, data, internal_state, last_action):
    """x for ONE env from mjx data (+ internal_state for the ground height). `last_action` is the chosen
    (undelayed) action of the previous control step (the env's previous-action observation)."""
    rotation = _imu_rotation(env, data)
    euler = rotation.as_euler("xyz")
    inverse = rotation.inv()
    gravity = inverse.apply(jnp.array([0.0, 0.0, -1.0]))
    gyro = data.sensordata[env.imu_angular_velocity_sensor_adr:env.imu_angular_velocity_sensor_adr + env.imu_angular_velocity_sensor_dim]
    base_lin_vel = inverse.apply(data.qvel[:3])
    imu_pos = data.site_xpos[env.imu_site_id]
    height = imu_pos[2] - env.terrain_function.ground_height_at(internal_state, imu_pos[0], imu_pos[1])
    return jnp.concatenate([
        data.qpos[env.actuator_joint_mask_qpos],
        data.qvel[env.actuator_joint_mask_qvel],
        gyro,
        gravity,
        jnp.asarray(last_action, dtype=jnp.float32).reshape(-1),
        base_lin_vel,
        jnp.reshape(height, (1,)),
        euler[:2],
        data.qpos[:2],
        euler[2:3],
    ]).astype(jnp.float32)


def extract_xi(env, mjx_model, internal_state):
    """xi for ONE env: every dynamics-changing DR parameter of the env's current model / internal state."""
    feet = jnp.asarray(env.foot_geom_indices)
    contact_geoms = jnp.concatenate([jnp.array([env.floor_geom_id]), feet])
    na = env.nr_actuator_joints
    f32 = lambda v: jnp.asarray(v, dtype=jnp.float32).reshape(-1)
    ones = jnp.ones(na)
    parts = [
        mjx_model.body_mass[1:], mjx_model.body_inertia[1:], mjx_model.body_ipos[1:], mjx_model.body_iquat[1:],
        mjx_model.body_pos[1:], mjx_model.body_quat[1:], mjx_model.site_pos[env.imu_site_id], mjx_model.geom_size[feet],
        mjx_model.jnt_axis[1:], mjx_model.actuator_forcerange[:, 1], mjx_model.jnt_range[1:],
        mjx_model.dof_damping[6:], mjx_model.dof_armature[6:], mjx_model.jnt_stiffness[1:], mjx_model.dof_frictionloss[6:],
        mjx_model.actuator_gainprm[:, 0], -mjx_model.actuator_biasprm[:, 2],
        mjx_model.geom_friction[feet], mjx_model.geom_solref[contact_geoms], mjx_model.geom_solimp[feet],
        mjx_model.opt.impratio, mjx_model.opt.gravity, mjx_model.opt.density, mjx_model.opt.viscosity,
        internal_state["actuator_joint_nominal_positions"], internal_state["actuator_joint_max_velocities"],
        internal_state.get("position_offsets", jnp.zeros(na)), internal_state.get("scaling_factor", 1.0),
        internal_state.get("joint_dropout_open_mask", ones), internal_state.get("joint_dropout_lock_mask", ones),
        internal_state.get("action_current_mixed", 0.0), internal_state.get("action_current_nr_delay_steps", 0.0),
        internal_state["env_curriculum_coeff"], internal_state["robot_nominal_imu_height_over_ground"],
        internal_state["center_height"],
    ]
    return jnp.concatenate([f32(p) for p in parts])


def realized_delay_steps(env, internal_state, action_delay_key):
    """Delay (control steps) the env's DefaultActionDelay applies in this step: same formula and key as
    `delay_action` (call with the internal_state BEFORE the step's DR / with the step's action_delay_key)."""
    fn = env.domain_randomization_action_delay_function
    max_nr = getattr(fn, "max_nr_delay_steps", None)
    if max_nr is None:
        return jnp.asarray(0, dtype=jnp.int32)
    return sample_delay_steps(internal_state["action_current_mixed"], internal_state["action_current_nr_delay_steps"],
                              internal_state["env_curriculum_coeff"], action_delay_key, max_nr)


def delay_buffer(env, internal_state):
    """Action history the delay function reads from (shape [max_delay + 1, A]); zeros [1, A] without delay DR."""
    if "action_history" in internal_state:
        return internal_state["action_history"]
    return jnp.zeros((1, env.nr_actuator_joints))


# ----------------------------------------------------------------------------------------------------------------
# Imagination helpers (pure, batched)
# ----------------------------------------------------------------------------------------------------------------
def sample_delay_steps(mixed, nr_delay_steps, curriculum_coeff, key, max_nr_delay_steps):
    """Env formula: d = ceil((mixed ? U{0..max} : nr_delay_steps) * curriculum_coeff). Unbatched key."""
    random_steps = jax.random.randint(key, (1,), 0, max_nr_delay_steps + 1)[0]
    return jnp.ceil(jnp.where(mixed, random_steps, nr_delay_steps) * curriculum_coeff).astype(jnp.int32)


def delay_step(buffer, u, delay_steps):
    """Push u into the delay buffer and return (u_tilde, buffer'). buffer [..., M+1, A] (oldest first, the env's
    internal_state["action_history"]), u [..., A], delay_steps [...] int in [0, M]. Same as
    DefaultActionDelay.delay_action given the realized delay; differentiable w.r.t. u."""
    new_buffer = jnp.concatenate([buffer[..., 1:, :], u[..., None, :]], axis=-2)
    index = new_buffer.shape[-2] - 1 - jnp.asarray(delay_steps, dtype=jnp.int32)
    u_tilde = jnp.take_along_axis(new_buffer, index[..., None, None], axis=-2)[..., 0, :]
    return u_tilde, new_buffer


def delay_steps_from_xi(xi, xi_lay, key=None, max_nr_delay_steps=1):
    """Delay for imagination from xi: the fixed delay when not mixed; when mixed, a random draw if `key` is given
    (shape of the batch), else the worst case max_nr_delay_steps."""
    mixed = xi_lay.get(xi, "delay_mixed")[..., 0] > 0.5
    fixed = xi_lay.get(xi, "delay_nr_steps")[..., 0]
    coeff = xi_lay.get(xi, "curriculum_coeff")[..., 0]
    if key is None:
        random_steps = jnp.full(mixed.shape, max_nr_delay_steps)
    else:
        random_steps = jax.random.randint(key, mixed.shape, 0, max_nr_delay_steps + 1)
    return jnp.ceil(jnp.where(mixed, random_steps, fixed) * coeff).astype(jnp.int32)


def policy_obs_blocks(env):
    """Check that the env's policy observation is [joint pos, joint vel, prev action, imu ang vel, command,
    gravity] (Go2 / G1 without ball-plate, obstacles or policy exteroception) and return the block order."""
    expected = np.concatenate([np.asarray(env.joint_positions_obs_idx), np.asarray(env.joint_velocities_obs_idx),
                               np.asarray(env.joint_previous_actions_obs_idx), np.asarray(env.imu_angular_vel_obs_idx),
                               np.asarray(env.goal_velocities_obs_idx), np.asarray(env.gravity_vector_obs_idx)])
    actual = np.asarray(env.policy_observation_indices)
    if actual.shape != expected.shape or not np.all(actual == expected):
        raise NotImplementedError("policy_obs_from_state supports the plain locomotion policy observation only "
                                  "(no ball-plate / obstacle / exteroception blocks)")
    if getattr(env.observation_noise_function, "handles_normalization", False):
        raise NotImplementedError("observation noise function with its own normalization")
    return ["joint_pos", "joint_vel", "last_action", "imu_ang_vel", "command", "projected_gravity"]


def policy_obs_from_state(x, command, nominal_joint_pos, lay):
    """Noise-free policy observation o(x) in the order of env.policy_observation_indices.
    x [..., S], command [..., 3], nominal_joint_pos [..., A] (xi group nominal_joint_pos)."""
    obs = jnp.concatenate([
        (lay.get(x, "joint_pos") - nominal_joint_pos) / OBS_JOINT_POS_SCALE,
        lay.get(x, "joint_vel") / OBS_JOINT_VEL_SCALE,
        lay.get(x, "last_action") / OBS_ACTION_SCALE,
        jnp.clip(lay.get(x, "imu_ang_vel") / OBS_ANG_VEL_SCALE, -1.0, 1.0),
        command,
        lay.get(x, "projected_gravity"),
    ], axis=-1)
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.clip(obs, -OBS_CLIP, OBS_CLIP)


def make_policy_obs_fn(env):
    """Returns (o(x, command, xi) -> policy obs [..., P], o_full(x, command, xi) -> full-size obs with the policy
    entries filled and zeros elsewhere, usable as `policy.apply(params, o_full)`)."""
    policy_obs_blocks(env)
    lay, xl = state_layout(env), xi_layout(env)
    indices = jnp.asarray(env.policy_observation_indices)
    obs_dim = int(env.single_observation_space.shape[0])

    def obs_fn(x, command, xi):
        return policy_obs_from_state(x, command, xl.get(xi, "nominal_joint_pos"), lay)

    def full_obs_fn(x, command, xi):
        o = obs_fn(x, command, xi)
        full = jnp.zeros(o.shape[:-1] + (obs_dim,), dtype=o.dtype)
        return full.at[..., indices].set(o)

    return obs_fn, full_obs_fn


def wrap_angle(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def window_origin(x0, lay):
    """(xy [..., 2], yaw [...]) of the window-start state."""
    return lay.get(x0, "base_xy"), lay.get(x0, "base_yaw")[..., 0]


def to_window_frame(x, origin_xy, origin_yaw, lay):
    """x with base_xy / base_yaw expressed in the frame at origin (rotation by -origin_yaw about z)."""
    s = lay.slices
    d = lay.get(x, "base_xy") - origin_xy
    c, sn = jnp.cos(origin_yaw)[..., None], jnp.sin(origin_yaw)[..., None]
    xy = jnp.concatenate([c * d[..., :1] + sn * d[..., 1:], -sn * d[..., :1] + c * d[..., 1:]], axis=-1)
    yaw = wrap_angle(lay.get(x, "base_yaw") - origin_yaw[..., None])
    return x.at[..., s["base_xy"]].set(xy).at[..., s["base_yaw"]].set(yaw)


def from_window_frame(x, origin_xy, origin_yaw, lay):
    s = lay.slices
    d = lay.get(x, "base_xy")
    c, sn = jnp.cos(origin_yaw)[..., None], jnp.sin(origin_yaw)[..., None]
    xy = jnp.concatenate([c * d[..., :1] - sn * d[..., 1:], sn * d[..., :1] + c * d[..., 1:]], axis=-1) + origin_xy
    yaw = wrap_angle(lay.get(x, "base_yaw") + origin_yaw[..., None])
    return x.at[..., s["base_xy"]].set(xy).at[..., s["base_yaw"]].set(yaw)


# ----------------------------------------------------------------------------------------------------------------
# Analytic margins (plan 4.5), consistent with the env termination
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarginConfig:
    height_percentage_threshold: float = 0.8
    height_curriculum_scaled: bool = True
    use_tilt: bool = False  # T06b: tilt termination off (threshold 1e6)
    tilt_threshold: float = 1e6
    use_qvel: bool = True  # the env's qvel-limit termination (|qvel[0:3]| clipped at 100)
    qvel_tolerance: float = 1e-3
    height_scale: float = 0.05  # m per margin unit (s_z)
    tilt_scale: float = 0.2  # rad per margin unit (s_theta)
    qvel_scale: float = 10.0  # m/s per margin unit

    def to_json(self):
        return dict(self.__dict__)


def margin_config_from_env(env, use_tilt=None, use_qvel=True):
    term = env.env_config["termination"]
    tilt = float(term.get("body_tilt_threshold", 1e6))
    return MarginConfig(
        height_percentage_threshold=float(term.get("height_percentage_threshold", 0.8)),
        height_curriculum_scaled=bool(term.get("height_curriculum_scaled", False)),
        use_tilt=(tilt < 1e5) if use_tilt is None else bool(use_tilt),
        tilt_threshold=tilt,
        use_qvel=use_qvel,
    )


def _rotation_matrix_xyz(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (scipy extrinsic 'xyz'), batched [..., 3, 3]."""
    cr, sr, cp, sp, cy, sy = jnp.cos(roll), jnp.sin(roll), jnp.cos(pitch), jnp.sin(pitch), jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([
        jnp.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        jnp.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        jnp.stack([-sp, cp * sr, cp * cr], -1),
    ], -2)


def margins(x, xi, cfg: MarginConfig, lay: Layout, xl: Layout):
    """Per-term margins in physical units (> 0 safe, < 0 exactly where that env termination fires)."""
    height = lay.get(x, "imu_height")[..., 0]
    coeff = xl.get(xi, "curriculum_coeff")[..., 0]
    scale = (1.0 - coeff) if cfg.height_curriculum_scaled else 1.0
    threshold = scale * cfg.height_percentage_threshold * xl.get(xi, "nominal_imu_height")[..., 0]
    out = {"height": height - threshold}
    rp = lay.get(x, "roll_pitch")
    out["tilt"] = cfg.tilt_threshold - jnp.sqrt(jnp.sum(rp ** 2, axis=-1))
    yaw = lay.get(x, "base_yaw")[..., 0]
    rotation = _rotation_matrix_xyz(rp[..., 0], rp[..., 1], yaw)
    v_world = jnp.einsum("...ij,...j->...i", rotation, lay.get(x, "base_lin_vel"))
    out["qvel"] = (QVEL_LIMIT - cfg.qvel_tolerance) - jnp.max(jnp.abs(v_world), axis=-1)
    return out


def margin_fall(x, xi, cfg: MarginConfig, lay: Layout, xl: Layout, softmin_temperature=0.0):
    """l_fall(x): min (or softmin with temperature T > 0) over the enabled terms, each divided by its scale."""
    m = margins(x, xi, cfg, lay, xl)
    terms = [m["height"] / cfg.height_scale]
    if cfg.use_tilt:
        terms.append(m["tilt"] / cfg.tilt_scale)
    if cfg.use_qvel:
        terms.append(m["qvel"] / cfg.qvel_scale)
    stacked = jnp.stack(terms, axis=-1)
    if softmin_temperature > 0.0:
        return -softmin_temperature * jax.nn.logsumexp(-stacked / softmin_temperature, axis=-1)
    return jnp.min(stacked, axis=-1)


# ----------------------------------------------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------------------------------------------
def schema_dict(env, margin_cfg=None, extra=None):
    lay, xl = state_layout(env), xi_layout(env)
    out = {
        "schema": SCHEMA_VERSION,
        "state": lay.to_json(),
        "action_dim": int(env.nr_actuator_joints),
        "xi": xl.to_json(),
        "delay_buffer_shape": [int(getattr(env.domain_randomization_action_delay_function, "max_nr_delay_steps", 0)) + 1,
                               int(env.nr_actuator_joints)],
        "policy_obs": {"dim": int(len(env.policy_observation_indices)), "blocks": policy_obs_blocks(env),
                       "scales": {"joint_pos": f"(q - nominal) / {OBS_JOINT_POS_SCALE}", "joint_vel": f"/ {OBS_JOINT_VEL_SCALE}",
                                  "last_action": f"/ {OBS_ACTION_SCALE}", "imu_ang_vel": f"clip(/ {OBS_ANG_VEL_SCALE}, -1, 1)",
                                  "command": "raw (vx, vy, yaw rate)", "projected_gravity": "raw", "all": f"clip +-{OBS_CLIP}"},
                       "full_obs_dim": int(env.single_observation_space.shape[0]),
                       "policy_observation_indices": [int(i) for i in np.asarray(env.policy_observation_indices)]},
        "dt": float(env.dt),
        "margin": (margin_cfg or margin_config_from_env(env)).to_json(),
    }
    if extra:
        out.update(extra)
    return out
