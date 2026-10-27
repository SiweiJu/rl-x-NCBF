"""T12: offline world-model data from a trained checkpoint (project plan 8 P1, v3.2).

Called from `PPO.test()` when `algorithm.wm_collect.enabled=True` (runner.mode=test). One process = one checkpoint,
all settings in `wm_collect.settings` one after the other. Per setting it writes one shard directory

    <out_dir>/<split>/<checkpoint_label>/<setting>/   (*.npy, meta.json, summary.json)

Protocol (T04 eval protocol, mode A): env in eval mode (curriculum 1, DR at every reset and with the env's per-step
probability inside episodes), observation noise on, deterministic policy (action mean, no shield), action noise
N(0, noise_std^2) with probability p per env and step (before the delay), env-internal pushes (the env's perturbation
function, between x^- and x^+) with probability p and bounds x push_scale; the action-noise settings keep the env's
background push probability. Envs run `nr_steps` control steps with auto-reset (episode = env horizon).

Every `oracle_interval` steps, the first `oracle_nr_envs` envs are branched from the exact MJX state (true xi) and
rolled out `oracle_horizon` steps with the deterministic policy on noise-free observations, no shocks, no DR
resampling and no command resampling (MJX oracle, plan v3.2 T12). Its first action is the policy action on the
noise-free observation o(x^+_t).

Field definitions: `SHARD_FIELDS` (also written to schema.json by `write_schema`).
"""
import contextlib
import functools
import json
import logging
import os
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import (
    _git_commit, _patched_attributes, _scaled_perturbation_function, apply_initial_commands, make_agent, RL_X_PATH)


rlx_logger = logging.getLogger("rl_x")

LABEL_HORIZONS = (1, 10, 25, 50)

# name -> (shape with symbolic dims, dtype, description). T = steps, N = envs, S = state dim, A = action dim,
# X = xi dim, M = xi table rows, B = delay buffer length, O = policy obs dim, L = len(LABEL_HORIZONS),
# P = oracle decision points, K = oracle envs, H = oracle horizon, F = full observation dim, Z = history steps.
SHARD_FIELDS = {
    "x_plus": (["T", "N", "S"], "float32", "x^+_t: state the policy acted on at t (after the reset if t starts an episode)"),
    "x_minus_next": (["T", "N", "S"], "float32", "x^-_{t+1}: after physics + qvel clip, before shock / DR (recorded before any reset)"),
    "x_plus_next": (["T", "N", "S"], "float32", "x^+_{t+1}: after shock / DR, before any reset (= x_plus[t+1] unless done[t])"),
    "u_policy": (["T", "N", "A"], "float32", "deterministic policy action at t (on the noisy observation obs[t])"),
    "u_cmd": (["T", "N", "A"], "float32", "action sent to the env = u_policy + action noise (chosen, undelayed; = last_action of x at t+1)"),
    "u_applied": (["T", "N", "A"], "float32", "u~_t: delayed action the PD controller used = delay_step(delay_buffer[t], u_cmd[t], delay_steps[t])[0]"),
    "delay_buffer": (["T", "N", "B", "A"], "float32", "env action_history before step t (oldest first)"),
    "delay_steps": (["T", "N"], "int8", "realized delay of step t (control steps)"),
    "xi_idx": (["T", "N"], "int32", "row of xi_table active during the physics of step t"),
    "xi_next_idx": (["T", "N"], "int32", "row of xi_table after the DR step of t (before any reset); differs from xi_idx iff dr_resampled_next"),
    "command": (["T", "N", "3"], "float32", "velocity command in the observation at t (vx, vy, yaw rate)"),
    "obs": (["T", "N", "O"], "float32", "policy observation at t as the policy saw it (with observation noise)"),
    "perturbed_next": (["T", "N"], "bool", "shock (push) between x^-_{t+1} and x^+_{t+1}"),
    "dr_resampled_next": (["T", "N"], "bool", "DR resampled between x^-_{t+1} and x^+_{t+1}"),
    "terminated": (["T", "N"], "bool", "transition t ended in a termination (fall / qvel limit)"),
    "truncated": (["T", "N"], "bool", "transition t hit the episode time limit (and did not terminate)"),
    "done": (["T", "N"], "bool", "terminated | truncated (the env reset after t)"),
    "noise_on": (["T", "N"], "bool", "action noise was added at t"),
    "episode_id": (["T", "N"], "int32", "episode counter per env (0 = first episode of the shard)"),
    "episode_step": (["T", "N"], "int32", "step index within the episode (0 = first state after a reset; mask this transition, see wm_state doc)"),
    "margin_next": (["T", "N"], "float32", "wm_state.margin_fall(x_plus_next, xi_table[xi_next_idx]); < 0 iff terminated"),
    "margin_height_next": (["T", "N"], "float32", "height term of the termination margin (m); < 0 iff the env's height termination fired"),
    "margin_qvel_next": (["T", "N"], "float32", "qvel-limit term (m/s); < 0 iff the qvel-limit termination fired"),
    "margin_physical_next": (["T", "N"], "float32", "wm_state.margin_fall_physical(x_plus_next, xi_next): < 0 = physically fallen (height < frac * nominal or tilt > thr), not a termination"),
    "fail_within": (["T", "N", "L"], "bool", "actual: a termination in transitions t..t+H-1 of the same episode, H in label_horizons"),
    "fail_within_mask": (["T", "N", "L"], "bool", "label known (failure seen, or H transitions observed before truncation / window end)"),
    "xi_table": (["M", "X"], "float32", "distinct xi vectors of the shard"),
    "oracle_t": (["P"], "int32", "decision steps t (index into T)"),
    "oracle_env": (["K"], "int32", "branched env indices"),
    "oracle_u0": (["P", "K", "A"], "float32", "first oracle action = policy on the noise-free o(x_plus[t])"),
    "oracle_u": (["P", "K", "H", "A"], "float32", "oracle actions (index 0 = oracle_u0)"),
    "oracle_x_plus_next": (["P", "K", "H", "S"], "float32", "oracle states after each branch step (no shocks, so x^+ == x^-)"),
    "oracle_margin": (["P", "K", "H"], "float32", "margin_fall of oracle_x_plus_next"),
    "oracle_margin_physical": (["P", "K", "H"], "float32", "margin_fall_physical of oracle_x_plus_next"),
    "oracle_terminated": (["P", "K", "H"], "bool", "oracle step k terminated (only the first one matters)"),
    "oracle_truncated": (["P", "K", "H"], "bool", "oracle step k truncated"),
    "oracle_first_failure": (["P", "K"], "int32", "first terminated step k (0 = the first transition); H if none"),
    "oracle_fail_within": (["P", "K", "L"], "bool", "oracle: first_failure < H"),
    "oracle_fail_within_mask": (["P", "K", "L"], "bool", "oracle label known (not truncated before H)"),
    "oracle_critic_p_safe": (["P", "K"], "float32", "this checkpoint's safety critic: ensemble-mean P(safe) for (obs[t], u_policy[t]) (NaN without critic)"),
    "oracle_critic_p_safe_std": (["P", "K"], "float32", "ensemble std of the above"),
    "oracle_critic_p_safe_cmd": (["P", "K"], "float32", "critic P(safe) for (obs[t], u_cmd[t])"),
    "oracle_obs_full": (["P", "K", "F"], "float32", "full env observation at t (noisy; critic input together with history)"),
    "oracle_history_stack": (["P", "K", "Z", "F"], "float32", "env history stack at t (input of the history encoder -> latent z of any critic)"),
}


def read_collect_config(config):
    cfg = config.algorithm.get("wm_collect", None)
    get = (lambda name, default: cfg.get(name, default)) if cfg is not None else (lambda name, default: default)
    settings = []
    for token in str(get("settings", "action_noise:0,action_noise:0.4,action_noise:0.8,push:0.005,push:0.02")).split(","):
        token = token.strip()
        if not token:
            continue
        kind, p = token.split(":")
        if kind not in ("action_noise", "push"):
            raise ValueError(f"wm_collect.settings: unknown perturbation {kind}")
        settings.append((kind, float(p)))
    return {
        "out_dir": get("out_dir", ""),
        "checkpoint_label": get("checkpoint_label", ""),
        "checkpoint_source": get("checkpoint_source", ""),
        "method": get("method", ""),
        "train_seed": int(get("train_seed", -1)),
        "split": get("split", "train"),
        "settings": settings,
        "nr_steps": int(get("nr_steps", 1000)),
        "eval_seed": int(get("eval_seed", 7)),
        "noise_std": float(get("noise_std", 1.0)),
        "push_scale": float(get("push_scale", 2.0)),
        "push_scale_joints": bool(get("push_scale_joints", True)),
        "background_push_prob": float(get("background_push_prob", -1.0)),
        "initial_command_zero_prob": float(get("initial_command_zero_prob", 0.25)),
        "oracle_interval": int(get("oracle_interval", 10)),
        "oracle_nr_envs": int(get("oracle_nr_envs", 256)),
        "oracle_horizon": int(get("oracle_horizon", 50)),
        "oracle_obs_noise": bool(get("oracle_obs_noise", False)),
        "oracle_store_states": bool(get("oracle_store_states", True)),
        "chunk_blocks": int(get("chunk_blocks", 5)),
        "skip_existing": bool(get("skip_existing", True)),
        "physical_height_fraction": float(get("physical_height_fraction", 0.5)),
        "physical_tilt_threshold": float(get("physical_tilt_threshold", 1.0)),
        "physical_min_steps": int(get("physical_min_steps", 25)),
    }


def setting_token(kind, p):
    return f"{kind}_p{p:g}".replace(".", "p")


def write_schema(env, path, extra=None):
    schema = wm_state.schema_dict(env, extra={
        "label_horizons": list(LABEL_HORIZONS),
        "layout": "one directory per shard (checkpoint x setting); every field is <name>.npy (np.load(mmap_mode='r'))",
        "time_major": True,
        "fields": {name: {"shape": shape, "dtype": dtype, "doc": doc} for name, (shape, dtype, doc) in SHARD_FIELDS.items()},
        **(extra or {}),
    })
    with open(path, "w") as f:
        json.dump(schema, f, indent=1)
    return schema


# ----------------------------------------------------------------------------------------------------------------
# Labels (host)
# ----------------------------------------------------------------------------------------------------------------
def actual_labels(terminated, truncated, horizons=LABEL_HORIZONS):
    """fail_within[t, n, h], mask[t, n, h] from the actual trajectory ([T, N] bool arrays)."""
    T, N = terminated.shape
    done = terminated | truncated
    inf = np.iinfo(np.int32).max // 2
    ttf = np.full(N, inf, dtype=np.int64)  # transitions until (and including) the termination
    rem = np.zeros(N, dtype=np.int64)  # transitions left in the episode within the window (incl. t)
    ttf_all = np.empty((T, N), dtype=np.int64)
    rem_all = np.empty((T, N), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        ttf = np.where(terminated[t], 1, np.where(truncated[t], inf, ttf + 1))
        rem = np.where(done[t], 1, rem + 1)
        ttf_all[t], rem_all[t] = ttf, rem
    hs = np.asarray(horizons)
    fail = ttf_all[..., None] <= hs
    mask = fail | (rem_all[..., None] >= hs)
    return fail, mask


def oracle_labels(terminated, truncated, horizons=LABEL_HORIZONS):
    """first_failure [..], fail_within [.., L], mask [.., L] from branch flags [..., H]."""
    H = terminated.shape[-1]
    done = terminated | truncated
    alive_before = np.concatenate([np.ones(done.shape[:-1] + (1,), bool), ~np.cumsum(done, -1).astype(bool)[..., :-1]], -1)
    term_alive = terminated & alive_before
    first_failure = np.where(term_alive.any(-1), np.argmax(term_alive, -1), H)
    trunc_alive = truncated & alive_before & ~terminated
    observed = np.where(trunc_alive.any(-1), np.argmax(trunc_alive, -1) + 1, H)  # transitions observed
    hs = np.asarray(horizons)
    fail = first_failure[..., None] < hs
    mask = fail | (observed[..., None] >= hs)
    return first_failure.astype(np.int32), fail, mask


class XiTable:
    """Host-side dedup of per-step xi vectors into a table + per-step row indices."""

    def __init__(self, nr_envs, dim):
        self.rows = []
        self.size = 0
        self.last = np.full((nr_envs, dim), np.nan, dtype=np.float32)
        self.last_idx = np.full(nr_envs, -1, dtype=np.int32)

    def index(self, xi):
        changed = np.any(xi != self.last, axis=1)
        k = int(changed.sum())
        if k:
            self.rows.append(xi[changed].copy())
            self.last[changed] = xi[changed]
            self.last_idx[changed] = np.arange(self.size, self.size + k, dtype=np.int32)
            self.size += k
        return self.last_idx.copy()

    def table(self):
        return np.concatenate(self.rows, 0) if self.rows else np.zeros((0, self.last.shape[1]), np.float32)


# ----------------------------------------------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _oracle_env_patches(env, obs_noise):
    """While tracing the oracle branch: no shocks, no DR / command resampling, (no observation noise), and the
    un-jitted _reset (so the branch never touches the jit cache of the main rollout)."""
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.none import NoneDRObservationNoise
    patches = [
        _patched_attributes(env.domain_randomization_perturbation_sampling_function, probability=0.0),
        _patched_attributes(env.domain_randomization_sampling_function, probability=0.0),
        _patched_attributes(env.command_sampling_function, probability=0.0),
    ]
    if not obs_noise:
        patches.append(_patched_attributes(env, observation_noise_function=NoneDRObservationNoise(env)))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        had_instance_reset = "_reset" in env.__dict__
        env._reset = functools.partial(type(env)._reset.__wrapped__, env)
        try:
            yield
        finally:
            if not had_instance_reset:
                del env._reset


def make_chunk_runner(model, env, cfg, nr_blocks):
    agent = make_agent(model)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    margin_cfg = wm_state.margin_config_from_env(env)
    phys_cfg = wm_state.PhysicalFallConfig(height_fraction=cfg["physical_height_fraction"], tilt_threshold=cfg["physical_tilt_threshold"])
    obs_fn, _ = wm_state.make_policy_obs_fn(env)
    pidx = jnp.asarray(env.policy_observation_indices)
    interval, K, H = cfg["oracle_interval"], cfg["oracle_nr_envs"], cfg["oracle_horizon"]
    raw_step = type(env)._step.__wrapped__
    has_critic = hasattr(model, "ncbf_state") and hasattr(model, "_safety_prediction")

    ext_x = jax.vmap(lambda d, i: wm_state.extract_state(env, d, i, i["last_action"]))
    ext_xi = jax.vmap(lambda m, i: wm_state.extract_xi(env, m, i))

    def margin(x, xi):
        return wm_state.margin_fall(x, xi, margin_cfg, lay, xl)

    def margin_phys(x, xi):
        return wm_state.margin_fall_physical(x, xi, phys_cfg, lay, xl)

    def critic(env_state, action):
        if not has_critic:
            nan = jnp.full(action.shape[0], jnp.nan)
            return nan, nan
        latent_z = model.encoder_state.apply_fn(model.encoder_state.params, env_state.history_stack)
        _, p, p_std = model._safety_prediction(env_state.next_observation, action, latent_z)
        return p, p_std

    def oracle(env_state, noise):
        sub = jax.tree_util.tree_map(lambda a: a[:K], env_state)
        u_pol, _ = agent.policy(sub)
        p_safe, p_std = critic(sub, u_pol)
        p_cmd, _ = critic(sub, u_pol + noise[:K])
        x0, xi0 = ext_x(sub.data, sub.internal_state), ext_xi(sub.mjx_model, sub.internal_state)
        clean = sub.replace(next_observation=sub.next_observation.at[:, pidx].set(obs_fn(x0, sub.internal_state["goal_velocities"], xi0)))
        u0, _ = agent.policy(clean)
        with _oracle_env_patches(env, cfg["oracle_obs_noise"]):
            step = jax.vmap(lambda s, a: raw_step(env, s, a))

            def branch_step(carry, k):
                state, u = carry
                state = step(state, u)
                x_next, xi_next = state.info["wm/x_plus"], state.info["wm/xi_next"]
                rec = {"u": u, "x": x_next, "margin": margin(x_next, xi_next), "margin_physical": margin_phys(x_next, xi_next),
                       "terminated": state.terminated, "truncated": state.truncated}
                u_new, _ = agent.policy(state)
                return (state, u_new), rec

            _, recs = jax.lax.scan(branch_step, (clean, u0), jnp.arange(H))
        out = {
            "oracle_u0": u0,
            "oracle_u": jnp.swapaxes(recs["u"], 0, 1),
            "oracle_margin": jnp.swapaxes(recs["margin"], 0, 1),
            "oracle_margin_physical": jnp.swapaxes(recs["margin_physical"], 0, 1),
            "oracle_terminated": jnp.swapaxes(recs["terminated"], 0, 1),
            "oracle_truncated": jnp.swapaxes(recs["truncated"], 0, 1),
            "oracle_critic_p_safe": p_safe, "oracle_critic_p_safe_std": p_std, "oracle_critic_p_safe_cmd": p_cmd,
            "oracle_obs_full": sub.next_observation, "oracle_history_stack": sub.history_stack,
        }
        if cfg["oracle_store_states"]:
            out["oracle_x_plus_next"] = jnp.swapaxes(recs["x"], 0, 1)
        return out

    def noise_at(noise_key, noise_prob, noise_std, t, nr_envs):
        k = jax.random.fold_in(noise_key, t)
        kb, kn = jax.random.split(k)
        on = jax.random.uniform(kb, (nr_envs,)) < noise_prob
        return jax.random.normal(kn, (nr_envs, noise_std.shape[0])) * noise_std * on[:, None], on

    def main_step(carry, t):
        env_state, noise_key, noise_prob, noise_std = carry
        nr_envs = env_state.next_observation.shape[0]
        internal = env_state.internal_state
        rec = {
            "x_plus": ext_x(env_state.data, internal),
            "xi": ext_xi(env_state.mjx_model, internal),
            "obs": env_state.next_observation[:, pidx],
            "command": internal["goal_velocities"],
            "delay_buffer": jax.vmap(lambda i: wm_state.delay_buffer(env, i))(internal),
            "episode_step": env_state.info_episode_store["episode_step"].astype(jnp.int32),
        }
        u_policy, _ = agent.policy(env_state)
        noise, on = noise_at(noise_key, noise_prob, noise_std, t, nr_envs)
        u_cmd = u_policy + noise
        env_state = env.step(env_state, u_cmd)
        info = env_state.info
        rec.update({
            "u_policy": u_policy, "u_cmd": u_cmd, "noise_on": on,
            "u_applied": info["wm/u_applied"], "delay_steps": info["wm/delay_steps"],
            "x_minus_next": info["wm/x_minus"], "x_plus_next": info["wm/x_plus"], "xi_next": info["wm/xi_next"],
            "perturbed_next": info["shock/perturbed"] > 0.5, "dr_resampled_next": info["shock/dr_resampled"] > 0.5,
            "terminated": env_state.terminated.astype(bool), "truncated": env_state.truncated.astype(bool) & ~env_state.terminated.astype(bool),
            "margin_next": margin(info["wm/x_plus"], info["wm/xi_next"]),
            "margin_physical_next": margin_phys(info["wm/x_plus"], info["wm/xi_next"]),
        })
        terms = wm_state.margins(info["wm/x_plus"], info["wm/xi_next"], margin_cfg, lay, xl)
        rec["margin_height_next"], rec["margin_qvel_next"] = terms["height"], terms["qvel"]
        return (env_state, noise_key, noise_prob, noise_std), rec

    def block(carry, block_index):
        env_state, noise_key, noise_prob, noise_std = carry
        t0 = block_index * interval
        noise0, _ = noise_at(noise_key, noise_prob, noise_std, t0, env_state.next_observation.shape[0])
        orc = oracle(env_state, noise0)
        carry, recs = jax.lax.scan(main_step, carry, t0 + jnp.arange(interval))
        return carry, (recs, orc)

    @jax.jit
    def run_chunk(env_state, noise_key, noise_prob, noise_std, first_block):
        carry = (env_state, noise_key, jnp.asarray(noise_prob, jnp.float32), jnp.asarray(noise_std, jnp.float32))
        carry, (recs, orcs) = jax.lax.scan(block, carry, first_block + jnp.arange(nr_blocks))
        recs = jax.tree_util.tree_map(lambda a: a.reshape((nr_blocks * interval,) + a.shape[2:]), recs)
        return carry[0], recs, orcs

    return run_chunk


def fall_episode_stats(fallen, terminated, done, min_steps):
    """Episodes with >= min_steps consecutive physically fallen steps, by episode outcome. fallen / terminated / done
    are [T, N] bool (transition t; fallen = margin_physical_next < 0)."""
    T, N = fallen.shape
    run = np.zeros(N, np.int64)
    fell = np.zeros(N, bool)
    first_long = np.full(N, -1, np.int64)
    ep_start = np.zeros(N, np.int64)
    out = {"episodes": 0, "episodes_terminated": 0, "fell_long_and_terminated": 0, "fell_long_not_terminated": 0,
           "fell_long_not_terminated_truncated": 0, "fell_long_not_terminated_window_end": 0,
           "steps_fallen_long_before_termination": [], "fallen_steps_not_terminated": 0}
    for t in range(T):
        run = np.where(fallen[t], run + 1, 0)
        newly = (run >= min_steps) & ~fell
        first_long = np.where(newly, t, first_long)
        fell |= run >= min_steps
        end = done[t] | (t == T - 1)
        for n in np.nonzero(end)[0]:
            out["episodes"] += 1
            if terminated[t, n]:
                out["episodes_terminated"] += 1
                if fell[n]:
                    out["fell_long_and_terminated"] += 1
                    out["steps_fallen_long_before_termination"].append(int(t - first_long[n] + min_steps))
            elif fell[n]:
                out["fell_long_not_terminated"] += 1
                out["fell_long_not_terminated_truncated" if done[t, n] else "fell_long_not_terminated_window_end"] += 1
        out["fallen_steps_not_terminated"] += int(np.sum(fallen[t] & ~terminated[t]))
        run = np.where(end, 0, run)
        fell = np.where(end, False, fell)
        first_long = np.where(end, -1, first_long)
    d = out.pop("steps_fallen_long_before_termination")
    out["median_fallen_steps_before_termination"] = float(np.median(d)) if d else None
    return out


def _summary(arrays, nr_envs, nr_steps, oracle_k, min_steps=25):
    term, trunc = arrays["terminated"], arrays["truncated"]
    fail, mask = arrays["fail_within"], arrays["fail_within_mask"]
    out = {
        "nr_transitions": int(term.size),
        "nr_terminations": int(term.sum()),
        "nr_truncations": int(trunc.sum()),
        "nr_episodes_started": int(arrays["episode_id"][-1].sum() + nr_envs),
        "shock_rate": float(arrays["perturbed_next"].mean()),
        "dr_resample_rate": float(arrays["dr_resampled_next"].mean()),
        "noise_rate": float(arrays["noise_on"].mean()),
        "delay_nonzero_rate": float((arrays["delay_steps"] > 0).mean()),
        "xi_table_rows": int(arrays["xi_table"].shape[0]),
        "margin_consistent": bool(np.array_equal(arrays["margin_next"] < 0, term)),
        "nr_margin_mismatch": int(np.sum((arrays["margin_next"] < 0) != term)),
        "termination_causes": {
            "imu_height_below_threshold": int(np.sum(term & (arrays["margin_height_next"] < 0))),
            "qvel_limit_only": int(np.sum(term & (arrays["margin_height_next"] >= 0) & (arrays["margin_qvel_next"] < 0))),
            "other": int(np.sum(term & (arrays["margin_height_next"] >= 0) & (arrays["margin_qvel_next"] >= 0))),
        },
        "physical_fall_steps": int(np.sum(arrays["margin_physical_next"] < 0)),
        "physical_fall": fall_episode_stats(arrays["margin_physical_next"] < 0, term, term | trunc, min_steps),
    }
    for i, h in enumerate(LABEL_HORIZONS):
        out[f"actual_fail_within_{h}_rate"] = float(fail[..., i][mask[..., i]].mean()) if mask[..., i].any() else None
        out[f"actual_fail_within_{h}_count"] = int(fail[..., i][mask[..., i]].sum())
        out[f"actual_label_known_{h}"] = float(mask[..., i].mean())
    # oracle vs actual on the same (t, env)
    ot, oe = arrays["oracle_t"], arrays["oracle_env"]
    a_fail = fail[ot][:, oe]
    a_mask = mask[ot][:, oe]
    for i, h in enumerate(LABEL_HORIZONS):
        o_fail = arrays["oracle_fail_within"][..., i]
        both = a_mask[..., i] & arrays["oracle_fail_within_mask"][..., i]
        a, o = a_fail[..., i][both], o_fail[both]
        out[f"oracle_vs_actual_{h}"] = {
            "n": int(both.sum()), "agree": float(np.mean(a == o)) if a.size else None,
            "actual_fail": int(a.sum()), "oracle_fail": int(o.sum()),
            "both_fail": int(np.sum(a & o)), "actual_only": int(np.sum(a & ~o)), "oracle_only": int(np.sum(~a & o)),
        }
    o_term = arrays["oracle_terminated"]
    o_alive = np.concatenate([np.ones(o_term.shape[:-1] + (1,), bool),
                              ~np.cumsum(arrays["oracle_terminated"] | arrays["oracle_truncated"], -1).astype(bool)[..., :-1]], -1)
    out["oracle_margin_consistent"] = bool(np.array_equal((arrays["oracle_margin"] < 0) & o_alive, o_term & o_alive))
    p = arrays["oracle_critic_p_safe"]
    out["critic_available"] = bool(np.isfinite(p).all())
    return out


def run_wm_collect(model):
    config, env = model.config, model.env
    cfg = read_collect_config(config)
    if not env.record_wm_state:
        raise ValueError("wm_collect needs --environment.record_wm_state=True")
    if not cfg["out_dir"]:
        cfg["out_dir"] = os.path.join(os.path.dirname(model.save_path), "wm_data")
    if not cfg["checkpoint_label"]:
        parts = os.path.abspath(config.runner.get("load_model", "") or "untrained").split(os.sep)
        cfg["checkpoint_label"] = parts[-3] if len(parts) >= 3 and parts[-2] == "models" else parts[-1]
    nr_envs = int(env.nr_envs)
    interval = cfg["oracle_interval"]
    if cfg["nr_steps"] % (interval * cfg["chunk_blocks"]) != 0:
        raise ValueError("wm_collect.nr_steps must be a multiple of oracle_interval * chunk_blocks")
    cfg["oracle_nr_envs"] = min(cfg["oracle_nr_envs"], nr_envs)
    nr_chunks = cfg["nr_steps"] // (interval * cfg["chunk_blocks"])
    commit = _git_commit()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    write_schema(env, os.path.join(cfg["out_dir"], "schema.json"))

    sampling_function = env.domain_randomization_perturbation_sampling_function
    perturbation_function = env.domain_randomization_perturbation_function
    env_push_prob = float(getattr(sampling_function, "probability", 0.0))
    background_push_prob = env_push_prob if cfg["background_push_prob"] < 0 else cfg["background_push_prob"]
    noise_std = jnp.full((env.nr_actuator_joints,), cfg["noise_std"], jnp.float32)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    results = []
    for setting_index, (kind, p) in enumerate(cfg["settings"]):
        token = setting_token(kind, p)
        shard = os.path.join(cfg["out_dir"], cfg["split"], cfg["checkpoint_label"], token)
        if cfg["skip_existing"] and os.path.exists(os.path.join(shard, "summary.json")):
            rlx_logger.info(f"[skip] {shard} exists")
            continue
        push_prob, push_scale = (p, cfg["push_scale"]) if kind == "push" else (background_push_prob, 1.0)
        noise_prob = p if kind == "action_noise" else 0.0
        scaled = _scaled_perturbation_function(perturbation_function, push_scale, cfg["push_scale_joints"])
        attrs = {n: getattr(scaled, n) for n in ("trunk_velocity_clip_mass_factor", "trunk_velocity_clip_limit", "max_joint_velocity", "max_joint_position")}
        seed = cfg["eval_seed"] * 1000 + setting_index
        base = jax.random.PRNGKey(seed)
        reset_key, noise_key, cmd_key = jax.random.fold_in(base, 0), jax.random.fold_in(base, 1), jax.random.fold_in(base, 3)
        start = time.time()
        with _patched_attributes(sampling_function, probability=push_prob), _patched_attributes(perturbation_function, **attrs):
            jax.clear_caches()  # env.step is jitted with the env as a static argument: retrace with this setting
            run_chunk = make_chunk_runner(model, env, cfg, cfg["chunk_blocks"])
            env_state = env.reset(jax.random.split(reset_key, nr_envs), True)
            env_state = apply_initial_commands(env, env_state, cmd_key, cfg["initial_command_zero_prob"])
            T = cfg["nr_steps"]
            host = {}
            xi_tab = XiTable(nr_envs, xl.dim)
            xi_idx = np.empty((T, nr_envs), np.int32)
            xi_next_idx = np.empty((T, nr_envs), np.int32)
            oracle_parts = []
            compile_s = None
            for c in range(nr_chunks):
                t_chunk = time.time()
                env_state, recs, orcs = run_chunk(env_state, noise_key, noise_prob, noise_std, c * cfg["chunk_blocks"])
                recs = {k: np.asarray(v) for k, v in jax.device_get(recs).items()}
                orcs = {k: np.asarray(v) for k, v in jax.device_get(orcs).items()}
                if compile_s is None:
                    compile_s = time.time() - t_chunk
                t0 = c * cfg["chunk_blocks"] * interval
                for i in range(recs["xi"].shape[0]):
                    xi_idx[t0 + i] = xi_tab.index(recs["xi"][i])
                    xi_next_idx[t0 + i] = xi_tab.index(recs["xi_next"][i])
                for k, v in recs.items():
                    if k in ("xi", "xi_next"):
                        continue
                    if k not in host:
                        dtype = np.int8 if k == "delay_steps" else v.dtype
                        host[k] = np.empty((T,) + v.shape[1:], dtype)
                    host[k][t0:t0 + v.shape[0]] = v
                oracle_parts.append(orcs)
            wall = time.time() - start
        arrays = host
        arrays["xi_idx"], arrays["xi_next_idx"], arrays["xi_table"] = xi_idx, xi_next_idx, xi_tab.table()
        arrays["done"] = arrays["terminated"] | arrays["truncated"]
        arrays["episode_id"] = np.concatenate([np.zeros((1, nr_envs), np.int32), np.cumsum(arrays["done"], 0)[:-1].astype(np.int32)], 0)
        arrays["fail_within"], arrays["fail_within_mask"] = actual_labels(arrays["terminated"], arrays["truncated"])
        for k in oracle_parts[0]:
            arrays[k] = np.concatenate([o[k] for o in oracle_parts], 0)
        arrays["oracle_t"] = (np.arange(arrays["oracle_u0"].shape[0]) * interval).astype(np.int32)
        arrays["oracle_env"] = np.arange(cfg["oracle_nr_envs"], dtype=np.int32)
        arrays["oracle_first_failure"], arrays["oracle_fail_within"], arrays["oracle_fail_within_mask"] = oracle_labels(
            arrays["oracle_terminated"], arrays["oracle_truncated"])
        summary = _summary(arrays, nr_envs, T, cfg["oracle_nr_envs"], cfg["physical_min_steps"])
        summary.update({"wall_time_s": wall, "first_chunk_s_incl_compile": compile_s})
        meta = {
            "schema": wm_state.SCHEMA_VERSION,
            "checkpoint": cfg["checkpoint_source"] or config.runner.get("load_model", None),
            "checkpoint_label": cfg["checkpoint_label"], "method": cfg["method"], "train_seed": cfg["train_seed"],
            "split": cfg["split"], "setting": token, "perturbation": kind, "p": p,
            "noise_std": cfg["noise_std"] if kind == "action_noise" else 0.0, "noise_prob": noise_prob,
            "push_prob": push_prob, "push_scale": push_scale, "push_scale_joints": cfg["push_scale_joints"],
            "env_push_prob_config": env_push_prob, "eval_seed": seed, "nr_envs": nr_envs, "nr_steps": T,
            "episode_horizon": int(env.horizon), "deployment_mode": "A", "deterministic_policy": True,
            "observation_noise": True, "eval_mode": True,
            "initial_command_zero_prob": cfg["initial_command_zero_prob"],
            "oracle": {"interval": interval, "nr_envs": cfg["oracle_nr_envs"], "horizon": cfg["oracle_horizon"],
                       "obs_noise": cfg["oracle_obs_noise"], "shocks": False, "dr_resampling": False,
                       "command_resampling": False, "first_action": "policy on noise-free o(x_plus[t])"},
            "terrain": config.environment.terrain.type,
            "termination": {k: config.environment.termination[k] for k in config.environment.termination},
            "margin": wm_state.margin_config_from_env(env).to_json(),
            "margin_physical": {**wm_state.PhysicalFallConfig(height_fraction=cfg["physical_height_fraction"], tilt_threshold=cfg["physical_tilt_threshold"]).to_json(),
                                "min_steps_for_episode_stats": cfg["physical_min_steps"]},
            "git_commit": commit, "rl_x_path": RL_X_PATH,
            "fields": sorted(arrays),
        }
        tmp = shard + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for k, v in arrays.items():
            if k not in SHARD_FIELDS:
                continue
            dtype = SHARD_FIELDS[k][1]
            np.save(os.path.join(tmp, f"{k}.npy"), np.asarray(v).astype(dtype if dtype != "bool" else bool))
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(tmp, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        if os.path.exists(shard):
            os.rename(shard, shard + f".old_{int(time.time())}")
        os.makedirs(os.path.dirname(shard), exist_ok=True)
        os.rename(tmp, shard)
        rlx_logger.info(f"[wm_collect {token}] {summary['nr_transitions']} transitions, {summary['nr_terminations']} terminations, "
                        f"shock rate {summary['shock_rate']:.4f}, margin consistent {summary['margin_consistent']}, "
                        f"oracle agree@25 {summary['oracle_vs_actual_25']['agree']}, {wall:.0f} s -> {shard}")
        results.append(summary)
    return results
