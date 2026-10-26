"""T13 speed check: world-model imagination (N envs x H steps + one VJP w.r.t. u0) vs one PPO env step of T06b.

With a checkpoint (on a GPU node), the T06b env/policy is built by the normal runner (`--runner.mode=test`) and this
script replaces PPO.test by the benchmark. Example (T06b ii_h25 seed 42, 1024 envs as in T06b):

  cd <tree>/experiments && python pncbf/wm/bench_wm_speed.py --bench_out OUT.json --bench_wm_envs 1024,4096 \
    --algorithm.name=ncbf_ppo.flax_full_jit --environment.name=ncbf_mujoco.robot_locomotion.mjx --runner.mode=test \
    --runner.load_model=<.../models/latest.model> --runner.track_console=False --runner.track_tb=False \
    --runner.track_wandb=False --runner.save_model=False --environment.nr_envs=1024 --environment.terrain.type=plane \
    <T06b env args>

Without a checkpoint (`--bench_no_env 1`, any device) only the world-model part runs, with a random policy MLP of the
T06b size (512, 256, 128). Times are medians of `--bench_reps` jitted calls after a warm-up; env steps are timed as a
scan of `--bench_scan` steps (as in the PPO rollout) and divided.

The state layout is the real schema (`--bench_schema`) or the stub Go2 "option 1" layout below; policy_obs_from_state
and margin_fall are the wm_state ones when importable, else stubs with the same cost profile (scatter of state groups
into the full observation vector; height margin).
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp


def _pop_bench_args(argv):
    out, rest = {}, []
    it = iter(argv)
    for a in it:
        if a.startswith("--bench_"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
            else:
                k, v = a[2:], next(it)
            out[k] = v
        else:
            rest.append(a)
    return out, rest


BENCH, rest = _pop_bench_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + rest
B = lambda k, d: type(d)(BENCH.get(f"bench_{k}", d)) if not isinstance(d, str) else BENCH.get(f"bench_{k}", d)

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W  # noqa: E402

STUB_GO2_SCHEMA = {   # the wm-data v1 state layout; xi_dim = number of varying xi dims (unknown before training)
    "state_dim": 51, "action_dim": 12, "xi_dim": int(BENCH.get("bench_xi_dim", 128)), "stub": True,
    "groups": {"joint_pos": [0, 12], "joint_vel": [12, 24], "imu_ang_vel": [24, 27], "projected_gravity": [27, 30],
               "last_action": [30, 42], "base_lin_vel": [42, 45], "imu_height": [45, 46], "roll_pitch": [46, 48],
               "base_xy": [48, 50], "base_yaw": [50, 51]},
}


def timeit(fn, *args, reps=20):
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def stub_obs_fn(layout, obs_dim, obs_groups):
    """obs_groups: [(state_group, obs_index_array)], command_idx: obs indices of the command."""
    gi = layout.group_index()

    def obs_fn(x, command):
        o = jnp.zeros(x.shape[:-1] + (obs_dim,), x.dtype)
        for g, idx in obs_groups:
            if g == "command":
                o = o.at[..., idx].set(command)
            else:
                o = o.at[..., idx].set(x[..., gi[g]])
        return o
    return obs_fn


def stub_margin(layout, z_min=0.2, scale=0.05):
    i = layout.group_index().get("imu_height", np.array([0]))[0]
    return lambda x: (x[..., i] - z_min) / scale


def safety_value(margins, T_l=0.1):
    """V = logit of P(no fall within H), plan sec. 4.5 (stable form). margins (H, N)."""
    logp = jnp.sum(-jax.nn.softplus(-margins / T_l), axis=0)
    log1mp = jnp.log(-jnp.expm1(jnp.minimum(logp, -1e-12)))
    return logp - log1mp


def wm_bench(wm, bundle, policy_fn, obs_fn, margin_fn, N, H, key, reps, open_loop=None, dtype=None):
    """Median ms of the jitted H-step rollout and of rollout + value_and_grad w.r.t. u0 (N envs)."""
    L = wm.layout
    open_loop = bool(int(B("open_loop", 0))) if open_loop is None else open_loop
    dtype = jnp.dtype(B("dtype", "float32")) if dtype is None else jnp.dtype(dtype)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x0 = (jax.random.normal(k1, (N, L.state_dim)) * 0.1).astype(dtype)
    xi = jax.random.normal(k2, (N, L.xi_dim)).astype(dtype)
    u0 = (jax.random.normal(k3, (N, L.action_dim)) * 0.3).astype(dtype)
    command = jax.random.normal(k4, (N, 3)).astype(dtype)
    bundle = jax.tree_util.tree_map(lambda a: a.astype(dtype), bundle)
    pf = lambda x: policy_fn(obs_fn(x, command)).astype(dtype)

    @jax.jit
    def fwd(x0, xi, u0):
        traj = W.rollout(wm, bundle, x0, xi, None, u0, pf, H, open_loop=open_loop)
        return safety_value(margin_fn(traj))

    @jax.jit
    def fwd_vjp(x0, xi, u0):
        def v(u):
            traj = W.rollout(wm, bundle, x0, xi, None, u, pf, H, open_loop=open_loop)
            return jnp.sum(safety_value(margin_fn(traj)))
        return jax.value_and_grad(v)(u0)

    return {"rollout_ms": timeit(fwd, x0, xi, u0, reps=reps), "rollout_vjp_ms": timeit(fwd_vjp, x0, xi, u0, reps=reps)}


def random_policy(obs_dim, action_dim, hidden=(512, 256, 128), seed=0):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.policy import Policy
    pol = Policy((action_dim,), 1.0, jnp.arange(obs_dim), hidden)
    params = pol.init(jax.random.PRNGKey(seed), jnp.zeros((1, obs_dim)))
    params = jax.tree_util.tree_map(lambda a: a.astype(jnp.dtype(B("dtype", "float32"))), params)
    return lambda o: pol.apply(params, o)[0]


def common_setup():
    schema = json.load(open(BENCH["bench_schema"])) if "bench_schema" in BENCH else STUB_GO2_SCHEMA
    layout = W.WMLayout.from_schema(schema)
    cfg = W.WMConfig(width=B("width", 512), depth=B("depth", 3), ensemble_size=B("ensemble", 1))
    wm = W.WorldModel(layout, cfg)
    bundle = wm.init(jax.random.PRNGKey(0))
    if "bench_wm_file" in BENCH:
        from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE
        wm, bundle, meta = WE.load_bundle(BENCH["bench_wm_file"])
        schema, layout = meta["schema"], wm.layout
    return schema, layout, wm, bundle


def results_out(res):
    res.update({"device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind, "bench_args": BENCH,
                "commit": os.popen(f"git -C {os.path.dirname(os.path.abspath(__file__))} rev-parse HEAD").read().strip()})
    print(json.dumps(res, indent=1))
    if "bench_out" in BENCH:
        with open(BENCH["bench_out"], "w") as f:
            json.dump(res, f, indent=1)


def run_no_env():
    schema, layout, wm, bundle = common_setup()
    obs_groups = []
    off = 0
    for g in ("joint_pos", "joint_vel", "last_action", "imu_ang_vel", "command", "projected_gravity"):
        n = 3 if g == "command" else len(layout.group_index()[g])
        obs_groups.append((g, np.arange(off, off + n)))
        off += n
    obs_fn = stub_obs_fn(layout, off, obs_groups)
    policy_fn = random_policy(off, layout.action_dim)
    res = {"wm": {}, "schema_stub": schema.get("stub", False), "wm_width": wm.cfg.width, "wm_depth": wm.cfg.depth,
           "open_loop": int(B("open_loop", 0)), "dtype": B("dtype", "float32"), "ensemble": wm.cfg.ensemble_size}
    for N in [int(n) for n in B("wm_envs", "1024,4096").split(",")]:
        for H in [int(h) for h in B("H", "25").split(",")]:
            res["wm"][f"N{N}_H{H}"] = wm_bench(wm, bundle, policy_fn, obs_fn, stub_margin(layout), N, H,
                                              jax.random.PRNGKey(1), B("reps", 20))
            print(f"N={N} H={H}: {res['wm'][f'N{N}_H{H}']}", flush=True)
    results_out(res)


def run_with_env(model, _episodes):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf_targets import episode_first_step  # noqa: F401
    env = model.env
    N_env = env.nr_envs
    reps, n_scan = B("reps", 10), B("scan", 32)
    key = jax.random.PRNGKey(0)
    env_state = env.reset(jax.random.split(key, N_env), True)

    def policy_action(env_state):
        latent_z = model.encoder_state.apply_fn(model.encoder_state.params, env_state.history_stack)
        dec = model._get_policy_decoder_output(model.decoder_state.params, latent_z, env_state.last_state,
                                               env_state.last_action, env_state.next_observation)
        obs_in = model._mask_policy_ball_plate_inputs(env_state.next_observation)
        mean, _ = model.policy.apply(model.policy_state.params, obs_in, latent_z, dec)
        return model.get_processed_action(mean)

    @jax.jit
    def plain_steps(env_state):
        def f(s, _):
            return env.step(s, policy_action(s)), None
        return jax.lax.scan(f, env_state, None, length=n_scan)[0]

    @jax.jit
    def shield_steps(env_state):
        def f(s, _):
            return env.step(s, model._policy_and_shield(s)[1]), None
        return jax.lax.scan(f, env_state, None, length=n_scan)[0]

    res = {"nr_envs_env": N_env, "scan": n_scan, "use_safety_layer": bool(model.ncbf_use_safety_layer)}
    res["env_step_policy_ms"] = timeit(plain_steps, env_state, reps=reps) / n_scan
    print(f"env step + policy ({N_env} envs): {res['env_step_policy_ms']:.3f} ms", flush=True)
    res["env_step_policy_shield_ms"] = timeit(shield_steps, env_state, reps=reps) / n_scan
    print(f"env step + policy + critic shield ({N_env} envs): {res['env_step_policy_shield_ms']:.3f} ms", flush=True)

    schema, layout, wm, bundle = common_setup()
    try:
        from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
        _, full_obs_fn = wm_state.make_policy_obs_fn(env)
        lay_s, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
        mcfg = wm_state.margin_config_from_env(env)
        xi_full = jnp.zeros((xl.dim,))   # the policy obs needs nominal_joint_pos from the full xi
        bx = lambda x: jnp.broadcast_to(xi_full, x.shape[:-1] + (xl.dim,))
        obs_fn = lambda x, command: full_obs_fn(x, command, bx(x))
        margin_fn = lambda x: wm_state.margin_fall(x, bx(x), mcfg, lay_s, xl)
        res["obs_fn"] = "wm_state"
    except Exception as e:  # stub until wm_state lands
        obs_groups = [("joint_pos", env.joint_positions_obs_idx), ("joint_vel", env.joint_velocities_obs_idx),
                      ("last_action", env.joint_previous_actions_obs_idx), ("imu_ang_vel", env.imu_angular_vel_obs_idx),
                      ("command", env.goal_velocities_obs_idx), ("projected_gravity", env.gravity_vector_obs_idx)]
        obs_fn = stub_obs_fn(layout, env.single_observation_space.shape[0], obs_groups)
        margin_fn = stub_margin(layout)
        res["obs_fn"] = f"stub ({type(e).__name__})"

    def policy_fn(obs):  # deterministic T06b policy on imagined observations (no history latent / decoder in T06b)
        mean, _ = model.policy.apply(model.policy_state.params, obs, None, None)
        return model.get_processed_action(mean)

    res["wm"] = {}
    res["wm_width"], res["wm_depth"], res["schema_stub"] = wm.cfg.width, wm.cfg.depth, schema.get("stub", False)
    for N in [int(n) for n in B("wm_envs", "1024,4096").split(",")]:
        for H in [int(h) for h in B("H", "25").split(",")]:
            r = wm_bench(wm, bundle, policy_fn, obs_fn, margin_fn, N, H, jax.random.PRNGKey(1), reps)
            res["wm"][f"N{N}_H{H}"] = r
            print(f"WM N={N} H={H}: {r}", flush=True)

    # D-style env step: policy + WM rollout/VJP for every env + env step, in the PPO scan
    H = int(B("H", "25").split(",")[0])
    gi = layout.group_index()

    @jax.jit
    def d_steps(env_state, key):
        def f(carry, _):
            s, k = carry
            k, k1 = jax.random.split(k)
            u = policy_action(s)
            x0 = jax.random.normal(k1, (N_env, layout.state_dim)) * 0.1 + jnp.mean(s.next_observation) * 0
            x0 = x0.at[:, gi["last_action"]].set(s.last_action[:, :layout.action_dim]) if "last_action" in gi else x0
            xi = jnp.zeros((N_env, layout.xi_dim))
            command = s.next_observation[:, env.goal_velocities_obs_idx]
            pf = lambda x: policy_fn(obs_fn(x, command))

            def v(uu):
                return jnp.sum(safety_value(margin_fn(W.rollout(wm, bundle, x0, xi, None, uu, pf, H))))
            _, g = jax.value_and_grad(v)(u)
            u = u + 0.0 * g   # the shield would project here
            return (env.step(s, u), k), None
        return jax.lax.scan(f, (env_state, key), None, length=n_scan)[0][0]

    res["env_step_policy_wm_vjp_ms"] = timeit(d_steps, env_state, jax.random.PRNGKey(2), reps=reps) / n_scan
    print(f"env step + policy + WM rollout/VJP ({N_env} envs, H={H}): {res['env_step_policy_wm_vjp_ms']:.3f} ms", flush=True)
    results_out(res)


if __name__ == "__main__":
    if int(B("no_env", 0)):
        run_no_env()
    else:
        from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
        from rl_x.runner.runner import Runner
        PPO.test = run_with_env
        Runner().run()
