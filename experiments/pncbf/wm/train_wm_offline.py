"""Offline world-model training (P1 / T13). Requeue-safe: fixed run dir, atomic checkpoints, fixed wandb id.

  python experiments/pncbf/wm/train_wm_offline.py --run_dir RUN --data '/home/ju/repo/pncbf/p1_data/<set>/*.npz' \
      --schema /home/ju/repo/pncbf/p1_data/<set>/schema.json --heldout_pattern seed_245 --updates 200000
  python experiments/pncbf/wm/train_wm_offline.py --run_dir /tmp/wm_smoke --data synthetic --updates 300   # toy data

Loss: k_train-step autoregressive MSE of x_hat-_{t+j} vs x-_{t+j} (scaled by the per-dim std of the one-step delta),
windows from all valid starts plus a near-failure share (`--neg_ratio`); `--method A` = teacher forcing at shocks /
DR resampling, `--method B` = windows containing one are masked.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data", required=True, help="glob of .npz shards, or 'synthetic'")
    p.add_argument("--schema", default=None)
    p.add_argument("--field_map", default=None, help="JSON {canonical: shard_key}")
    p.add_argument("--heldout_pattern", default="seed_145/,seed_245/,seed_1342/,seed_1442/",
                   help="comma-separated substrings; shards whose path contains one are held out (default: the last two "
                        "seeds per method; use the split from schema.json when it states one)")
    p.add_argument("--max_train_rows", type=int, default=0, help="0 = all")
    # model
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--layer_norm", type=int, default=1)
    p.add_argument("--activation", default="elu")
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--use_xi", type=int, default=1)
    p.add_argument("--last_action_source", default="chosen")
    # training
    p.add_argument("--k_train", type=int, default=8)
    p.add_argument("--method", default="A", choices=["A", "B"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--neg_ratio", type=float, default=0.3)
    p.add_argument("--near_failure_window", type=int, default=50)
    p.add_argument("--updates", type=int, default=200000)
    p.add_argument("--updates_per_call", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--eval_ks", default="1,8,25,50")
    p.add_argument("--eval_windows", type=int, default=20000)
    p.add_argument("--save_minutes", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    # synthetic data
    p.add_argument("--synthetic_envs", type=int, default=256)
    p.add_argument("--synthetic_T", type=int, default=1000)
    # logging
    p.add_argument("--wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="pncbf-2026")
    p.add_argument("--wandb_entity", default="catherineju-rwth-aachen-university")
    p.add_argument("--wandb_tags", default="phase/P1,exp/T13")
    p.add_argument("--run_name", default=None)
    return p.parse_args(argv)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True).strip()
    except Exception:
        return "unknown"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(args):
    if args.data == "synthetic":
        train, schema = WD.synthetic_dataset(n_envs=args.synthetic_envs, T=args.synthetic_T, seed=args.seed)
        held, _ = WD.synthetic_dataset(n_envs=max(args.synthetic_envs // 4, 8), T=args.synthetic_T, seed=args.seed + 1000)
        return train, held, schema, {"train": ["synthetic"], "heldout": ["synthetic_heldout"]}
    schema = WD.load_schema(args.schema)
    field_map = json.loads(args.field_map) if args.field_map else schema.get("field_map")
    paths = WD.find_shards(args.data)
    if not paths:
        raise FileNotFoundError(args.data)
    pats = [q for q in (schema.get("heldout_patterns") or args.heldout_pattern.split(",")) if q]
    is_held = [any(q in p for q in pats) for p in paths]
    tr_paths = [p for p, h in zip(paths, is_held) if not h]
    he_paths = [p for p, h in zip(paths, is_held) if h]
    log(f"shards: {len(tr_paths)} train, {len(he_paths)} held out ({args.heldout_pattern})")
    train, tr_src = WD.load_shards(tr_paths, field_map)
    held, he_src = WD.load_shards(he_paths, field_map) if he_paths else (None, [])
    return train, held, schema, {"train": tr_src, "heldout": he_src}


def subsample_tracks(data, max_rows, seed):
    if not max_rows or len(data["done"]) <= max_rows:
        return data
    tracks = np.unique(data["track_id"])
    rng = np.random.default_rng(seed)
    rng.shuffle(tracks)
    lens = np.bincount(data["track_id"])[tracks]
    keep = np.sort(tracks[np.cumsum(lens) <= max_rows])
    m = np.isin(data["track_id"], keep)
    return {f: v[m] for f, v in data.items()}


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.run_dir, exist_ok=True)
    ckpt_path = os.path.join(args.run_dir, "ckpt_latest.msgpack")
    commit = git_commit()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.run_dir))
    log(f"run {run_name} dir={args.run_dir} commit={commit} devices={jax.devices()} "
        f"SLURM_RESTART_COUNT={os.environ.get('SLURM_RESTART_COUNT', 0)}")

    t0 = time.time()
    train, held, schema, sources = load_data(args)
    train = subsample_tracks(train, args.max_train_rows, args.seed)
    log(f"data loaded in {time.time() - t0:.0f}s: train rows {len(train['done'])}, held rows "
        f"{0 if held is None else len(held['done'])}")
    layout = W.WMLayout.from_schema(schema)
    cfg = W.WMConfig(width=args.width, depth=args.depth, layer_norm=bool(args.layer_norm), activation=args.activation,
                     ensemble_size=args.ensemble, use_xi=bool(args.use_xi), last_action_source=args.last_action_source)
    wm = W.WorldModel(layout, cfg)

    pos, neg = WD.split_window_starts(train, args.k_train, args.near_failure_window)
    log(f"windows k={args.k_train}: {len(pos)} valid, {len(neg)} near-failure; shock rows "
        f"{int(train['shock_next'].sum())}, terminations {int(train['terminated'].sum())}")
    rows = WD.transition_rows_for_norm(train, seed=args.seed)
    norm = W.fit_norm(layout, cfg, train["x_plus"][rows], train["u_applied"][rows], train["xi"][rows],
                      train["x_minus_next"][rows])
    dev = WD.to_device(train)
    held_dev = WD.to_device(held) if held is not None else None
    if held is not None:
        hpos, hneg = WD.split_window_starts(held, args.k_train, args.near_failure_window)

    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    bundle = wm.init(init_key, norm)
    opt_state = tx.init(bundle["params"])
    step = 0
    state = {"params": bundle["params"], "opt_state": opt_state, "norm": norm, "step": np.int64(0), "key": key}
    if os.path.exists(ckpt_path):
        state = WE.load_msgpack(ckpt_path, state)
        step = int(state["step"])
        log(f"resumed from {ckpt_path} at update {step}")
    else:
        log("no checkpoint, starting fresh")

    sampler = WD.make_window_sampler(pos, neg, args.k_train, args.batch, args.neg_ratio)

    @jax.jit
    def train_chunk(params, opt_state, norm, key, dev):
        def one(carry, k):
            params, opt_state = carry
            batch = sampler(k, dev)
            (loss, aux), grads = jax.value_and_grad(
                lambda p: W.k_step_loss(wm, p, norm, batch, args.method), has_aux=True)(params)
            gnorm = optax.global_norm(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), {**aux, "grad_norm": gnorm}
        keys = jax.random.split(key, args.updates_per_call)
        (params, opt_state), m = jax.lax.scan(one, (params, opt_state), keys)
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, m)

    if held is not None:
        hsampler = WD.make_window_sampler(hpos, hneg, args.k_train, min(args.batch, 4096), args.neg_ratio)

        @jax.jit
        def held_loss(params, norm, key, dev):
            return W.k_step_loss(wm, params, norm, hsampler(key, dev), args.method)[1]["loss"]

    wb = None
    if args.wandb:
        import wandb
        wid = hashlib.md5(os.path.abspath(args.run_dir).encode()).hexdigest()[:16]
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, id=wid, resume="allow",
                        tags=[t for t in args.wandb_tags.split(",") if t] + [f"method/wm{args.method}"],
                        config={**vars(args), "commit": commit, "schema": schema, "sources": sources,
                                "n_train_rows": int(len(train["done"])), "n_pos_windows": int(len(pos)),
                                "n_neg_windows": int(len(neg))}, dir=args.run_dir)

    def save():
        WE.save_msgpack(ckpt_path, {**state, "step": np.int64(step)})

    def do_eval(final=False):
        if held is None:
            return {}
        ks = [int(k) for k in args.eval_ks.split(",")]
        b = {"params": state["params"], "norm": state["norm"]}
        res = WE.evaluate(wm, b, held, ks, args.near_failure_window, args.eval_windows, seed=0, dev=held_dev)
        if final:
            with open(os.path.join(args.run_dir, "eval_heldout.json"), "w") as f:
                json.dump(res, f, indent=1)
        return WE.summary_scalars(res, "heldout")

    last_save = time.time()
    t_train, step0 = time.time(), step
    while step < args.updates:
        key, sub = jax.random.split(state["key"])
        params, opt_state, m = train_chunk(state["params"], state["opt_state"], state["norm"], sub, dev)
        state.update(params=params, opt_state=opt_state, key=key)
        step += args.updates_per_call
        crossed = lambda every: every and step // every != (step - args.updates_per_call) // every
        if crossed(args.log_every) or step >= args.updates:
            m = {k: float(v) for k, v in jax.device_get(m).items()}
            if held is not None:
                key, hk = jax.random.split(state["key"])
                state["key"] = key
                m["heldout_loss"] = float(held_loss(state["params"], state["norm"], hk, held_dev))
            m["updates_per_s"] = (step - step0) / max(time.time() - t_train, 1e-9)
            log(f"update {step}: " + " ".join(f"{k}={v:.4g}" for k, v in m.items()))
            if wb:
                wb.log({f"train/{k}": v for k, v in m.items()}, step=step)
        if crossed(args.eval_every) and step < args.updates:
            sc = do_eval()
            log("eval " + " ".join(f"{k}={v:.3g}" for k, v in sc.items() if k.endswith("rel_rmse")))
            if wb:
                wb.log(sc, step=step)
        if time.time() - last_save > 60 * args.save_minutes:
            save()
            last_save = time.time()
            log(f"checkpoint at update {step}")
    save()
    sc = do_eval(final=True)
    log("final eval " + " ".join(f"{k}={v:.3g}" for k, v in sc.items() if k.endswith("rel_rmse")))
    if wb:
        wb.log(sc, step=step)
    WE.save_bundle(os.path.join(args.run_dir, "wm_final.msgpack"), wm,
                   {"params": state["params"], "norm": state["norm"]}, schema,
                   extra={"args": vars(args), "commit": commit, "updates": step, "sources": sources})
    log(f"done: {step} updates, wm_final.msgpack written")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
