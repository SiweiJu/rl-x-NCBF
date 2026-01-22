#!/usr/bin/env python3
"""
Generate deterministic action-noise trajectories (50 episodes, 1000 steps each)
using the exact JAX RNG logic provided, and save each episode to a separate file.

Action shape is (1, 12) per step (i.e., env batch dim = 1, action_dim = 12).

Default output directory:
  /home/siwei/Documents/repos/rl-x-NCBF/misc/evaluations/actions

Noise logic (per step):
    key, sampling_key, action_offset_key = jax.random.split(key, 3)
    if_sampling = jax.random.uniform(sampling_key, (1,)) < sampling_ratio
    noise = jax.random.normal(action_offset_key, (1, 12)) * 2.0 * if_sampling

This script saves the *additive noise term* that you would add to raw_processed_action.

Usage:
  python gen_action_noise_episodes.py --seed 0 --sampling_ratio 0.3
  python gen_action_noise_episodes.py --episodes 50 --steps 1000 --seed 2025 --sampling_ratio 0.1
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_OUT_DIR = "/home/siwei/Documents/repos/rl-x-NCBF/misc/evaluations/actions"


def generate_episode(
    key: jax.Array, T: int, action_shape: Tuple[int, int], sampling_ratio: float
) -> Tuple[Dict[str, Any], jax.Array]:
    """Generate one episode of (mask, noise) using the given RNG split pattern."""
    def step_fn(carry_key, _):
        carry_key, sampling_key, action_offset_key = jax.random.split(carry_key, 3)
        if_sampling = jax.random.uniform(sampling_key, (1,)) < sampling_ratio  # (1,) bool
        noise = (
            jax.random.normal(action_offset_key, action_shape)
            * 2.0
            * if_sampling.astype(jnp.float32)  # broadcasts (1,) -> (1,12)
        )
        return carry_key, (if_sampling[0], noise)

    key_out, (if_sampling_seq, noise_seq) = jax.lax.scan(step_fn, key, xs=None, length=T)
    ep = {
        "if_sampling": np.asarray(if_sampling_seq, dtype=bool),   # (T,)
        "noise": np.asarray(noise_seq, dtype=np.float32),         # (T, 1, 12)
        "T": int(T),
        "action_shape": tuple(int(x) for x in action_shape),
        "sampling_ratio": float(sampling_ratio),
    }
    return ep, key_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to write episode files.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--episodes", type=int, default=50, help="Number of episodes to generate.")
    ap.add_argument("--steps", type=int, default=1000, help="Steps per episode.")
    ap.add_argument("--sampling_ratio", type=float, default=0.2, help="Probability of adding noise each step.")
    ap.add_argument("--prefix", type=str, default="episode", help="Output filename prefix.")
    # Fixed per your note:
    ap.add_argument("--action_batch", type=int, default=1, help="Batch dimension (fixed to 1).")
    ap.add_argument("--action_dim", type=int, default=12, help="Action dimension (fixed to 12).")
    args = ap.parse_args()

    action_shape = (args.action_batch, args.action_dim)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base key -> deterministic sequence across episodes
    key = jax.random.PRNGKey(args.seed)

    manifest = {
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "action_shape": action_shape,
        "sampling_ratio": args.sampling_ratio,
        "files": [],
        "generator": "Per-step: split(key,3); if_sampling = uniform()<ratio; noise = normal()*2*if_sampling",
    }

    for ep_idx in range(args.episodes):
        ep_data, key = generate_episode(key, args.steps, action_shape, args.sampling_ratio)

        ep_path = out_dir / f"{args.prefix}_{ep_idx:03d}_ratio_{str(args.sampling_ratio).replace('.', 'p')}.pkl"
        payload = {
            "episode_index": ep_idx,
            **ep_data,
        }
        with ep_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        manifest["files"].append(str(ep_path))

    manifest_path = out_dir / f"manifest_ratio_{str(args.sampling_ratio).replace('.', 'p')}.pkl"
    with manifest_path.open("wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

    with (out_dir / f"files_ratio_{str(args.sampling_ratio).replace('.', 'p')}.txt").open("w") as f:
        for p in manifest["files"]:
            f.write(p + "\n")

    print(f"Wrote {args.episodes} episodes to: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Action shape per step: {action_shape} (saved noise has shape (T, 1, 12))")


if __name__ == "__main__":
    # Optional: deterministic CPU reductions; not required for RNG determinism.
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_deterministic_reductions")
    main()
