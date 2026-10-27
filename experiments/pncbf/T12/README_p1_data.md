# P1 offline world-model data (T12)

Written by `rl_x/algorithms/ncbf_ppo/flax_full_jit/wm_collect.py` (branch `pncbf-v2-wm-data`), job script
`experiments/pncbf/T12/t12_collect.sbatch`. Field definitions, state / xi layouts, policy-obs and margin
configuration: `schema.json` (same directory). Report: `experiments/pncbf/reports/T11_T12.md`.

## Layout
```
v1/schema.json
v1/<split>/<checkpoint_label>/<setting>/{<field>.npy, meta.json, summary.json}
```
- split: `train` | `test` (held out by checkpoint seed: 145, 245; ppo 1342, 1442)
- checkpoint_label: `T06_fixed_<method>_seed_<seed>` (T06b, pncbf-v2 @ 62d1616, 986M steps)
- setting: `action_noise_p0`, `action_noise_p0p4`, `action_noise_p0p8`, `push_p0p005`, `push_p0p02`
- every field is a plain `.npy`: `np.load(path, mmap_mode="r")`. Per-step fields are time-major `[T=1000, N=2048, ...]`.

## Protocol (per shard)
Go2, flat terrain, T06b env config (paper height termination with curriculum scaling, tilt termination off),
eval mode (curriculum 1, DR at each reset and with p=0.002 per step), observation noise on, deterministic policy
(mode A, no shield), auto-reset, 20 s episodes. Action noise N(0, 1) with probability p per env and step; pushes =
the env's own perturbation (between x^- and x^+, flagged `perturbed_next`) with probability p and bounds x2
(background p=0.002 at x1 in the noise settings). Initial command 0 with probability 0.25.

MJX oracle: every 10 steps the first 256 envs are branched from the exact MJX state (true xi) for 50 steps:
deterministic policy on noise-free observations, no shocks, no DR / command resampling.

## Important
- At eval the T06b / paper termination threshold is (1 - 1) * 0.8 * h0 = 0: a termination means the IMU went
  below the ground (robot on its back / side), or the qvel limit. `margin_next` reproduces it exactly.
  `margin_physical_next` (IMU height < 0.5 nominal or tilt > 1 rad) marks physical falls that do not terminate.
- Mask the first transition of each episode (`episode_step == 0`): the env does not recompute kinematics at reset,
  so its IMU groups are stale.
- `xi` is stored once per distinct value: `xi_table[xi_idx[t, n]]`.
