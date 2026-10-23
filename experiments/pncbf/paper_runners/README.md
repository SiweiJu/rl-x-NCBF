# Paper run scripts (copied from IAS, verbatim)

Copied 2026-10-22 by agent `recover` (T03), read-only, with `scp -p` (mtimes kept, md5 checked against the source).
They were never committed. Several were edited in place after the paper runs were launched, so the
`args` stored in wandb (`../paper_configs/<group>/<run_id>.json`, field `args`) are the authoritative
command lines. Each script still matches its wandb group by `exp_name`/`run_name`/seed list.

| copied file | source on IAS (`ju@mn`) | mtime | wandb group (202601_paper) / Slurm job | matches? |
|---|---|---|---|---|
| `rl-x-NCBF_runner_archiv/ncbf-q/ncbf-q-seeds.sh` | `/home/ju/repo/rl-x-NCBF/experiments/runner/archiv/ncbf-q/` | 2026-01-28 11:45 | `ncbf-q-no-safe` (**PPO-RL in Fig. 2**), job 1373786_0-4, seeds 1042–1442 | yes, args identical |
| `rl-x-NCBF_runner_archiv/ncbf-q/q_safe_sweep.sh` | same dir | 2026-01-29 18:26 | `ncbf-q-safe` `q-safe-H-gamma_0.405_seed_*_H_*` (**ours / H sweep**), jobs 1374015, 1374359, 1374629 | edited after launch: now γ_c sweep on `hfield_diverse`, project `202601_RSS_paper`, run name `q-safe-gamma-hfield-…` (no such runs in 202601_paper). The run-time version had `H_VALUES=(1 10 25 50)`, `GAMMA_C_VALUES=(0.405)`, run name `q-safe-H-gamma_${GAMMA_C}_seed_${SEED}_H_${H}`, terrain `plane` (see wandb args) |
| `rl-x-NCBF_runner_archiv/ncbf-q/ncbf-q-safe.sh` | same dir | 2026-01-28 22:40 | `ncbf-q-safety` `ncbf-q-safety_wcbf0d01_…` (w_cbf 0.1, job 1374396) | yes (env `rlx-ncbf-q`, project now `202601_RSS_paper`) |
| `rl-x-NCBF_runner_archiv/ncbf-q/ncbf-q-no-safe.sh` | same dir | 2026-01-28 11:34 | `ncbf-q` `no_safe_seed_42` style (w_cbf 0) | partly (copy of the rl-x-NCBF-q one) |
| `rl-x-NCBF_runner_archiv/ncbf-q/dual_buffer_sweep.sh` | same dir | 2026-01-28 11:34 | project `202601_ncbf` (not paper) | – |
| `rl-x-NCBF_runner_archiv/ncbf-q/ncbf-q-g1.sh` | same dir | 2026-03-02 16:39 | project `202603_ncbf_g1` (not paper) | – |
| `rl-x-NCBF_runner_archiv/ncbf/ncbf_long_run.sh` | `/home/ju/repo/rl-x-NCBF/experiments/runner/archiv/ncbf/` | 2026-01-25 20:21 | `no_safety_H_sweep` (job 1371776_0-4, H 1/5/10/25/50, γ_c 0.5, no shield) | yes |
| `rl-x-NCBF_runner_archiv/ncbf/dual_buffer_sweep.sh` | same dir | 2026-02-04 12:00 | `ncbf` = **version (i) state critic, App. B** (job 1378806_0-4, seeds 42 142 242 343 442) | yes; but the 5 tasks ran on 3 different commits (78d9f1c, 0d21d1c, 312e49b) because the live tree changed while they ran |
| `rl-x-NCBF_runner_archiv/ncbf/ncbf-safe-fall.sh` | same dir | 2026-02-04 16:23 | project `202601_ncbf` (SafeFall without shield; not in paper project) | – |
| `rl-x-NCBF_runner_archiv/ncbf/ncbf_no_slayer.sh`, `ncbf_single.sh` | same dir | 2026-01-10 / 01-13 | projects `202511_ncbf`, `202601_ncbf` (early tests) | – |
| `rl-x-NCBF-q_experiments_ncbf/ncbf-q-safe-fall.sh` | `/home/ju/repo/rl-x-NCBF-q/experiments/ncbf/` | 2026-02-06 08:29 | `ncbf-q-safe-fall` = **SafeFall (App.)**, job 1380059_0-4, commit 804c7a5 (`ncbf-q-safefall`) | yes |
| `rl-x-NCBF-q_experiments_ncbf/ncbf-q-safe.sh` | same dir | 2026-01-27 23:21 | `ncbf-q-safety` gc 5, H 1/10 (job 1373469); H 25/50 came from job 1373465 (script overwritten) | yes for 1373469 |
| `rl-x-NCBF-q_experiments_ncbf/ncbf-q-no-safe.sh` | same dir | 2026-01-27 14:02 | `ncbf-q` `no_safe_seed_42` (job 1372950) | yes |
| `rl-x-NCBF-q_experiments_ncbf/ncbf-q-seeds.sh` | same dir | 2026-01-30 12:49 | project `202601_RSS_paper` (`ncbf-q-hfield_seed_2025`, H=1, hfield; not in 202601_paper) | – |
| `rl-x-NCBF-q_experiments_ncbf/dual_buffer_sweep.sh`, `ncbf-g1.sh` | same dir | 2026-01-26 / 03-02 | projects `202601_ncbf`, `202603_ncbf_g1` | – |

Not copied (not paper runs): `/home/ju/repo/rl-x-NCBF/experiments/runner/ncbf-q-latent/` (2026-03-24 … 04-30,
after the submission) and `/home/ju/repo/runner_script/` (2024–2025, other projects).

Slurm logs that still exist: `/home/ju/repo/rl-x-NCBF-q/experiments/log/out_and_err_{1372895,1372950,1373465,1373469,1380059}_*.txt`.
The logs of the runs launched from `/home/ju/repo/rl-x-NCBF/experiments` (all Fig. 2 / Fig. 4 runs) are gone;
wandb keeps each run's `output.log`.

Important: the env `rlx-ncbf` has an editable install of `rl_x` pointing at `/home/ju/repo/rl-x-NCBF`, so these
scripts ran whatever that tree had checked out at (re)start time. The commits are in wandb (`commit` field).
