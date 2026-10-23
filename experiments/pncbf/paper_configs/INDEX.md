# Paper configs: figure → wandb runs (T03)

Source: wandb `catherineju-rwth-aachen-university/202601_paper` (59 runs, all `finished`, all have
`latest.model`), dumped 2026-10-22 with `dump_wandb.py` → `<group>/<run_id>.json` (config, summary,
CLI `args`, git commit, Slurm metadata, host/GPU, file list). `runs.tsv` is the one-line-per-run overview.
`paper_curves.csv` holds the numbers recovered from the paper figure PDFs (`extract_paper_curves.py`).

How the mapping was made: the plot scripts (`origin/plots:misc/rss/plot_line_plot_one_figure.py`,
`plot_perturbation_subplot.py`) do not query wandb; they read eval CSVs whose `file` column is
`.../experiments/runs/<run_root>/<setting>/<entry>/rollout/eval_*.pkl` and label curves by `run_root`
(`no_safety`→PPO-RL, `NCBF`→H=1 / PPO-RL-NCBF, `ours`, `H10`, `H50`, `ncbf`→state critic, SafeFall).
Checkpoints were pulled from wandb with `download_tagged_model_from_wandb.py` (default `--tag ours`) and
`download_model_from_wandb.py` (default regex `safe_fall_safety`). The run ids below come from those
tags/regexes, the run names, the configs, and the number of entries per curve that the figures' SE
bands imply (`extract_paper_curves.py` + a brute-force fit; the band is std(ddof=1)/2.236, sqrt(5) hard-coded in the plot script).

| figure / curve | run_root | wandb group | run ids (seed) | commit | confidence |
|---|---|---|---|---|---|
| Fig. 2 PPO-RL; App. H-full PPO-RL | `no_safety` | `ncbf-q-no-safe` | ym84duai (1042), g3vqg0f8 (1142), p37kpgcs (1242), 2k30xlgc (1342), wcofn9cv (1442) | 614c6b9 | high (`write_evaluation_to_csv.py` docstring example path `runs/no_safety/action_noise_hfield/no_safe_seed_1042`). The curve pools 8 (action noise) / 7 (perturbation) pkls per model, i.e. a γ_c list was also looped for PPO |
| Fig. 2 ours; Fig. 4 "best (H=25)"; App. H=25, SafeFall-"ours", state-action critic | `ours` | `ncbf-q-safe` | 5wpajwko (42), zdqj50w0 (43), wr9r03p2 (44), lpukb0yz (145), 27cfbtby (245) | cacffe7 | high for 42/43/44 (tag `ours`); 145/245 inferred (5 entries fit all 22 points). ag5rwgy3 `q-safe-gamma_0.405_seed_42` has the identical config and is also tagged `ours` |
| Fig. 2 PPO-RL-NCBF; Fig. 4 H=1; App. H=1 | `NCBF` | `ncbf-q-safe` | ksaqa0ay (42), e1oolmdg (43), 5xsqhdr5 (44) [tag `H1`], z84dphij (145), f3s2gzet (245) | cacffe7 | medium: the curve is NOT 5 entries × 10 episodes. Means are multiples of 1/40 and the bands fit 4 entries × 10 or 20 episodes (or 8 × 10) on 20–21 of 22 points. Most likely 4 of these 5 runs (3 tagged `H1` + one of 145/245), possibly with 2 pkls each. Other H=1 candidates: 3lyiu904 (`ncbf-q-safety` γ_c=5, tag `H1`), kn8h2mmb (`no_safety_H_sweep`), `202601_RSS_paper/ncbf-q-H_1_seed_2025` |
| Fig. 4 / App. H=10 | `H10` | `ncbf-q-safe` | m9j107v4 (42), 7596c1fl (43), xxn6cuqz (44), rvc4q1sa (145), ixnfwras (245) | cacffe7 | high (5 entries fit) |
| Fig. 4 / App. H=50 | `H50` | `ncbf-q-safe` | pa7gv3f0 (42), cazxxpqv (43), fzhmn9ai (44), m37s9i3z (145), ptkx5nwh (245) | cacffe7 | high (5 entries fit) |
| App. SafeFall (exps_safe_fall, prediction table) | `safe_fall` | `ncbf-q-safe-fall` | cr77zxxi (12142), 1kgldart (22142), u8vt9x0t (22242), b0xzdofb (22342), 6x0xfgyy (24442) | 804c7a5 (`ncbf-q-safefall`) | high |
| App. B state critic, version (i) (exps_state) | `ncbf` (`ncbf-state` on disk) | `ncbf` | flclqegy (42), 6r5bglx3 (142), hfg5gt43 (242) @78d9f1c; b76fjubv (343) @0d21d1c; tlufhugo (442) @312e49b | mixed! | high for the set; the 5 seeds ran on 3 commits (live tree changed during the array job) |

Not used in any figure: `no_safety_H_sweep` (5, branch `ncbf`, no shield, γ_c 0.5), `ncbf-q` (6, `testq_no_safe_*` = shield on but w_cbf 0, plus `no_safe_seed_42`),
`ncbf-q-safety` (8, γ_c 5 or w_cbf 0.1 variants), `q-safe-gamma_{0,1.386,4.595}_seed_42`, `debug_original_gamma_0_seed_42`.

## Config consistency (diff across seeds)
Within each figure set every config key is identical except `environment.seed`, `runner.run_name` and,
for seeds 43/44, `runner.project_name` (`202601_RSS_paper`, a logging-only field; the runs live in
`202601_paper`). Ours vs H=1/10/50 differ only in `algorithm.ncbf.H`. PPO-RL differs from ours only in
`use_safety_layer=False`, `gamma_c=0` (the critic is still trained with H=25, w_cbf=1, η=0.5).

## F4: values actually used (Fig. 2 / Fig. 4 runs)
| quantity | paper text / Table I | code default (`flax_full_jit/default_config.py`) | **actual (wandb)** |
|---|---|---|---|
| H | text 25, Table I 5 | 10 | **25** (ours), 1 (NCBF), 10/50 (Fig. 4); SafeFall 150 + `safe_fall_H=5`; state critic 25 |
| γ_c (logit scale, `logits = h − γ_c`) | 0.5 | 1.0 | **0.405** = logit(0.6) (ours, H sweep); PPO 0; state critic 0; SafeFall 0 |
| η_CBF | 0.5 | 1.0 | **0.5** |
| w_cbf | 0.5 | 1.0 | **1** |
| w_clf (label) | 1 | 1 | 1 |
| γ_label (`coef_decay_lambda`) | 0.95 | 0.95 | **0.95** (SafeFall 1.0) |
| w_lip / w_wd | 1e-4 / 1e-5 | 0 / 0 | **0 / 0** |
| ensemble M, CVaR α | 5, 0.4 | 5, α=0.4 hard-coded (k = ceil(0.6·5) = 3) | 5, 0.4 |
| buffer pos/neg (rollouts), neg sampling ratio | ρ⁺ = 0.2 "from safe buffer" | – / 0.5 | **50 / 20, 0.2 = ratio from the NEGATIVE buffer** |
| λ_slack, anticipation coef | 1000, 0.01 | 1000, 0.01 | 1000, 0.01 |
| shield action clipping | – | – | `action_clipping=False` (state critic: True) |
| ncbf lr, minibatches | – | – | 1e-3, 64 |
| critic input | – | – | `ncbf_use_policy_observations=True` (policy obs + action) |
| PPO | – | – | 1024 envs × 128 steps, 1e9 steps, lr 4e-4 annealed, clip 0.1, 10 epochs, minibatch 32768, γ 0.99, λ_GAE 0.9 |
| env | – | – | MJX Go2, **trained on flat `plane` only**, command clip_max_velocity 2.0, curriculum success return 8, full DR defaults |

## F10: evaluation
See `../reports/T03_T05.md`. Short version: CPU MuJoCo (`ncbf_mujoco.robot_locomotion.mujoco`, 1 env) via
`ncbf_ppo.flax` `test()`, 10 episodes per pkl; x values action noise {0,.2,.4,.6,.8,1},
perturbation {.002,.005,.01,.015,.02}; pkls/CSVs are not on IAS (they were on the laptop under
`/home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/` and `.../rl-x-NCBF/experiments/runs/ncbf-state/`).
