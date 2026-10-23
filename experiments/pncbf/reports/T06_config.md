# T06 config: paper runs (cacffe7 / ncbf-q) vs pncbf-v2 (agent core, 2026-10-23)

Reference run: wandb `202601_paper/zdqj50w0` (ours, H=25, seed 43). Its CLI args are the T05 args
(`experiments/pncbf/T05/paper_fig2.sbatch`). The other paper sets differ only in H (1/10/50) and, for PPO, in
`use_safety_layer=False gamma_c=0` (INDEX.md). v2 = `pncbf-v2` @ aec2975 (merged) + this branch.
Method: I diffed the flattened configs, then diffed the code paths
(`flax_full_jit/*`, mjx env, Go2 default reward/termination/command) for behavior changes that have no config key.

## Decision (main session, 2026-10-23)
Both. **T06a** (`ncbf-q-fixed`, paper code + fixes only) is the G0 comparison against T05. **T06b** (v2 with paper
values where a switch exists, "full v2") is the reference point for P2/P3. Since this table was first written, three
of the no-switch rows got switches on pncbf-v2 (5b06f9f) and T06b uses them:
- `ncbf.label_source=termination_window`: paper label, one head.
- `environment.termination.height_curriculum_scaled=True`: paper height termination.
- `environment.termination.body_tilt_threshold=1e6`: no tilt termination.

## T06a: `ncbf-q-fixed` (from cacffe7, commit bdd62eb)
| fix | switch (default = fixed) | paper value |
|---|---|---|
| F1 critic input | `ncbf.critic_input=pre_step`: (s_t, a_t) recorded before `env.step`, a_t = executed action | `post_step`: env state after the step, i.e. (0, a_0) on the **first** step (the env reset internal last_state to 0) and (0, 0) on the last step of every episode |
| F1 env | `environment.reset_last_state=observation`: internal last_state = s_0 after a reset | `zeros` |
| F1 shield | `ncbf.first_step_prev_value=gamma_c`: step 0 of an episode uses h_prev = gamma_c (logit 0.405) | `input`: h(0, 0) |
| F2 | `ncbf.temporal_mode=adjacent`: paper eq. 4, ReLU(−(h(s_{t+1},a_{t+1}) − (1−η)h(s_t,a_t) − η γ_c)) on adjacent safe→safe pairs inside one episode. a_{t+1}, y_{t+1} and pair validity are stored in the pos/neg buffers. `ncbf.temporal_space=prob` (sigmoid h and sigmoid(γ_c) = 0.6, the scale of the paper's term) or `logit` (the space of the paper's shield) | `legacy_raw_vs_executed` (h(s_t,a_exec) vs h(s_t,a_raw)) |
| replay bug | `ncbf.fix_replay_action_permutation=True` | `False`: `actions` not shuffled with the other fields; only the legacy term uses them |

Not changed in T06a:
- the anticipation loss still uses the post-step (last_state, last_action);
- the labels (ncbf-q window) and everything else stay as in the paper code.

PPO is not affected by any of these fixes (shield off; the critic does not act on the policy; no RNG stream
changes), so T06a reuses T05 PPO. Tests: `tests/test_t06a_fixes.py` on that branch (6 passed). Tiny CPU training
runs pass with all fixes on and with all switches set to the paper values.

## Main finding
**v2 cannot reproduce the paper code with only the fixes changed.** About 15 behavior changes between cacffe7
and v2 have **no switch**. They are marked "no switch" below and include: labels, the critic input and training
schedule, the shield's ensemble aggregation and scale, height termination, and policy LayerNorm.
Comparing T05 (paper code) with a v2 run therefore mixes the fixes with all of these. Two options were proposed to
the main session (decision pending):
- **T06a** = branch `ncbf-q-fixed` from cacffe7 with only F1, F2 (+ the `sample_and_merge` action bug) patched,
  each behind a switch. This isolates the fixes.
- **T06b** = v2 with the paper value wherever a switch exists (the table below). This is the "full v2" point.

B1/B2/B3/B7/B8 were regressions introduced on `multistep`. The paper code did not have them:
- B1: it trained 5 independent members.
- B2: it paired (s_t, a_exec,t), and that pairing is correct except on the terminal step (= F1).
- B3/B7: its shield already used the η-scaled form `c = h_x − η(h_x − γ_c) + a·u0 − h_u0`.
- B8: its anticipation target did not depend on the policy parameters, i.e. effectively stop-gradient.
So in T06b these are set to the paper-equivalent value, which is also the fixed value.

## Table (T06b column = value used by `experiments/pncbf/T06/t06.sbatch`)

### Config keys whose v2 default differs from the paper run
| key | paper | v2 default | T06b | note |
|---|---|---|---|---|
| algorithm.total_timesteps | 1e9 | 2e9 | 1e9 | |
| environment.nr_envs | 1024 | 4096 | 1024 | |
| environment.terrain.type | plane | hfield_diverse | plane | |
| environment.command.clip_max_velocity | 2 | 1 | 2 | |
| environment.command.max_velocity_per_m_factor | 2 | 2 | 2 | set explicitly (as in the paper args) |
| environment.env_curriculum_level_success_episode_return | 8 | 8 | 8 | explicit |
| environment.ncbf_use_policy_observations | True | (key) | True | explicit |
| algorithm.ncbf.use_safety_layer | True (PPO: False) | False | per method | |
| algorithm.ncbf.H | 25 / 1 / 10 / 50 | 25 | per method | |
| algorithm.ncbf.gamma_c | 0.405 **logit** (PPO 0) | 0.5 prob | **0.6** (= sigmoid(0.405)) | v2 thresholds probabilities (B3); same threshold, not the same geometry (see shield rows) |
| algorithm.ncbf.eta_cbf | 0.5 | 1.0 | 0.5 | |
| algorithm.ncbf.lr | 1e-3 | 1e-3 | 1e-3 | |
| algorithm.ncbf.coef_decay_lambda | 0.95 | 0.95 | 0.95 | |
| algorithm.ncbf.nr_minibatches | 64 | 50 | 64 | **not read by v2** (no separate critic update) |
| algorithm.ncbf.w_cbf / w_clf / w_wd | 1 / 1 / 0 | 1 / 1 / 0 | same | **not read by v2**; v2's term is `w_temporal` |
| algorithm.ncbf.w_lip | 0 | 0 | 0 | |
| algorithm.ncbf_buffer.pos_buffer_size | 50 | (no key) | – | **no pos buffer in v2** |
| algorithm.ncbf_buffer.neg_buffer_size | 20 | 1 | **5 (deviation)** | 20 OOMs on 24 GB: v2's buffer also stores the 10-step history stack (~5.5 KB/sample, 2.6 M samples); smoke 180597 failed allocating 16.6 GiB. Override with NEG_BUFFER=… |
| algorithm.ncbf_buffer.neg_sampling_ratio | 0.2 (share of the 32768 critic minibatch drawn from the neg buffer, rest from pos) | 0.5 | 0.2 | v2: extra 0.2×32768 neg-replay samples on top of the on-policy minibatch; different meaning |
| algorithm.ncbf.pretrain.* | 2 steps / 100 mb | (no key) | – | **no pretraining in v2** |
| algorithm.ncbf.action_clipping | False | False | False | |

### v2-only keys (paper behavior where a switch exists; fix switches = fixed)
| key | v2 default | paper behavior | T06b | call |
|---|---|---|---|---|
| ncbf.independent_ensemble (B1) | True | independent members | True | paper = fixed |
| ncbf.ensemble_bootstrap_prob | 0.8 | no bootstrap; each member drew its own minibatches | 1.0 | closest; v2 members then see identical batches (diversity from init only) |
| ncbf.critic_action_index (B2) | current | (s_t, a_exec,t) | current | paper = fixed |
| ncbf.terminal_critic_input (F1) | pre_reset | zeros | **pre_reset** | fix |
| ncbf.first_step_prev_value (F1) | gamma_c | h(0,0) | **gamma_c** | fix (plan 4.6) |
| ncbf.w_temporal / temporal_mode / temporal_pairs (F2) | 0 / adjacent / safe_safe | w_cbf=1, raw-vs-exec at same state | **1.0 / adjacent / safe_safe** | fix; weight matches the paper's w_cbf=1 |
| ncbf.gamma_c_scaled_by_eta (B3/B7) | True | η-scaled (already in the paper code) | True | paper = fixed |
| ncbf.label_discount | False | always on (λ^k on negatives) | **True** | paper value |
| ncbf.loss_coef | 0.1 | 1 (separate optimizer) | **1.0** | paper scale of the critic loss |
| ncbf.output_distribution | deterministic | deterministic | deterministic | |
| ncbf.tighten_uncertainty / safety_layer_std_coeff_start,final | total / −2, 1 | no β·σ tightening | **none** | paper value |
| ncbf.slack_uncertainty | none (1/λ) | σ²/λ with σ = clip(ens. std of logits, 1, 10) | none | no exact switch; v2 "ensemble" σ is a different quantity |
| ncbf.max_delta_u | 0.5 | no cap | **0** | paper value |
| ncbf.safety_layer_projection | soft_slack | soft slack | soft_slack | |
| ncbf.safety_layer_min_grad_norm / post_check_actual_residual | 0 / False | – | same | |
| ncbf.anticipation_through_shield (B8) | False | target independent of θ | False | paper = stop-gradient; plan text says through the shield → keep False for T06 (paper code), revisit in P2 |
| ncbf.policy_loss_coef | 0.01 | 0.01 | 0.01 | |
| algorithm.target_kl | 0.03 (gates policy updates) | none | **0** | paper value |
| algorithm.adaptive_lr | False | annealed lr | False | |
| algorithm.next_step_predictor.aux_loss_coef | 1 | no encoder/decoder | 1 | **no switch to remove the latent from the critic input**; aux loss only trains encoder/decoder |
| use_history_latent_for_policy/value, use_decoder_output_for_policy | False | – | False | |
| environment.nr_history_steps | 10 | – | 10 | feeds the critic latent |
| environment.command.velocity_ratio | (1, 0.5, 0.2) | (1, 1, 1) | **(1,1,1)** | paper value (made a tuple so the CLI can override it) |
| environment.termination.body_tilt_threshold | 0.4 rad (terminates) | no tilt termination | **1e6** | paper value; with label_source=termination_window the tilt head is not used |
| environment.termination.height_curriculum_scaled | False | True | **True** | paper value (new switch) |
| ncbf.label_source / safe_fall_H | env_targets / −1 | termination window, 1 head | **termination_window** / −1 | paper value (new switch) |
| environment.record_transition_physics_state | False | – | False | |

### Behavior changes with **no switch** (present in any v2 run)
| area | paper (cacffe7) | v2 |
|---|---|---|
| ~~height termination~~ | now switchable: `termination.height_curriculum_scaled` | |
| ~~critic label / heads~~ | now switchable: `ncbf.label_source=termination_window` | |
| critic input | policy obs + action | policy obs + action + history-encoder latent (64-d) |
| critic training | separate update per iteration: 64 minibatches per member from pos/neg buffers (80/20), Adam per member, clip 5 per member; pretraining | joint with encoder/decoder inside the PPO minibatch scan (nr_epochs × nr_minibatches steps on the on-policy batch + neg replay), one stacked Adam, global-norm clip over all members |
| shield value | CVaR (mean of lowest 3 of 5) on **logits**, linear CBF in logits | ensemble mean of soft-min(sigmoid(logits)), linear CBF in **probabilities** |
| policy / value nets | 512-LayerNorm-elu, 256, 128 | same sizes, no LayerNorm |
| GAE | bootstraps through truncations (1 − terminations) | stops at truncations (1 − dones) |
| anticipation loss | shields the sampled action with post-step (s_t, a_t) as h_prev, mean over action dims | shields the mean action, pre-step h_prev, β curriculum, sum over action dims |
| in-training eval | deterministic policy, shield off | shield on (β = 1) |
| paper bug | `sample_and_merge` permuted all fields but `actions` → CBF term paired a_raw with wrong states | n/a (term rewritten) |
| obs | – | +2 safety bits at the end of the observation (not in the policy/critic index sets) |
| RNG | 5-way key split | 7-way + per-minibatch splits |

## Legacy checkpoint loader (fixed on pncbf-v2)
`PPO.load` used to set `gamma_c_scaled_by_eta=False` and `critic_action_index=previous` on every checkpoint whose
config lacks those keys, which includes cacffe7 checkpoints. That was wrong for them, because the paper code already
had the η form and the (s_t, a_t) pairing. Now:
- `checkpoint_code_family()` tells "paper" (has `ncbf.pretrain` / `pos_buffer_size`) from "multistep" (pre-fix)
  and "current";
- `legacy_ncbf_defaults(family)` gives the family's behavior;
- loading a paper checkpoint raises a clear error, since 5 separate TrainStates and no encoder/decoder cannot be
  restored into v2 (eval uses the paper code via `PaperAgent`).
Test: `test_legacy_loader_defaults_by_code_family`, which uses the real zdqj50w0 config.

## Label convention (asked: keep ncbf-q's window)
`ncbf_targets.window_any_done_next_H` is identical to the paper's (y = no termination at transitions
t..t+H, i.e. H+1 transitions; mask drops windows that hit a truncation/rollout end). But v2 trains the critic on
`receding_min_target` (min of env indicator dims over H next states), and uses y only to select replay
negatives. So by default the paper window is **not** v2's training label. The new switch
`ncbf.label_source=termination_window` (1 head) restores it, and T06b uses it. The one-step mismatch: the termination
window covers H+1 transitions while the target covers H states, so when termination is exactly H+1 steps away
y = 0 but target = safe. In T06a (paper code) the paper window is used unchanged.

## Smoke tests (2026-10-23, scavenger, 60 updates of 1024x128 = 7.86 M steps)
| job | code | GPU | result | peak GPU mem | MaxRSS (host) | speed | notes |
|---|---|---|---|---|---|---|---|
| 180597 | T06b, neg_buffer 20 | 3090 | **OOM** (alloc 16.6 GiB at start) | – | 9.2 GB | – | → neg_buffer_size 5 (deviation) |
| 180603 | T06b @62d1616, neg_buffer 5 | 3090 | COMPLETED, 60 updates, 0 restarts | 18.5 GB | 9.3 GB | 3.9 s/update ≈ 34k steps/s (≈ 8.3 h per 1e9) | compile 5.5 min; wandb `T06_fixed_ii_h25_seed_42_smoke-dc0bdba2` |
| 180598 | T06a @bdd62eb | A6000 (last attempt) | COMPLETED, 60 updates, **2 scavenger preemptions** (restarted from scratch, earlier run dirs moved to `_failed_attempts/`) | 18.8 GB | 12.0 GB | 9 s/update ≈ 14.6k steps/s (≈ 19 h per 1e9, paper 15k) | wandb `rt7ymemj`; log of earlier attempts lost (t06a now has `--open-mode=append`) |
Both fit a 24 GB GPU and the 16 GB host limit (8G x 2 CPUs). The jax cache on the shared home was not exercised by
a requeue (180603 was not preempted).
