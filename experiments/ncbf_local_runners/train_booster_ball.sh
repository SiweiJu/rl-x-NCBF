#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXPERIMENTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SEED="${SEED:-0}"
NR_ENVS="${NR_ENVS:-1024}"
RUN_NAME="${RUN_NAME:-booster_refactored_no_safety}"

cd "${EXPERIMENTS_DIR}"

conda run --no-capture-output -n ncbf-mjx python experiment.py \
    --algorithm.name="ncbf_ppo.flax_full_jit" \
    --algorithm.total_timesteps=200000000 \
    --algorithm.minibatch_size=4096 \
    --algorithm.learning_rate=4e-4 \
    --algorithm.nr_steps=128 \
    --algorithm.nr_epochs=4 \
    --algorithm.ncbf.use_safety_layer=False \
    --algorithm.ncbf.H=25 \
    --algorithm.ncbf.action_clipping=False \
    --algorithm.ncbf.gamma_c=0.1 \
    --algorithm.ncbf.eta_cbf=0.5 \
    --algorithm.ncbf_buffer.neg_buffer_size=1 \
    --algorithm.ncbf.coef_decay_lambda=0.95 \
    --algorithm.next_step_predictor.lr=1e-5 \
    --algorithm.next_step_predictor.aux_loss_coef=1 \
    --algorithm.use_decoder_output_for_policy=True \
    --algorithm.adaptive_lr=True \
    --environment.name="ncbf_mujoco.robot_locomotion.mjx_boosterball" \
    --environment.seed="${SEED}" \
    --environment.nr_envs="${NR_ENVS}" \
    --environment.train_robot="booster_t1" \
    --environment.use_booster_defaults=True \
    --environment.ncbf_use_policy_observations=True \
    --environment.command.sampling_type="step_probability_and_reset" \
    --environment.episode_length_in_seconds=20 \
    --environment.env_curriculum_level_success_episode_return=20 \
    --environment.terrain.type="plane" \
    --environment.termination.terminate_on_ball_plate_drop=False \
    --environment.ball_plate.enabled=True \
    --environment.reward.type="G1ball" \
    --environment.ball_plate.stand_after_drop=True \
    --environment.ball_plate.post_drop_truncation_seconds=3.0 \
    --environment.ball_plate.include_observations=False \
    --environment.reward.ball_plate_drop_penalty_coeff=10.0 \
    --environment.reward.below_height_penalty_coeff=50.0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="catherineju-rwth-aachen-university" \
    --runner.project_name="202605_ncbf" \
    --runner.exp_name="ncbf_in_aux_debug" \
    --runner.run_name="${RUN_NAME}"
