#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p log

scripts=(
    train_go2_locomotion_sl_off.sh
    train_go2_locomotion_sl_on.sh
    train_g1_locomotion_sl_off.sh
    train_g1_locomotion_sl_on.sh
    train_booster_locomotion_sl_off.sh
    train_booster_locomotion_sl_on.sh
    train_g1_ball_sl_off_obs_on.sh
    train_g1_ball_sl_off_obs_off.sh
    train_g1_ball_sl_on_obs_on.sh
    train_g1_ball_sl_on_obs_off.sh
    train_booster_ball_sl_off_obs_on.sh
    train_booster_ball_sl_off_obs_off.sh
    train_booster_ball_sl_on_obs_on.sh
    train_booster_ball_sl_on_obs_off.sh
)

for script in "${scripts[@]}"; do
    echo "Submitting ${script}"
    sbatch "${script}"
done
