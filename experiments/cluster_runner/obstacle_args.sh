#!/bin/bash
# Sourced by run_cluster_experiment.sh and run_constrained_baseline_experiment.sh.
# Appends moving-obstacle flags (environment.obstacles.*) to the caller's extra_args array.
# Ranges are tuples on the command line, e.g. OBSTACLES_SPEED_RANGE="(2.0, 2.0)".
if [[ "${OBSTACLES_ENABLED:-False}" == "True" ]]; then
    source "$(dirname "${BASH_SOURCE[0]}")/obstacle_presets.sh"
    extra_args+=(--environment.obstacles.enabled=True)
    optional_obstacle_flags=(
        "OBSTACLES_NR:nr_obstacles"
        "OBSTACLES_POLICY_OBSERVATION:policy_observation"
        "OBSTACLES_NCBF_OBSERVATION:ncbf_observation"
        "OBSTACLES_VALUE_OBSERVATION:value_observation"
        "OBSTACLES_PREDICT_NEXT_OBSERVATION:predict_next_observation"
        "OBSTACLES_SCAN_NR_RAYS:scan.nr_rays"
        "OBSTACLES_SCAN_FOV_DEGREES:scan.fov_degrees"
        "OBSTACLES_SCAN_MAX_RANGE:scan.max_range"
        "OBSTACLES_SCAN_NOISE_RANGE:scan.noise_range"
        "OBSTACLES_SCAN_DROPOUT_PROB_RANGE:scan.dropout_prob_range"
        "OBSTACLES_SCAN_LATENCY_STEPS_RANGE:scan.latency_steps_range"
        "OBSTACLES_SCAN_ANGLE_JITTER:scan.angle_jitter"
        "OBSTACLES_SCAN_SPURIOUS_PROB:scan.spurious_prob"
        "OBSTACLES_BOX_HALF_LENGTH_RANGE:shapes.box_half_length_range"
        "OBSTACLES_BOX_HALF_WIDTH_RANGE:shapes.box_half_width_range"
        "OBSTACLES_AIM_OFFSET:aim_offset"
        "OBSTACLES_AIM_LEAD:aim_lead"
        "OBSTACLES_TERMINATE_ON_COLLISION:terminate_on_collision"
        "OBSTACLES_COLLISION_PENALTY:collision_penalty"
        "OBSTACLES_SPEED_RANGE:speed_range"
        "OBSTACLES_SPEED_CURRICULUM:speed_curriculum"
        "OBSTACLES_SPAWN_DISTANCE_RANGE:spawn_distance_range"
        "OBSTACLES_RADIUS_RANGE:radius_range"
        "OBSTACLES_ROBOT_RADIUS:robot_radius"
    )
    for entry in "${optional_obstacle_flags[@]}"; do
        env_name="${entry%%:*}"
        flag_name="${entry#*:}"
        if [[ -n "${!env_name:-}" ]]; then
            extra_args+=(--environment.obstacles."${flag_name}"="${!env_name}")
        fi
    done
    # Shape and motion mixes, e.g. OBSTACLES_SHAPE_WEIGHTS="circle=1 box=1 circles=1".
    # Every shape (motion) not listed gets weight 0.
    obstacle_weight_args() {
        local group="$1" listed="$2"
        shift 2
        local name weight entry
        for name in "$@"; do
            weight=0.0
            for entry in ${listed}; do
                if [[ "${entry%%=*}" == "${name}" ]]; then
                    weight="${entry#*=}"
                fi
            done
            extra_args+=(--environment.obstacles."${group}".weights."${name}"="${weight}")
        done
    }
    if [[ -n "${OBSTACLES_SHAPE_WEIGHTS:-}" ]]; then
        obstacle_weight_args shapes "${OBSTACLES_SHAPE_WEIGHTS}" circle box circles l_shape
    fi
    if [[ -n "${OBSTACLES_MOTION_WEIGHTS:-}" ]]; then
        obstacle_weight_args motions "${OBSTACLES_MOTION_WEIGHTS}" constant turning stop_and_go accelerating random_walk
    fi
fi
