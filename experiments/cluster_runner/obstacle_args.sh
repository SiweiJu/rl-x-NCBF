#!/bin/bash
# Sourced by run_cluster_experiment.sh and run_constrained_baseline_experiment.sh.
# Appends moving-obstacle flags (environment.obstacles.*) to the caller's extra_args array.
# Ranges are tuples on the command line, e.g. OBSTACLES_SPEED_RANGE="(2.0, 2.0)".
if [[ "${OBSTACLES_ENABLED:-False}" == "True" ]]; then
    extra_args+=(--environment.obstacles.enabled=True)
    optional_obstacle_flags=(
        "OBSTACLES_NR:nr_obstacles"
        "OBSTACLES_INCLUDE_OBSERVATIONS:include_observations"
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
fi
