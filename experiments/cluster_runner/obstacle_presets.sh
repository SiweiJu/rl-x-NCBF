#!/bin/bash
# Sourced by obstacle_args.sh. OBSTACLES_PRESET selects an obstacle randomization preset; it only fills
# OBSTACLES_* variables that are still unset, so explicit settings win. The presets match
# ncbf_tools/obstacle_generalization_probe.py:
#   train            circles, boxes and chains of circles; constant, turning, stop-and-go and
#                    accelerating motion; sensor noise, dropout, latency, jitter, spurious returns
#   held_out_shape   train, but L-shaped obstacles only
#   held_out_motion  train, but random-walk motion only
#   held_out_size    train, but larger circles and boxes
#   held_out_sensor  train, but more noise, dropout, latency and spurious returns
if [[ -n "${OBSTACLES_PRESET:-}" ]]; then
    case "${OBSTACLES_PRESET}" in
        train|held_out_shape|held_out_motion|held_out_size|held_out_sensor) ;;
        *) echo "Unknown OBSTACLES_PRESET=${OBSTACLES_PRESET}" >&2; exit 2 ;;
    esac
    case "${OBSTACLES_PRESET}" in
        held_out_shape) : "${OBSTACLES_SHAPE_WEIGHTS:=l_shape=1}" ;;
        held_out_motion) : "${OBSTACLES_MOTION_WEIGHTS:=random_walk=1}" ;;
        held_out_size)
            : "${OBSTACLES_RADIUS_RANGE:=(0.3, 0.45)}"
            : "${OBSTACLES_BOX_HALF_LENGTH_RANGE:=(0.5, 0.8)}"
            : "${OBSTACLES_BOX_HALF_WIDTH_RANGE:=(0.3, 0.45)}"
            ;;
        held_out_sensor)
            : "${OBSTACLES_SCAN_NOISE_RANGE:=(0.06, 0.08)}"
            : "${OBSTACLES_SCAN_DROPOUT_PROB_RANGE:=(0.15, 0.2)}"
            : "${OBSTACLES_SCAN_LATENCY_STEPS_RANGE:=(2, 3)}"
            : "${OBSTACLES_SCAN_SPURIOUS_PROB:=0.01}"
            ;;
    esac
    # Everything not changed above comes from the training distribution.
    : "${OBSTACLES_SHAPE_WEIGHTS:=circle=1 box=1 circles=1}"
    : "${OBSTACLES_MOTION_WEIGHTS:=constant=1 turning=1 stop_and_go=1 accelerating=1}"
    : "${OBSTACLES_SCAN_NOISE_RANGE:=(0.0, 0.04)}"
    : "${OBSTACLES_SCAN_DROPOUT_PROB_RANGE:=(0.0, 0.05)}"
    : "${OBSTACLES_SCAN_LATENCY_STEPS_RANGE:=(0, 1)}"
    : "${OBSTACLES_SCAN_ANGLE_JITTER:=0.01}"
    : "${OBSTACLES_SCAN_SPURIOUS_PROB:=0.002}"
fi
