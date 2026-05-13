from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from observation import CommandState, T1ObservationBuilder
from policy import load_policy, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "config.yaml")
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_dof = int(cfg["robot"]["num_dof"])
    builder = T1ObservationBuilder(cfg)
    observation = builder.build(
        dof_pos=np.asarray(cfg["common"]["default_qpos"], dtype=np.float32),
        dof_vel=np.zeros(num_dof, dtype=np.float32),
        previous_action=np.zeros(num_dof, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        command=CommandState(height=float(cfg["command"]["standing_height"])),
    )
    print(f"observation_shape={observation.shape}")

    model_path = resolve_path(cfg["policy"]["model_path"], cfg_path.parent)
    if not model_path.exists():
        message = f"model_missing={model_path}"
        if args.require_model:
            raise FileNotFoundError(message)
        print(message)
        return

    policy = load_policy(cfg["policy"], config_dir=cfg_path.parent)
    policy.reset(observation)
    output = policy.act(observation)
    print(f"action_shape={output.action.shape}")
    print(f"action_min={float(np.min(output.action)):.6f}")
    print(f"action_max={float(np.max(output.action)):.6f}")


if __name__ == "__main__":
    main()

