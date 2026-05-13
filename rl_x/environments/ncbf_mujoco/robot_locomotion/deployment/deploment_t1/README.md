# Booster T1 Deployment

This folder runs a Booster T1 policy trained by this repo through the Booster
`humanoid_bridge` `RobotClient`.

Put an RL-X saved model in:

```bash
rl_x/environments/ncbf_mujoco/robot_locomotion/deployment/deploment_t1/policies/latest.model
```

The default loader expects this repo's Orbax zip-style `.model` artifact. A
TorchScript `.pt` policy also works if `policy.model_path` points to it.

Run a local smoke check:

```bash
python smoke_test.py
```

Run on the robot:

```bash
python run.py --config config.yaml
```

Controls match the Booster bridge example: start bridge control with
`LT+START`, start policy with `LT+A`, stop with `BACK`, use the d-pad for
linear velocity, right stick x for yaw, and stick press to zero commands.

