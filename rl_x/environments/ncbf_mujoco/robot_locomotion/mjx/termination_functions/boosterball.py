from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.termination_functions.booster import BoosterTermination


class BoosterBallTermination(BoosterTermination):
    def __init__(self, env):
        super().__init__(env)
        self.terminate_on_ball_plate_drop = self.env.env_config["termination"].get("terminate_on_ball_plate_drop", True)


    def should_terminate(self, internal_state):
        booster_terminated = super().should_terminate(internal_state)
        ball_dropped = self.env.ball_plate_ball_has_dropped(internal_state)
        plate_dropped = self.env.ball_plate_plate_has_dropped(internal_state)
        if self.terminate_on_ball_plate_drop and not self.env.stand_after_ball_plate_drop:
            ball_plate_drop_terminated = ball_dropped | plate_dropped
        else:
            ball_plate_drop_terminated = False
        return booster_terminated | ball_plate_drop_terminated
