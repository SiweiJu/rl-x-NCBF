from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.termination_functions.booster import BoosterTermination


class BoosterBallTermination(BoosterTermination):
    def __init__(self, env):
        super().__init__(env)
        self.terminate_on_ball_plate_drop = self.env.env_config["termination"].get("terminate_on_ball_plate_drop", True)


    def should_terminate(self):
        booster_terminated = super().should_terminate()
        ball_dropped = self.env.ball_plate_ball_has_dropped()
        plate_dropped = self.env.ball_plate_plate_has_dropped()
        ball_plate_drop_terminated = self.terminate_on_ball_plate_drop and (ball_dropped or plate_dropped)
        return booster_terminated or ball_plate_drop_terminated
