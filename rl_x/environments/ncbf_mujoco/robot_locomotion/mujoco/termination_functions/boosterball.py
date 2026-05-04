from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.termination_functions.booster import BoosterTermination


class BoosterBallTermination(BoosterTermination):
    def should_terminate(self):
        booster_terminated = super().should_terminate()
        ball_dropped = self.env.ball_plate_ball_has_dropped()
        plate_dropped = self.env.ball_plate_plate_has_dropped()
        return booster_terminated or ball_dropped or plate_dropped
