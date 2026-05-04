from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.termination_functions.booster import BoosterTermination


class BoosterBallTermination(BoosterTermination):
    def should_terminate(self, internal_state):
        booster_terminated = super().should_terminate(internal_state)
        ball_dropped = self.env.ball_plate_ball_has_dropped(internal_state)
        plate_dropped = self.env.ball_plate_plate_has_dropped(internal_state)
        return booster_terminated | ball_dropped | plate_dropped
