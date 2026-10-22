from ml_collections import config_dict
import time


def get_config(runner_mode):
    config = config_dict.ConfigDict()

    config.mode = runner_mode

    config.track_console = False
    config.track_tb = False
    config.track_wandb = False
    config.wandb_entity = "placeholder"
    config.project_name = "placeholder"
    config.exp_name = "placeholder"
    config.run_name = f"{int(time.time())}"
    config.notes = "placeholder"
    config.wandb_tags = ""  # comma-separated wandb tags, e.g. "phase/P0,method/pncbf,robot/go2"
    config.wandb_init_timeout = 600

    config.save_model = False
    config.load_model = ""

    # Requeue-safe resume (e.g. Slurm --qos=scavenger --requeue). With resume=True the run dir and the wandb run id
    # are derived from project/exp/run_name (run_name must be set explicitly), resume checkpoints are written
    # atomically to <run_dir>/resume/ and a rerun of the same command continues from the latest one (or starts
    # fresh if there is none). Only supported by algorithms whose model class sets SUPPORTS_RESUME.
    config.resume = False
    config.save_interval_iterations = 0  # resume checkpoint every N learning iterations (0 = only by wall time)
    config.save_interval_minutes = 20.0  # resume checkpoint at least every N minutes of wall time (<= 0 disables)

    config.nr_test_episodes = 10  # if runner mode = test

    config.jax_cache_dir = "/tmp/jax_cache"
    config.jax_default_matmul_precision = "default"

    return config
