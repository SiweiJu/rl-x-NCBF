from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


def _add_repo_root_to_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "rl_x").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    return (Path(base_dir) / path).resolve()


@dataclass
class PolicyOutput:
    action: np.ndarray
    raw_action: np.ndarray


class DeploymentPolicy:
    def act(self, observation: np.ndarray) -> PolicyOutput:
        raise NotImplementedError

    def reset(self, observation: np.ndarray | None = None) -> None:
        pass


class TorchScriptPolicy(DeploymentPolicy):
    def __init__(self, cfg: dict, config_dir: str | Path | None = None):
        import torch

        self.torch = torch
        self.model_path = resolve_path(cfg["model_path"], config_dir)
        self.action_size = int(cfg.get("action_size", 23))
        self.clip_actions = float(cfg.get("clip_actions", cfg.get("action_clip", 0.0)))

        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()

    def act(self, observation: np.ndarray) -> PolicyOutput:
        with self.torch.no_grad():
            obs = self.torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
            action = self.model(obs)
            if isinstance(action, (tuple, list)):
                action = action[0]
            action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if action_np.shape[0] != self.action_size:
            raise ValueError(f"Policy returned {action_np.shape[0]} actions, expected {self.action_size}.")
        raw_action = action_np.copy()
        if self.clip_actions > 0.0:
            action_np = np.clip(action_np, -self.clip_actions, self.clip_actions)
        return PolicyOutput(action=action_np, raw_action=raw_action)


class _ActionSpaceType:
    CONTINUOUS = "continuous"


class _ObservationSpaceType:
    FLAT_VALUES = "flat_values"


def _get_policy(config, env):
    import jax
    import jax.numpy as jnp

    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(
        env,
        "policy_observation_indices",
        jnp.arange(env.single_observation_space.shape[0]),
    )
    use_history_latent = getattr(config.algorithm, "use_history_latent_for_policy", False)
    use_decoder_output = getattr(config.algorithm, "use_decoder_output_for_policy", False)

    if (
        action_space_type == _ActionSpaceType.CONTINUOUS
        and observation_space_type == _ObservationSpaceType.FLAT_VALUES
    ):
        return (
            _Policy(
                env.single_action_space.shape,
                config.algorithm.std_dev,
                policy_observation_indices,
                config.algorithm.hidden_layers,
                use_history_latent,
                use_decoder_output,
            ),
            _get_processed_action_function(
                config.algorithm.action_clipping_and_rescaling,
                getattr(config.algorithm, "action_clip", 0.0),
                jnp.array(env.single_action_space.low),
                jnp.array(env.single_action_space.high),
            ),
        )
    raise ValueError("Unsupported action/observation space for deployment policy.")


def _get_processed_action_function(action_clipping_and_rescaling, action_clip, env_as_low, env_as_high):
    import jax
    import jax.numpy as jnp

    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))

        return jax.jit(get_clipped_and_scaled_action)
    if action_clip > 0.0:
        return jax.jit(lambda x: jnp.clip(x, -action_clip, action_clip))
    return jax.jit(lambda x: x)


def _get_history_encoder(config, env):
    encoder_type = getattr(config.algorithm.next_step_predictor, "history_encoder_type", "FFNN")
    hidden_size = getattr(config.algorithm.next_step_predictor, "history_encoder_hidden_size", 128)
    observation_indices = getattr(env, "policy_observation_indices")
    history_length = getattr(env, "nr_history_steps", 10)

    if encoder_type == "FFNN":
        return _FFNNEncoder(
            hidden_size=hidden_size,
            observation_indices=observation_indices,
            history_length=history_length,
        )
    if encoder_type == "GRU":
        return _GRUEncoder(hidden_size=hidden_size)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def _get_decoder(config, env):
    history_encoder_hidden_size = getattr(
        config.algorithm.next_step_predictor,
        "history_encoder_hidden_size",
        128,
    )
    prediction_indices = getattr(env, "next_state_indices", None)
    decoder_output_dim = prediction_indices.shape[0]
    return _FeedforwardDecoder(hidden_size=history_encoder_hidden_size, output_dim=decoder_output_dim)


def _dense_init_scale(scale):
    return orthogonal(scale)


def _zero_init():
    return constant(0.0)


def _logstd_init(std_dev):
    return constant(jnp.log(std_dev))


class _Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    policy_observation_indices: Sequence[int]
    hidden_layers: Sequence[int]
    use_history_latent: bool = False
    use_decoder_output: bool = False

    @nn.compact
    def __call__(self, x, history_latent=None, decoder_output=None):
        x = x[..., self.policy_observation_indices]
        if self.use_history_latent:
            if history_latent is None:
                raise ValueError("Policy was configured to use history latent, but none was passed.")
            x = jnp.concatenate([x, history_latent], axis=-1)
        if self.use_decoder_output:
            if decoder_output is None:
                raise ValueError("Policy was configured to use decoder output, but none was passed.")
            x = jnp.concatenate([x, decoder_output], axis=-1)
        policy_mean = x
        for hidden_units in self.hidden_layers:
            policy_mean = nn.Dense(
                hidden_units,
                kernel_init=_dense_init_scale(np.sqrt(2)),
                bias_init=_zero_init(),
            )(policy_mean)
            policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(
            np.prod(self.as_shape).item(),
            kernel_init=_dense_init_scale(0.01),
            bias_init=_zero_init(),
        )(policy_mean)
        policy_logstd = self.param(
            "policy_logstd",
            _logstd_init(self.std_dev),
            (1, np.prod(self.as_shape).item()),
        )
        return policy_mean, policy_logstd


class _FFNNEncoder(nn.Module):
    hidden_size: int
    observation_indices: Sequence[int]
    history_length: int

    @nn.compact
    def __call__(self, x):
        x = x[..., self.observation_indices]
        x = x.reshape(*x.shape[:-2], self.history_length * len(self.observation_indices))
        x = nn.Dense(512, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(x)
        x = nn.elu(x)
        x = nn.Dense(256, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(x)
        x = nn.elu(x)
        return nn.Dense(self.hidden_size, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(x)


class _GRUEncoder(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        gru = nn.RNN(
            nn.GRUCell(features=self.hidden_size),
            return_carry=True,
            time_major=False,
        )
        carry, _ = gru(x)
        return carry


class _FeedforwardDecoder(nn.Module):
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, z, x, a):
        decoder_input = jnp.concatenate([z, a, x], axis=-1)
        x = nn.Dense(128, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(decoder_input)
        x = nn.elu(x)
        x = nn.Dense(256, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(x)
        x = nn.elu(x)
        return nn.Dense(self.output_dim, kernel_init=_dense_init_scale(np.sqrt(2)), bias_init=_zero_init())(x)


class RlxFlaxPolicy(DeploymentPolicy):
    def __init__(self, cfg: dict, config_dir: str | Path | None = None):
        import jax
        import jax.numpy as jnp
        import optax
        import orbax.checkpoint
        from flax.training import orbax_utils
        from flax.training.train_state import TrainState
        from ml_collections import ConfigDict

        self.jax = jax
        self.jnp = jnp
        self.optax = optax
        self.orbax_utils = orbax_utils
        self.TrainState = TrainState
        self.ConfigDict = ConfigDict
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        self.model_path = resolve_path(cfg["model_path"], config_dir)
        self.observation_size = int(cfg.get("observation_size", 81))
        self.action_size = int(cfg.get("action_size", 23))
        self.history_length = int(cfg.get("history_length", 10))
        self.decoder_output_size = int(cfg.get("decoder_output_size", 49))
        self.clip_actions = float(cfg.get("clip_actions", 0.0))
        self.use_jit = bool(cfg.get("jit", True))
        self.policy_observation_indices = self._policy_observation_indices(cfg)

        action_low = cfg.get("action_low", [-np.inf] * self.action_size)
        action_high = cfg.get("action_high", [np.inf] * self.action_size)
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        if action_low.shape[0] != self.action_size or action_high.shape[0] != self.action_size:
            raise ValueError("policy.action_low and policy.action_high must match policy.action_size.")

        algorithm_config = self._read_algorithm_config()
        algorithm_config = self._with_algorithm_defaults(algorithm_config)
        self.config = ConfigDict({"algorithm": algorithm_config})

        env = _DeploymentEnvSpec(
            action_size=self.action_size,
            observation_size=self.observation_size,
            action_low=action_low,
            action_high=action_high,
            history_length=self.history_length,
            decoder_output_size=self.decoder_output_size,
            policy_observation_indices=self.policy_observation_indices,
            action_space_type=_ActionSpaceType.CONTINUOUS,
            observation_space_type=_ObservationSpaceType.FLAT_VALUES,
            jnp=jnp,
        )

        self.policy, self.get_processed_action = _get_policy(self.config, env)
        self.encoder = _get_history_encoder(self.config, env)
        self.decoder = _get_decoder(self.config, env)

        self.use_history_latent = bool(getattr(self.config.algorithm, "use_history_latent_for_policy", False))
        self.use_decoder_output = bool(getattr(self.config.algorithm, "use_decoder_output_for_policy", False))

        key = jax.random.PRNGKey(0)
        policy_key, encoder_key, decoder_key = jax.random.split(key, 3)
        dummy_obs = jnp.zeros((1, self.observation_size), dtype=jnp.float32)
        dummy_history = jnp.zeros((1, self.history_length, self.observation_size), dtype=jnp.float32)
        dummy_latent = self.encoder.apply(self.encoder.init(encoder_key, dummy_history), dummy_history)
        dummy_decoder_output = (
            jnp.zeros((1, self.decoder_output_size), dtype=jnp.float32)
            if self.use_decoder_output
            else None
        )

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_obs, dummy_latent, dummy_decoder_output),
            tx=optax.chain(
                optax.clip_by_global_norm(0.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
            ),
        )
        self.encoder_state = TrainState.create(
            apply_fn=self.encoder.apply,
            params=self.encoder.init(encoder_key, dummy_history),
            tx=optax.chain(
                optax.clip_by_global_norm(0.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
            ),
        )
        self.decoder_state = TrainState.create(
            apply_fn=self.decoder.apply,
            params=self.decoder.init(
                decoder_key,
                dummy_latent,
                dummy_obs,
                jnp.zeros((1, self.action_size), dtype=jnp.float32),
            ),
            tx=optax.chain(
                optax.clip_by_global_norm(0.0),
                optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
            ),
        )

        self._restore_checkpoint()

        if self.use_jit:
            self.policy.apply = jax.jit(self.policy.apply)
            self.encoder.apply = jax.jit(self.encoder.apply)
            self.decoder.apply = jax.jit(self.decoder.apply)
            self.get_processed_action = jax.jit(self.get_processed_action)

        self.history = np.zeros((self.history_length, self.observation_size), dtype=np.float32)
        self.previous_observation = np.zeros(self.observation_size, dtype=np.float32)
        self.previous_action = np.zeros(self.action_size, dtype=np.float32)
        self.reset()

    def reset(self, observation: np.ndarray | None = None) -> None:
        if observation is None:
            observation = np.zeros(self.observation_size, dtype=np.float32)
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.history[:] = observation
        self.previous_observation[:] = observation
        self.previous_action[:] = 0.0

    def act(self, observation: np.ndarray) -> PolicyOutput:
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape[0] != self.observation_size:
            raise ValueError(f"Observation size {observation.shape[0]} does not match {self.observation_size}.")

        self.history[:-1] = self.history[1:]
        self.history[-1] = observation

        obs = self.jnp.asarray(observation[None, :])
        latent = None
        decoder_output = None

        if self.use_history_latent or self.use_decoder_output:
            latent = self.encoder.apply(self.encoder_state.params, self.jnp.asarray(self.history[None, ...]))

        if self.use_decoder_output:
            decoder_output = self.decoder.apply(
                self.decoder_state.params,
                self.jax.lax.stop_gradient(latent),
                self.jnp.asarray(self.previous_observation[None, :]),
                self.jnp.asarray(self.previous_action[None, :]),
            )
            decoder_output = self.jax.lax.stop_gradient(decoder_output)

        action_mean, _ = self.policy.apply(self.policy_state.params, obs, latent, decoder_output)
        processed_action = self.get_processed_action(action_mean)
        raw_action = np.asarray(self.jax.device_get(action_mean))[0].astype(np.float32)
        action = np.asarray(self.jax.device_get(processed_action))[0].astype(np.float32)
        if self.clip_actions > 0.0:
            action = np.clip(action, -self.clip_actions, self.clip_actions)

        self.previous_observation[:] = observation
        self.previous_action[:] = action
        return PolicyOutput(action=action, raw_action=raw_action)

    def _read_algorithm_config(self) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Policy model not found: {self.model_path}")
        with tempfile.TemporaryDirectory(prefix="rlx_t1_cfg_") as tmp_dir:
            shutil.unpack_archive(str(self.model_path), tmp_dir, "zip")
            config_path = Path(tmp_dir) / "config_algorithm.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config_algorithm.json inside {self.model_path}.")
            return json.loads(config_path.read_text(encoding="utf-8"))

    def _policy_observation_indices(self, cfg: dict) -> np.ndarray:
        configured_indices = cfg.get("policy_observation_indices")
        if configured_indices is not None:
            indices = np.asarray(configured_indices, dtype=np.int32).reshape(-1)
        elif self.observation_size == 81:
            indices = np.concatenate([
                np.arange(75, dtype=np.int32),
                np.arange(78, 81, dtype=np.int32),
            ])
        else:
            indices = np.arange(self.observation_size, dtype=np.int32)

        if indices.size == 0:
            raise ValueError("policy.policy_observation_indices must not be empty.")
        if np.any(indices < 0) or np.any(indices >= self.observation_size):
            raise ValueError(
                "policy.policy_observation_indices contains values outside "
                f"the observation size {self.observation_size}."
            )
        return indices

    def _restore_checkpoint(self) -> None:
        target = {
            "policy": {"params": self.policy_state.params},
        }
        if self.use_history_latent or self.use_decoder_output:
            target["encoder"] = {"params": self.encoder_state.params}
        if self.use_decoder_output:
            target["decoder"] = {"params": self.decoder_state.params}

        with tempfile.TemporaryDirectory(prefix="rlx_t1_model_") as tmp_dir:
            shutil.unpack_archive(str(self.model_path), tmp_dir, "zip")
            sharding_file = Path(tmp_dir) / "_sharding"
            if sharding_file.exists():
                sharding_file.unlink()
            restore_args = self.orbax_utils.restore_args_from_target(target)
            restored = self.checkpointer.restore(
                tmp_dir,
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            )
            self._assign_restored_model_params(restored)

    def _assign_restored_model_params(self, restored) -> None:
        if "policy" not in restored or "params" not in restored["policy"]:
            raise ValueError("Restored checkpoint does not contain policy params.")
        self.policy_state = self.policy_state.replace(params=restored["policy"]["params"])
        if self.use_history_latent or self.use_decoder_output:
            if "encoder" not in restored or "params" not in restored["encoder"]:
                raise ValueError("Policy needs history encoder params, but checkpoint has no encoder entry.")
            self.encoder_state = self.encoder_state.replace(params=restored["encoder"]["params"])
        if self.use_decoder_output:
            if "decoder" not in restored or "params" not in restored["decoder"]:
                raise ValueError("Policy needs decoder params, but checkpoint has no decoder entry.")
            self.decoder_state = self.decoder_state.replace(params=restored["decoder"]["params"])

    def _assign_restored_objects(self, restored) -> None:
        if "policy" not in restored:
            raise ValueError("Restored checkpoint does not contain a policy entry.")
        self.policy_state = self._coerce_train_state(self.policy_state, restored["policy"])
        if self.use_history_latent or self.use_decoder_output:
            if "encoder" not in restored:
                raise ValueError("Policy needs a history encoder, but checkpoint has no encoder entry.")
            self.encoder_state = self._coerce_train_state(self.encoder_state, restored["encoder"])
        if self.use_decoder_output:
            if "decoder" not in restored:
                raise ValueError("Policy needs a decoder output, but checkpoint has no decoder entry.")
            self.decoder_state = self._coerce_train_state(self.decoder_state, restored["decoder"])

    @staticmethod
    def _coerce_train_state(template, restored):
        if hasattr(restored, "params"):
            return restored
        if isinstance(restored, dict) and "params" in restored:
            updates = {"params": restored["params"]}
            if "step" in restored:
                updates["step"] = restored["step"]
            if "opt_state" in restored:
                updates["opt_state"] = restored["opt_state"]
            return template.replace(**updates)
        raise TypeError(f"Unsupported checkpoint state type: {type(restored)!r}")

    @staticmethod
    def _with_algorithm_defaults(algorithm_config: dict) -> dict:
        cfg = dict(algorithm_config)
        cfg.setdefault("std_dev", 1.0)
        cfg.setdefault("hidden_layers", [512, 256, 128])
        cfg.setdefault("action_clipping_and_rescaling", False)
        cfg.setdefault("action_clip", 0.0)
        cfg.setdefault("use_history_latent_for_policy", False)
        cfg.setdefault("use_decoder_output_for_policy", False)
        next_step_predictor = dict(cfg.get("next_step_predictor", {}))
        next_step_predictor.setdefault("history_encoder_type", "FFNN")
        next_step_predictor.setdefault("history_encoder_hidden_size", 128)
        cfg["next_step_predictor"] = next_step_predictor
        return cfg


class _DeploymentEnvSpec:
    def __init__(
        self,
        action_size: int,
        observation_size: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        history_length: int,
        decoder_output_size: int,
        policy_observation_indices: np.ndarray,
        action_space_type,
        observation_space_type,
        jnp,
    ):
        self.single_action_space = SimpleNamespace(
            shape=(action_size,),
            low=action_low,
            high=action_high,
        )
        self.single_observation_space = SimpleNamespace(shape=(observation_size,))
        self.policy_observation_indices = jnp.asarray(policy_observation_indices, dtype=jnp.int32)
        self.critic_observation_indices = jnp.arange(observation_size)
        self.next_state_indices = jnp.arange(decoder_output_size)
        self.nr_history_steps = history_length
        self.general_properties = SimpleNamespace(
            action_space_type=action_space_type,
            observation_space_type=observation_space_type,
        )


def load_policy(cfg: dict, config_dir: str | Path | None = None) -> DeploymentPolicy:
    model_type = str(cfg.get("model_type", "auto")).lower()
    model_path = resolve_path(cfg["model_path"], config_dir)

    if model_type == "auto":
        if model_path.suffix == ".pt":
            model_type = "torchscript"
        else:
            model_type = "rlx_flax"

    if model_type in {"torchscript", "torch", "jit"}:
        return TorchScriptPolicy(cfg, config_dir)
    if model_type in {"rlx_flax", "flax", "jax"}:
        return RlxFlaxPolicy(cfg, config_dir)
    raise ValueError(f"Unsupported policy.model_type: {model_type}")
