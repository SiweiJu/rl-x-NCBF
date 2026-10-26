"""k-step prediction error of the world model (T13 evaluation), checkpoint I/O for the world-model bundle."""
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD


def sample_starts(starts, n, seed):
    if len(starts) <= n:
        return np.asarray(starts)
    return np.sort(np.random.default_rng(seed).choice(starts, n, replace=False))


def make_window_predictor(wm, k, teacher_forcing=True):
    @jax.jit
    def predict(bundle, dev, starts):
        b = WD.gather_windows(dev, starts, k)
        step_fn = lambda x, u, xi, uc: wm.step(bundle, x, u, xi, uc)
        preds = W.k_step_rollout_on_window(wm, step_fn, b, teacher_forcing=teacher_forcing)
        err_model = W.state_error(wm.layout, preds[:, -1], b["x_minus_next"][:, -1])
        err_persist = W.state_error(wm.layout, b["x_plus"][:, 0], b["x_minus_next"][:, -1])
        clean = W.window_mask_method_b(b["shock_next"])
        return err_model, err_persist, clean
    return predict


def k_step_errors(wm, bundle, dev, starts, k, max_windows=20000, batch=4096, seed=0, teacher_forcing=True):
    """Error of x_hat-_{t+k} after k autoregressive steps (recorded actions; teacher forcing at shocks by default).
    Returns per-dim squared-error sums for the model and for persistence (x_hat = x+_t), on all windows and on the
    shock-free ones, plus the window counts."""
    starts = sample_starts(starts, max_windows, seed)
    if len(starts) == 0:
        return None
    predict = make_window_predictor(wm, k, teacher_forcing)
    D = wm.layout.state_dim
    acc = {key: np.zeros(D) for key in ("se_model", "se_persist", "se_model_clean")}
    n = n_clean = 0
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        pad = batch - len(s)
        s_dev = jnp.asarray(np.concatenate([s, np.full(pad, s[0])]), jnp.int32)
        em, ep, cl = jax.device_get(predict(bundle, dev, s_dev))
        em, ep, cl = em[:len(s)], ep[:len(s)], cl[:len(s)]
        acc["se_model"] += (em ** 2).sum(0)
        acc["se_persist"] += (ep ** 2).sum(0)
        acc["se_model_clean"] += (em[cl] ** 2).sum(0)
        n += len(s)
        n_clean += int(cl.sum())
    return {**acc, "n": n, "n_clean": n_clean}


def group_rmse(layout, acc):
    """Per-group RMSE (mean over the group's dims of the per-dim MSE, then sqrt), physical units."""
    if acc is None:
        return None
    out = {}
    for name, idx in layout.group_index().items():
        out[name] = {
            "model": float(np.sqrt(acc["se_model"][idx].mean() / acc["n"])),
            "persist": float(np.sqrt(acc["se_persist"][idx].mean() / acc["n"])),
            "model_clean": float(np.sqrt(acc["se_model_clean"][idx].mean() / max(acc["n_clean"], 1))),
        }
    out["_n"], out["_n_clean"] = acc["n"], acc["n_clean"]
    return out


def evaluate(wm, bundle, data, ks=(1, 8, 25, 50), near_failure_window=50, max_windows=20000, seed=0, dev=None):
    """{"overall": {k: group_rmse}, "near_failure": {k: group_rmse}} on `data` (track format)."""
    dev = dev if dev is not None else WD.to_device(data)
    nf = WD.near_failure_rows(data["track_id"], data["done"], data["terminated"], near_failure_window)
    res = {"overall": {}, "near_failure": {}}
    for k in ks:
        starts = WD.valid_window_starts(data["track_id"], data["done"], k)
        res["overall"][k] = group_rmse(wm.layout, k_step_errors(wm, bundle, dev, starts, k, max_windows, seed=seed))
        nf_starts = starts[nf[starts + k - 1]]
        res["near_failure"][k] = group_rmse(wm.layout, k_step_errors(wm, bundle, dev, nf_starts, k, max_windows, seed=seed))
    return res


def summary_scalars(res, prefix="eval"):
    """Flat {name: value} for wandb: mean over groups of model RMSE / persistence RMSE (scale-free)."""
    out = {}
    for split, by_k in res.items():
        for k, g in by_k.items():
            if g is None:
                continue
            # groups with exactly zero model error (last_action, copied from the recorded action) are left out
            ratios = [v["model"] / max(v["persist"], 1e-8) for n, v in g.items() if not n.startswith("_") and v["model"] > 0]
            out[f"{prefix}/{split}/k{k}_rel_rmse"] = float(np.mean(ratios))
            for n, v in g.items():
                if not n.startswith("_"):
                    out[f"{prefix}/{split}/k{k}/{n}"] = v["model"]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Checkpoints (atomic msgpack)
# ---------------------------------------------------------------------------------------------------------------
def save_msgpack(path, tree):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(serialization.msgpack_serialize(serialization.to_state_dict(jax.device_get(tree))))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_msgpack(path, target=None):
    with open(path, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    return serialization.from_state_dict(target, raw) if target is not None else raw


def save_bundle(path, wm, bundle, schema, extra=None):
    """Self-contained world-model file for T14 / P2: bundle + layout schema + WMConfig."""
    meta = {"schema": schema, "wm_config": wm.cfg.__dict__, **(extra or {})}
    save_msgpack(path, {"bundle": bundle, "meta_json": json.dumps(meta)})


def load_bundle(path):
    raw = load_msgpack(path)
    meta = json.loads(raw["meta_json"])
    wm = W.WorldModel(W.WMLayout.from_schema(meta["schema"]), W.WMConfig(**meta["wm_config"]))
    target = wm.init(jax.random.PRNGKey(0))
    bundle = serialization.from_state_dict(target, raw["bundle"])
    return wm, bundle, meta
