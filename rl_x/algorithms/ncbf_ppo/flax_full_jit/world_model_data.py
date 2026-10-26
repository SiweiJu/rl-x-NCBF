"""Data side of the world-model training (T13): window indexing, near-failure masks, device-side window sampling,
a synthetic toy dataset (tests / smokes before the real T12 data exists), and the shard loader.

Canonical in-memory format ("tracks", env-major and flat): every field is an array (S, ...) where the rows of one
env track are contiguous and in time order; `track_id` (S,) names the track. Per row t:
  x_plus        x+_t, the model input state
  u_applied     u~_t, the delayed action that acted during the physics step
  u_chosen      u_t, the policy's (undelayed) action (sets the last_action group)
  xi            xi_t, the DR parameters during the step
  x_minus_next  x-_{t+1}, after physics, before shock / DR / reset  (the target)
  x_plus_next   x+_{t+1}, after shock / DR, before a reset          (teacher forcing, method A)
  shock_next    1 if a shock or a DR resampling happened between x-_{t+1} and x+_{t+1}
  done          1 if the episode ended after this step (x+_{t+1} is not the next row's input)
  terminated    1 if it ended by a termination (failure), 0 for a truncation
Only `world_model_data.load_shards` knows the on-disk T12 layout; adapt it when the real schema changes.
"""
import glob
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

FIELDS = ("x_plus", "u_applied", "u_chosen", "xi", "x_minus_next", "x_plus_next", "shock_next", "done", "terminated")
WINDOW_FIELDS = ("x_plus", "u_applied", "u_chosen", "xi", "x_minus_next", "x_plus_next", "shock_next")


# ---------------------------------------------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------------------------------------------
def valid_window_starts(track_id, done, k):
    """Rows s such that s..s+k-1 lie in one track and no episode ends before the last step (done[s..s+k-2] == 0).
    An episode may end AT the last step: its target x-_{t+k} is recorded before the reset."""
    S = len(track_id)
    if S < k:
        return np.zeros(0, np.int64)
    s = np.arange(S - k + 1)
    same = track_id[s] == track_id[s + k - 1]
    if k > 1:
        c = np.concatenate([[0], np.cumsum(done.astype(np.int64))])
        no_done = (c[s + k - 1] - c[s]) == 0
    else:
        no_done = np.ones_like(same)
    return s[same & no_done].astype(np.int64)


def steps_to_termination(track_id, done, terminated):
    """For every row: steps until the row where the episode ends by a termination (0 at that row); a large value
    if the episode is truncated or runs past the end of the data."""
    S = len(track_id)
    big = np.iinfo(np.int64).max // 4
    idx = np.where(done > 0, np.arange(S), big)
    next_done = np.minimum.accumulate(idx[::-1])[::-1]
    ok = next_done < S
    nd = np.where(ok, next_done, 0)
    ok &= (track_id[nd] == track_id) & (terminated[nd] > 0)
    return np.where(ok, next_done - np.arange(S), big)


def near_failure_rows(track_id, done, terminated, window=50):
    """Rows within the last `window` steps before a termination (the terminating step included)."""
    return steps_to_termination(track_id, done, terminated) < window


def split_window_starts(data, k, near_failure_window=50):
    """(pos_starts, neg_starts): all valid starts, and those whose last step is a near-failure row."""
    starts = valid_window_starts(data["track_id"], data["done"], k)
    nf = near_failure_rows(data["track_id"], data["done"], data["terminated"], near_failure_window)
    return starts, starts[nf[starts + k - 1]]


# ---------------------------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------------------------
def to_device(data, dtype=jnp.float32):
    return {f: jnp.asarray(data[f], dtype) for f in WINDOW_FIELDS}


def gather_windows(dev, starts, k):
    """dev: device dict of (S, ...) arrays; starts (B,) -> dict of (B, k, ...)."""
    rows = starts[:, None] + jnp.arange(k)[None, :]
    return {f: dev[f][rows] for f in WINDOW_FIELDS}


def make_window_sampler(pos_starts, neg_starts, k, batch_size, neg_ratio):
    """Returns sample(key, dev) -> batch, mixing neg_ratio of the windows from the near-failure set (if any)."""
    pos = jnp.asarray(pos_starts, jnp.int32)
    neg = jnp.asarray(neg_starts if len(neg_starts) else pos_starts, jnp.int32)
    n_neg = int(round(batch_size * neg_ratio)) if len(neg_starts) else 0
    n_pos = batch_size - n_neg

    def sample(key, dev):
        kp, kn = jax.random.split(key)
        s = jax.random.choice(kp, pos, (n_pos,))
        if n_neg:
            s = jnp.concatenate([s, jax.random.choice(kn, neg, (n_neg,))])
        return gather_windows(dev, s, k)

    return sample


def transition_rows_for_norm(data, max_rows=2_000_000, seed=0):
    """Non-shock rows (x-_{t+1} reachable by physics only) for the normalization fit."""
    rows = np.nonzero(data["shock_next"] < 0.5)[0]
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Shard loader (real T12 data)
# ---------------------------------------------------------------------------------------------------------------
def _to_tracks(arrs):
    """Time-major (T, N, ...) shard -> env-major flat (N*T, ...)."""
    return {f: np.swapaxes(a, 0, 1).reshape((-1,) + a.shape[2:]) for f, a in arrs.items()}


def load_shards(paths, field_map=None, time_major=True, track_offset=0):
    """Load .npz shards into the canonical track format. field_map maps canonical names to shard keys.
    Each shard is (T, N, ...) time-major (time_major=True) and its envs become new tracks.
    Returns (data, source) where source[i] is the shard's "source" entry (checkpoint id) if present."""
    field_map = field_map or {}
    out = {f: [] for f in FIELDS + ("track_id",)}
    sources = []
    nxt = track_offset
    for p in paths:
        z = np.load(p, allow_pickle=False)
        arrs = {f: np.asarray(z[field_map.get(f, f)]) for f in FIELDS}
        if time_major:
            T, N = arrs["done"].shape[:2]
            arrs = _to_tracks(arrs)
            tid = np.repeat(np.arange(nxt, nxt + N), T)
            nxt += N
        else:
            tid = np.asarray(z[field_map.get("track_id", "track_id")]) + nxt
            nxt = tid.max() + 1
        for f in FIELDS:
            out[f].append(arrs[f])
        out["track_id"].append(tid)
        sources.append(str(z["source"]) if "source" in z.files else os.path.basename(p))
    data = {f: np.concatenate(v) for f, v in out.items()}
    for f in ("shock_next", "done", "terminated"):
        data[f] = data[f].astype(np.float32).reshape(-1)
    return data, sources


def concat_data(parts):
    off, out = 0, []
    for d in parts:
        d = dict(d)
        d["track_id"] = d["track_id"] + off
        off = int(d["track_id"].max()) + 1
        out.append(d)
    return {f: np.concatenate([d[f] for d in out]) for f in out[0]}


# ---------------------------------------------------------------------------------------------------------------
# Synthetic toy data (tests and smoke runs until the real schema / data exist)
# ---------------------------------------------------------------------------------------------------------------
def synthetic_schema(action_dim=3, xi_dim=2):
    A = action_dim
    names = [("q", A), ("qd", A), ("last_action", A), ("height", 1), ("base_lin_vel", 2), ("yaw_rate", 1),
             ("base_xy", 2), ("base_yaw", 1)]
    groups, s = {}, 0
    for n, d in names:
        groups[n] = [s, s + d]
        s += d
    return {"state_dim": s, "action_dim": A, "xi_dim": xi_dim, "groups": groups, "synthetic": True}


def synthetic_dataset(n_envs=64, T=400, action_dim=3, xi_dim=2, shock_prob=0.01, dr_prob=0.002,
                      max_episode_len=300, delay_steps=1, seed=0):
    """A toy legged-ish system with the special groups (last_action, base_xy, base_yaw), shocks, DR resampling,
    an action delay, height terminations and truncations. Returns (data in track format, schema)."""
    rng = np.random.default_rng(seed)
    sch = synthetic_schema(action_dim, xi_dim)
    g = {n: np.arange(a, b) for n, (a, b) in sch["groups"].items()}
    A, D, dt = action_dim, sch["state_dim"], 0.05

    def new_xi(n):
        return np.stack([rng.uniform(0.7, 1.3, n), rng.uniform(-0.2, 0.2, n)] + [rng.uniform(-1, 1, n)] * (xi_dim - 2), 1)

    def reset(n):
        x = np.zeros((n, D))
        x[:, g["q"]] = rng.normal(0, 0.1, (n, A))
        x[:, g["height"]] = 0.35
        x[:, g["base_xy"]] = rng.normal(0, 1, (n, 2))
        x[:, g["base_yaw"]] = rng.uniform(-np.pi, np.pi, (n, 1))
        return x

    def physics(x, ut, xi):
        y = x.copy()
        q, qd = x[:, g["q"]], x[:, g["qd"]]
        acc = xi[:, :1] * 20.0 * (ut - q) - 2.0 * qd + xi[:, 1:2]
        qd2 = qd + dt * acc
        y[:, g["qd"]] = qd2
        y[:, g["q"]] = q + dt * qd2
        h = x[:, g["height"][0]]
        y[:, g["height"][0]] = h + dt * (-1.0 * (h - 0.35) + 0.2 * np.tanh(q.sum(1)) * (h - 0.15)
                                         - 0.8 * np.maximum(np.abs(q).mean(1) - 0.3, 0.0))
        v = np.tanh(q[:, :2]) if A >= 2 else np.stack([np.tanh(q[:, 0]), 0 * q[:, 0]], 1)
        y[:, g["base_lin_vel"]] = v
        yaw = x[:, g["base_yaw"][0]]
        y[:, g["yaw_rate"][0]] = 0.5 * np.tanh(qd2[:, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        y[:, g["base_xy"][0]] = x[:, g["base_xy"][0]] + dt * (c * v[:, 0] - s * v[:, 1])
        y[:, g["base_xy"][1]] = x[:, g["base_xy"][1]] + dt * (s * v[:, 0] + c * v[:, 1])
        y[:, g["base_yaw"][0]] = (yaw + dt * y[:, g["yaw_rate"][0]] + np.pi) % (2 * np.pi) - np.pi
        return y

    rec = {f: np.zeros((T, n_envs) + shp) for f, shp in
           [("x_plus", (D,)), ("u_applied", (A,)), ("u_chosen", (A,)), ("xi", (xi_dim,)), ("x_minus_next", (D,)),
            ("x_plus_next", (D,)), ("shock_next", ()), ("done", ()), ("terminated", ())]}
    x, xi = reset(n_envs), new_xi(n_envs)
    hist = np.zeros((n_envs, delay_steps + 1, A))
    ep_len = np.zeros(n_envs, int)
    drive = rng.normal(0, 0.6, (n_envs, A))
    for t in range(T):
        u = -0.8 * x[:, g["q"]] + drive + rng.normal(0, 0.3, (n_envs, A))
        drive = 0.97 * drive + rng.normal(0, 0.15, (n_envs, A))
        hist = np.roll(hist, -1, axis=1)
        hist[:, -1] = u
        ut = hist[:, -1 - delay_steps]
        xm = physics(x, ut, xi)
        xm[:, g["last_action"]] = u
        xp = xm.copy()
        shock = rng.uniform(size=n_envs) < shock_prob
        xp[np.ix_(shock, g["qd"])] += rng.normal(0, 2.0, (shock.sum(), A))
        xp[np.ix_(shock, g["base_lin_vel"])] += rng.normal(0, 0.5, (shock.sum(), 2))
        dr = rng.uniform(size=n_envs) < dr_prob
        xi_next = np.where(dr[:, None], new_xi(n_envs), xi)
        ep_len += 1
        term = xm[:, g["height"][0]] < 0.2
        trunc = ep_len >= max_episode_len
        done = term | trunc
        for f, v in [("x_plus", x), ("u_applied", ut), ("u_chosen", u), ("xi", xi), ("x_minus_next", xm),
                     ("x_plus_next", xp), ("shock_next", (shock | dr).astype(float)), ("done", done.astype(float)),
                     ("terminated", term.astype(float))]:
            rec[f][t] = v
        r = reset(n_envs)
        x = np.where(done[:, None], r, xp)
        xi = np.where(done[:, None], new_xi(n_envs), xi_next)
        hist = np.where(done[:, None, None], 0.0, hist)
        drive = np.where(done[:, None], rng.normal(0, 0.6, (n_envs, A)), drive)
        ep_len = np.where(done, 0, ep_len)
    data = _to_tracks({f: v.astype(np.float32) for f, v in rec.items()})
    for f in ("shock_next", "done", "terminated"):
        data[f] = data[f].reshape(-1)
    data["track_id"] = np.repeat(np.arange(n_envs), T)
    return data, sch


def synthetic_delay_step(delay_steps):
    """Delay buffer matching synthetic_dataset: buffer (N, delay+1, A), newest last."""
    def step(buf, u):
        buf = jnp.concatenate([buf[:, 1:], u[:, None]], axis=1)
        return buf[:, -1 - delay_steps], buf
    return step


def load_schema(path):
    with open(path) as f:
        return json.load(f)


def find_shards(pattern):
    return sorted(glob.glob(pattern))
