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
# Compact format (real data): x_plus_next is not stored (inside a valid window it equals x_plus of the next row),
# x_minus_next only on the rows where it differs from x_plus[s+1] (shock / DR / episode end): xm_rows (sorted row ids)
# + xm_vals; xi is xi_table[xi_idx]. Optional start_ok (bool, S): rows that may start a window.


# ---------------------------------------------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------------------------------------------
def valid_window_starts(track_id, done, k, start_ok=None):
    """Rows s such that s..s+k-1 lie in one track and no episode ends before the last step (done[s..s+k-2] == 0),
    and start_ok[s] if given. An episode may end AT the last step: its target x-_{t+k} is recorded before the reset."""
    S = len(track_id)
    if S < k:
        return np.zeros(0, np.int64)
    s = np.arange(S - k + 1)
    ok = track_id[s] == track_id[s + k - 1]
    if k > 1:
        c = np.concatenate([[0], np.cumsum(done.astype(np.int64))])
        ok &= (c[s + k - 1] - c[s]) == 0
    if start_ok is not None:
        ok &= start_ok[s].astype(bool)
    return s[ok].astype(np.int64)


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


def data_window_starts(data, k):
    return valid_window_starts(data["track_id"], data["done"], k, data.get("start_ok"))


def split_window_starts(data, k, near_failure_window=50):
    """(pos_starts, neg_starts): all valid starts, and those whose last step is a near-failure row."""
    starts = data_window_starts(data, k)
    nf = near_failure_rows(data["track_id"], data["done"], data["terminated"], near_failure_window)
    return starts, starts[nf[starts + k - 1]]


def n_rows(data):
    return len(data["done"])


def rows_arrays(data, rows):
    """Numpy (x_plus, u_applied, u_chosen, xi, x_minus_next) at the given rows, for either format."""
    rows = np.asarray(rows)
    xi = data["xi"][rows] if "xi" in data else data["xi_table"][data["xi_idx"][rows]]
    if "x_minus_next" in data:
        xm = data["x_minus_next"][rows]
    else:
        nxt = np.minimum(rows + 1, n_rows(data) - 1)
        xm = data["x_plus"][nxt].copy()
        pos = np.searchsorted(data["xm_rows"], rows)
        pos_c = np.minimum(pos, max(len(data["xm_rows"]) - 1, 0))
        hit = (len(data["xm_rows"]) > 0) & (data["xm_rows"][pos_c] == rows)
        xm[hit] = data["xm_vals"][pos_c[hit]]
    return data["x_plus"][rows], data["u_applied"][rows], data["u_chosen"][rows], xi, xm


# ---------------------------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------------------------
def to_device(data, dtype=jnp.float32):
    """Device copy of what gather_windows needs (either format)."""
    dev = {f: jnp.asarray(data[f], dtype) for f in ("x_plus", "u_applied", "u_chosen", "shock_next")}
    if "x_plus_next" in data:
        dev["x_plus_next"] = jnp.asarray(data["x_plus_next"], dtype)
    if "x_minus_next" in data:
        dev["x_minus_next"] = jnp.asarray(data["x_minus_next"], dtype)
    else:   # dense pointer row -> index into xm_vals (-1: x_plus of the next row)
        ptr = np.full(n_rows(data), -1, np.int32)
        ptr[data["xm_rows"]] = np.arange(len(data["xm_rows"]), dtype=np.int32)
        dev["xm_ptr"] = jnp.asarray(ptr)
        dev["xm_vals"] = jnp.asarray(data["xm_vals"] if len(data["xm_vals"]) else np.zeros((1, data["x_plus"].shape[1])), dtype)
    if "xi" in data:
        dev["xi"] = jnp.asarray(data["xi"], dtype)
    else:
        dev["xi_idx"] = jnp.asarray(data["xi_idx"], jnp.int32)
        dev["xi_table"] = jnp.asarray(data["xi_table"], dtype)
    return dev


def gather_windows(dev, starts, k):
    """dev: device dict from to_device; starts (B,) -> dict of (B, k, ...) with all WINDOW_FIELDS."""
    rows = starts[:, None] + jnp.arange(k)[None, :]
    S = dev["x_plus"].shape[0]
    nxt = jnp.minimum(rows + 1, S - 1)
    out = {f: dev[f][rows] for f in ("x_plus", "u_applied", "u_chosen", "shock_next")}
    out["x_plus_next"] = dev["x_plus_next"][rows] if "x_plus_next" in dev else dev["x_plus"][nxt]
    if "x_minus_next" in dev:
        out["x_minus_next"] = dev["x_minus_next"][rows]
    else:
        ptr = dev["xm_ptr"][rows]
        out["x_minus_next"] = jnp.where((ptr >= 0)[..., None], dev["xm_vals"][jnp.maximum(ptr, 0)], dev["x_plus"][nxt])
    out["xi"] = dev["xi"][rows] if "xi" in dev else dev["xi_table"][dev["xi_idx"][rows]]
    return out


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


def transition_rows_for_norm(data, max_rows=1_000_000, seed=0):
    """Rows for the normalization fit: no shock (x-_{t+1} reachable by physics only) and start_ok."""
    ok = data["shock_next"] < 0.5
    if "start_ok" in data:
        ok &= data["start_ok"].astype(bool)
    rows = np.nonzero(ok)[0]
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Loader for the T12 shards (wm-data schema pncbf-p1-wmdata/v1: one directory per shard, <field>.npy time-major
# [T, N, ...], meta.json with checkpoint / method / split)
# ---------------------------------------------------------------------------------------------------------------
def find_shard_dirs(pattern):
    """Shard directories matching a glob (a directory counts if it holds x_plus.npy)."""
    out = []
    for p in sorted(glob.glob(pattern)):
        if os.path.isfile(os.path.join(p, "x_plus.npy")):
            out.append(p)
        elif os.path.isdir(p):
            out += sorted(os.path.dirname(q) for q in glob.glob(os.path.join(p, "**", "x_plus.npy"), recursive=True))
    return sorted(set(out))


def shard_meta(path):
    f = os.path.join(path, "meta.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def _tm(a, envs):
    """[T, N, ...] mmap -> env-major flat [n*T, ...] for the selected envs."""
    a = np.asarray(a[:, envs])
    return np.ascontiguousarray(np.swapaxes(a, 0, 1)).reshape((-1,) + a.shape[2:])


def load_wmdata_shard(path, max_envs=0, seed=0, check=False):
    """One shard in the compact track format. max_envs > 0 keeps a random subset of envs."""
    ld = lambda f: np.load(os.path.join(path, f + ".npy"), mmap_mode="r")
    done_tm = ld("done")
    T, N = done_tm.shape
    envs = np.arange(N)
    if max_envs and max_envs < N:
        envs = np.sort(np.random.default_rng(seed).choice(N, max_envs, replace=False))
    d = {"x_plus": _tm(ld("x_plus"), envs).astype(np.float32),
         "u_applied": _tm(ld("u_applied"), envs).astype(np.float32),
         "u_chosen": _tm(ld("u_cmd"), envs).astype(np.float32),
         "xi_idx": _tm(ld("xi_idx"), envs).astype(np.int32),
         "done": _tm(done_tm, envs).astype(np.float32),
         "terminated": _tm(ld("terminated"), envs).astype(np.float32),
         "shock_next": (_tm(ld("perturbed_next"), envs) | _tm(ld("dr_resampled_next"), envs)).astype(np.float32),
         "start_ok": _tm(ld("episode_step"), envs) != 0,      # the first transition of an episode has stale IMU values
         "track_id": np.repeat(np.arange(len(envs)), T)}
    xm = _tm(ld("x_minus_next"), envs).astype(np.float32)
    S = len(d["done"])
    last_of_track = np.zeros(S, bool)
    last_of_track[T - 1::T] = True
    keep = (d["shock_next"] > 0) | (d["done"] > 0) | last_of_track
    nxt = np.minimum(np.arange(S) + 1, S - 1)
    differs = np.any(xm != d["x_plus"][nxt], axis=1)
    stats = {"rows": S, "rows_kept_x_minus": int(keep.sum()),
             "x_minus_ne_next_x_plus_outside_kept": int((differs & ~keep).sum())}
    keep |= differs     # never lose information: store every row where the shortcut would be wrong
    d["xm_rows"] = np.nonzero(keep)[0].astype(np.int64)
    d["xm_vals"] = xm[keep]
    d["xi_table"] = np.asarray(ld("xi_table"), np.float32)
    if check:
        xpn = _tm(ld("x_plus_next"), envs)
        inner = (d["done"] < 0.5) & ~last_of_track
        stats["x_plus_next_ne_next_x_plus"] = int(np.any(xpn[inner] != d["x_plus"][nxt][inner], axis=1).sum())
    return d, stats


def _to_tracks(arrs):
    """Time-major (T, N, ...) arrays -> env-major flat (N*T, ...)."""
    return {f: np.swapaxes(a, 0, 1).reshape((-1,) + a.shape[2:]) for f, a in arrs.items()}


def concat_data(parts):
    """Concatenate track-format datasets (renumbering tracks; offsetting xi_idx / xm_rows in the compact format)."""
    out = {}
    row_off = track_off = xi_off = 0
    acc = {}
    for d in parts:
        d = dict(d)
        d["track_id"] = d["track_id"] + track_off
        if "xm_rows" in d:
            d["xm_rows"] = d["xm_rows"] + row_off
        if "xi_idx" in d:
            d["xi_idx"] = d["xi_idx"] + xi_off
            xi_off += len(d["xi_table"])
        track_off = int(d["track_id"].max()) + 1
        row_off += len(d["done"])
        for f, v in d.items():
            acc.setdefault(f, []).append(v)
    for f, v in acc.items():
        out[f] = np.concatenate(v)
    return out


def select_xi_dims(data, idx):
    """Keep only xi dims `idx` (in xi_table or xi)."""
    d = dict(data)
    if "xi_table" in d:
        d["xi_table"] = d["xi_table"][:, idx]
    else:
        d["xi"] = d["xi"][:, idx]
    return d


def active_xi_dims(data, tol=1e-6):
    """xi dims that vary over the rows of `data` (std > tol, weighted by how often each table row is used)."""
    if "xi_table" in data:
        w = np.bincount(data["xi_idx"], minlength=len(data["xi_table"])).astype(np.float64)
        t = data["xi_table"].astype(np.float64)
        m = (w[:, None] * t).sum(0) / w.sum()
        var = (w[:, None] * (t - m) ** 2).sum(0) / w.sum()
    else:
        var = data["xi"].astype(np.float64).var(0)
    return np.nonzero(np.sqrt(var) > tol)[0]


def wm_schema_from_wmdata(schema, xi_active):
    """The world model's layout schema from the wm-data schema (pncbf-p1-wmdata/v1) and the kept xi dims."""
    return {"state_dim": int(schema["state"]["dim"]), "action_dim": int(schema["action_dim"]),
            "xi_dim": int(len(xi_active)), "groups": schema["state"]["groups"],
            "xi_active": [int(i) for i in xi_active], "xi_full_dim": int(schema["xi"]["dim"]),
            "source_schema": schema.get("schema")}


def load_wmdata(pattern, heldout_patterns=(), max_envs_per_shard=0, seed=0, which=("train", "heldout"), log=print):
    """Load the shard directories matching `pattern`, split into train / heldout by meta.json "split" (anything but
    "train" is held out); shards without a split are held out if their path contains one of heldout_patterns.
    Returns {name: (data or None, [per-shard info])}."""
    dirs = find_shard_dirs(pattern)
    if not dirs:
        raise FileNotFoundError(f"no shard directories (with x_plus.npy) match {pattern}")
    groups = {"train": [], "heldout": []}
    for d in dirs:
        m = shard_meta(d)
        held = (m["split"] != "train") if "split" in m else any(q and q in d for q in heldout_patterns)
        groups["heldout" if held else "train"].append((d, m))
    out = {}
    for name in which:
        info, data = [], None
        sizes = []
        for d, m in groups[name]:
            T, N = np.load(os.path.join(d, "done.npy"), mmap_mode="r").shape
            sizes.append(T * (min(N, max_envs_per_shard) if max_envs_per_shard else N))
        total, row = int(sum(sizes)), 0
        sparse = {"xm_rows": [], "xm_vals": [], "xi_table": []}
        track_off = xi_off = 0
        for i, (d, m) in enumerate(groups[name]):
            part, st = load_wmdata_shard(d, max_envs_per_shard, seed + i)
            n = n_rows(part)
            if data is None:   # preallocate the dense fields once (no 2x peak from a final concatenate)
                data = {f: np.empty((total,) + part[f].shape[1:], part[f].dtype) for f in part if f not in sparse}
            for f in data:
                v = part[f]
                if f == "track_id":
                    v = v + track_off
                elif f == "xi_idx":
                    v = v + xi_off
                data[f][row:row + n] = v
            sparse["xm_rows"].append(part["xm_rows"] + row)
            sparse["xm_vals"].append(part["xm_vals"])
            sparse["xi_table"].append(part["xi_table"])
            track_off += int(part["track_id"].max()) + 1
            xi_off += len(part["xi_table"])
            row += n
            del part
            info.append({"path": d, **{k: m[k] for k in ("checkpoint", "checkpoint_label", "method", "train_seed",
                                                         "split", "setting", "p") if k in m}, **st})
            log(f"loaded {name} shard {i + 1}/{len(groups[name])} {d}: {st}")
        if data is not None:
            data.update({f: np.concatenate(v) for f, v in sparse.items()})
        out[name] = (data, info)
    return out


def load_shards(paths, field_map=None, time_major=True, track_offset=0):
    """Legacy loader for single-file .npz shards with canonical field names (tests / synthetic exports)."""
    field_map = field_map or {}
    parts, sources = [], []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        arrs = {f: np.asarray(z[field_map.get(f, f)]) for f in FIELDS}
        T, N = arrs["done"].shape[:2]
        d = {f: np.swapaxes(a, 0, 1).reshape((-1,) + a.shape[2:]) for f, a in arrs.items()}
        for f in ("shock_next", "done", "terminated"):
            d[f] = d[f].astype(np.float32).reshape(-1)
        d["track_id"] = np.repeat(np.arange(N), T)
        parts.append(d)
        sources.append(str(z["source"]) if "source" in z.files else os.path.basename(p))
    return concat_data(parts), sources


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
