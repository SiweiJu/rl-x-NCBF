"""Pure functions for the safety-critic training data: labels, critic inputs, temporal pairs and losses.

They are used by ppo.py inside the jitted training loop and are kept free of `self` so the unit tests
(tests/test_p0_fixes.py) can check them directly.

Time indexing of one rollout of T steps (all arrays are [T, N, ...]):
    states[t]       = s_t, the observation the policy and the shield saw at step t
    env_actions[t]  = a_t, the executed (shielded, processed) action at step t
    next_states[t]  = s_{t+1} before any episode reset (env_state.actual_next_observation)
    dones[t]        = the step t -> t+1 ended the episode (termination or truncation)
The label y_t covers s_{t+1..t+H}, i.e. it is the outcome of executing a_t at s_t, so the critic is trained
on h(s_t, a_t) at the same index t.
"""
import jax
import jax.numpy as jnp


CRITIC_ACTION_INDICES = ("current", "previous")
TERMINAL_CRITIC_INPUTS = ("pre_reset", "zeros")
TEMPORAL_MODES = ("adjacent", "legacy_raw_vs_executed")
TEMPORAL_PAIRS = ("safe_safe", "safe_safe_and_unsafe")
FIRST_STEP_PREV_VALUES = ("gamma_c", "input")
LABEL_SOURCES = ("env_targets", "termination_window")


def window_any_done_next_H(dones, terminates, H):
    """
    dones, terminates: [T, N] bool
    Returns:
      y:    [T, N] bool   True if NO terminate in [t, t+H], else False
      mask: [T, N] bool   True if NO (done & ~terminate) in [t, t+H], else False
      dists_to_next_term: [T, N] int32  distance to next terminate at/after t within same episode
                           0 means terminate at t; cap (=H+10) means none within the episode
    """
    cap = jnp.int32(H + 10)
    dones = dones.at[-1, :].set(True)

    def _future_terms_within_H(events, dones_boundary):
        def body(dist_prev, inp):
            ev_t, done_t = inp
            # Reset distance when crossing episode boundary
            dist_prev = jnp.where(done_t, cap, dist_prev)
            dist_t = jnp.where(ev_t, 0, jnp.minimum(dist_prev + 1, cap))
            return dist_t, dist_t

        init = jnp.full((events.shape[1],), cap, dtype=jnp.int32)
        _, dists_rev = jax.lax.scan(body, init, (events[::-1], dones_boundary[::-1]))
        dists = dists_rev[::-1]
        any_next_H = jnp.logical_and(dists >= 0, dists <= H)
        return any_next_H, dists

    any_term_next_H, dists_to_next_term = _future_terms_within_H(terminates, dones)
    y = ~any_term_next_H

    def _future_trunc_within_H(terms, truncs):
        # Mask out samples whose window hits a truncation (time limit or the end of the rollout).
        def body(dist_prev, inp):
            term_t, trunc_t = inp
            dist_prev = jnp.where(trunc_t, cap, dist_prev)
            dist_prev = jnp.where(term_t, cap, dist_prev)
            dist_t = jnp.where(trunc_t, 0, jnp.minimum(dist_prev + 1, cap))
            return dist_t, dist_t

        init = jnp.full((truncs.shape[1],), cap, dtype=jnp.int32)
        _, dists_rev = jax.lax.scan(body, init, (terms[::-1], truncs[::-1]))
        dists = dists_rev[::-1]
        any_next_H = jnp.logical_and(dists >= 0, dists <= H)
        return any_next_H, dists

    trunc_done = dones & (~terminates)
    any_done_next_H, _ = _future_trunc_within_H(terminates, trunc_done)
    mask = ~any_done_next_H
    return y, mask, dists_to_next_term


def receding_min_target(future_states, dones, terminates, H, target_indices):
    """
    future_states: [T, N, D] actual next observations s_{t+1} of each transition
    Returns target [T, N, K] (min of the target dims over s_{t+1..t+H} within the episode), y, mask, dists.
    """
    idx = jnp.asarray(target_indices)
    vals = future_states[..., idx]
    y, mask, dists_to_next_term = window_any_done_next_H(dones, terminates, H)

    T, N, K = vals.shape
    offsets = jnp.arange(H, dtype=jnp.int32)
    time_idx = jnp.arange(T, dtype=jnp.int32)[:, None] + offsets[None, :]
    pad = jnp.full((H - 1, N, K), jnp.inf, dtype=vals.dtype)
    vals_pad = jnp.concatenate([vals, pad], axis=0)
    windows = vals_pad[time_idx]  # [T, H, N, K]
    # Include the terminal state if termination occurs within the horizon, nothing after it.
    eff_len = jnp.where(dists_to_next_term < H, dists_to_next_term + 1, H).astype(jnp.int32)
    valid = offsets[None, :, None] < eff_len[:, None, :]
    windows = jnp.where(valid[..., None], windows, jnp.inf)
    target = jnp.min(windows, axis=1)
    return target, y, mask, dists_to_next_term


def termination_window_labels(dones, terminates, H, safe_fall_H=-1):
    """Paper (ncbf-q) labels: one head, y_t = no termination at transitions t..t+H (window_any_done_next_H).

    safe_fall_H >= 0 applies the SafeFall rule of branch ncbf-q-safefall (804c7a5): y_t = dist > H + 1 (no
    termination within H + 1 transitions), and samples with safe_fall_H < dist <= H are masked out, so only
    failures within safe_fall_H steps are negatives. (A sample with dist == H + 1 stays a negative, as in that code.)
    Returns target [T, N, 1] float, y [T, N] bool, mask [T, N] bool, dists [T, N] int32.
    """
    y, mask, dists = window_any_done_next_H(dones, terminates, H)
    if safe_fall_H is not None and safe_fall_H >= 0:
        y = dists > H + 1
        not_defined = (dists <= H) & (dists > safe_fall_H)
        mask = mask & ~not_defined
    return y.astype(jnp.float32)[..., None], y, mask, dists


def critic_training_inputs(states, env_actions, last_actions, dones, action_index="current", terminal_input="pre_reset"):
    """(state, action) the safety critic is trained on for each transition (any leading batch shape).

    action_index "current" pairs s_t with the executed a_t (B2 fix); "previous" is the legacy (s_t, a_{t-1}).
    terminal_input "pre_reset" keeps the real (s_t, a_t) of the step that ends an episode (F1 fix); "zeros"
    reproduces the paper code, which read the inputs from the env state after the auto-reset and therefore
    trained the episode-ending step on (0, 0).
    """
    if action_index not in CRITIC_ACTION_INDICES:
        raise ValueError(f"critic_action_index must be one of {CRITIC_ACTION_INDICES}")
    if terminal_input not in TERMINAL_CRITIC_INPUTS:
        raise ValueError(f"terminal_critic_input must be one of {TERMINAL_CRITIC_INPUTS}")
    actions = env_actions if action_index == "current" else last_actions
    if terminal_input == "zeros":
        dones = jnp.asarray(dones, dtype=bool)
        states = jnp.where(dones[..., None], jnp.zeros_like(states), states)
        actions = jnp.where(dones[..., None], jnp.zeros_like(actions), actions)
    return states, actions


def successor_pairs(env_actions, y_target, masks, dones):
    """Adjacent pairs (s_t, a_t) -> (s_{t+1}, a_{t+1}) within one episode, for the temporal loss.

    The successor state is next_states[t] (= states[t+1] when the episode continues); its action is
    env_actions[t+1], the same executed-action scale as a_t. A pair is valid only if the episode does not end at
    t, both labels are unmasked and t+1 is inside the rollout.
    """
    T = env_actions.shape[0]
    next_env_actions = jnp.concatenate([env_actions[1:], env_actions[-1:]], axis=0)
    next_y_target = jnp.concatenate([y_target[1:], y_target[-1:]], axis=0)
    next_masks = jnp.concatenate([masks[1:], jnp.zeros_like(masks[-1:])], axis=0)
    not_last_step = (jnp.arange(T) < T - 1)[:, None]
    temporal_valid = not_last_step & (~dones) & masks & next_masks
    return next_env_actions, next_y_target, temporal_valid


def cbf_residual(h_next, h_cur, gamma_c, eta, gamma_c_scaled_by_eta=True):
    """Paper eq. (4)/(5): h(x_{t+1}) - h(x_t) + eta (h(x_t) - gamma_c) = h_next - (1-eta) h_cur - eta gamma_c.
    gamma_c_scaled_by_eta=False is the legacy "- gamma_c" form."""
    gamma_term = eta * gamma_c if gamma_c_scaled_by_eta else gamma_c
    return h_next - (1.0 - eta) * h_cur - gamma_term


def temporal_consistency_loss(h_cur, h_next, y_now, y_next, valid, gamma_c, eta,
                              gamma_c_scaled_by_eta=True, pairs="safe_safe"):
    """DT-CBF temporal loss on adjacent pairs, ReLU(-residual) (paper eq. 4-6).

    h_cur, h_next: [E, B, K] safe probabilities of h(s_t, a_t) and h(s_{t+1}, a_{t+1})
    y_now, y_next: [B, K] labels; valid: [B] pair is inside one episode.
    pairs "safe_safe" applies it to safe->safe pairs only (default); "safe_safe_and_unsafe" also adds the
    previous multistep term that asks h to decrease on unsafe samples.
    Returns the sum over ensemble members of each member's mean over the selected pairs (like the
    classification loss in ppo.py).
    """
    if pairs not in TEMPORAL_PAIRS:
        raise ValueError(f"temporal_pairs must be one of {TEMPORAL_PAIRS}")
    valid = valid.astype(h_cur.dtype)[None, :, None]
    y_now = y_now.astype(h_cur.dtype)[None]
    y_next = y_next.astype(h_cur.dtype)[None]
    safe_pair = valid * (y_now > 0.5) * (y_next > 0.5)
    residual = cbf_residual(h_next, h_cur, gamma_c, eta, gamma_c_scaled_by_eta)
    loss = jnp.sum(safe_pair * jax.nn.relu(-residual)) / (jnp.sum(safe_pair) + 1e-8)
    if pairs == "safe_safe_and_unsafe":
        unsafe_pair = valid * (y_now < 0.5)
        loss_unsafe = jax.nn.relu(h_next - h_cur - eta * (gamma_c - h_cur))
        loss = loss + jnp.sum(unsafe_pair * loss_unsafe) / (jnp.sum(unsafe_pair) + 1e-8)
    return loss


def legacy_raw_vs_executed_temporal_loss(h_exec, h_raw, y_now, masks, indices_to_term, H):
    """The paper code's "CBF" term (F2, kept for reproduction): it compares h(s_t, a_t^exec) with h(s_t, a_t^raw)
    at the SAME state, without the eta term. Unsafe samples: ReLU(h_raw - h_exec); safe samples that do not
    terminate within H steps: ReLU(h_exec - h_raw).

    h_exec, h_raw: [E, B, K]; y_now: [B, K]; masks, indices_to_term: [B].
    """
    m = masks.astype(h_exec.dtype)[None, :, None]
    y = y_now.astype(h_exec.dtype)[None]
    far = (indices_to_term > H).astype(h_exec.dtype)[None, :, None]
    decrease = h_raw - h_exec
    neg_mask = (1.0 - y) * m
    pos_mask = y * m * far
    neg_loss = jnp.sum(neg_mask * jax.nn.relu(decrease)) / (jnp.sum(neg_mask) + 1e-8)
    pos_loss = jnp.sum(pos_mask * jax.nn.relu(-decrease)) / (jnp.sum(pos_mask) + 1e-8)
    return neg_loss + pos_loss


def episode_first_step(env_state):
    """True for envs whose next step is the first step of an episode (no previous critic value exists)."""
    return env_state.info_episode_store["episode_step"] == 0
