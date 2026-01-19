import jax
import jax.numpy as jnp
import numpy as np


def load_dones_and_terminations(path: str = "/home/siwei/Documents/repos/rl-x-NCBF/experiments/cbf_pretrain_batch.npz"):
    data = np.load(path)
    dones = data["dones"]
    terminations = data["terminations"]
    indices = data["indices"]
    return dones, terminations, indices


def window_any_done_next_H(dones: jnp.ndarray, terminates: jnp.ndarray):
    H = 10
    cap = jnp.int32(H + 1)

    dones = dones.at[-1, :].set(True)

    def _future_terms_within_H(events: jnp.ndarray, dones_boundary: jnp.ndarray):
        def body(dist_prev, inp):
            ev_t, done_t = inp

            # Reset distance when crossing episode boundary
            dist_prev = jnp.where(done_t, cap, dist_prev)

            dist_t = jnp.where(ev_t, 0, jnp.minimum(dist_prev + 1, cap))
            return dist_t, dist_t

        init = jnp.full((events.shape[1],), cap, dtype=jnp.int32)
        # Scan BACKWARD then reverse output
        _, dists_rev = jax.lax.scan(body, init, (events[::-1], dones_boundary[::-1]))
        dists = dists_rev[::-1]

        any_next_H = jnp.logical_and(dists >= 0, dists <= H)
        return any_next_H, dists

    any_term_next_H, dists_to_next_term = _future_terms_within_H(terminates, dones)
    y = ~any_term_next_H

    def _future_trunc_within_H(terms, truncs):
        def body(dist_prev, inp):
            term_t, trunc_t = inp

            # Reset distance when crossing episode boundary
            dist_prev = jnp.where(trunc_t, cap, dist_prev)

            # if term, set to cap
            dist_prev = jnp.where(term_t, cap, dist_prev)

            dist_t = jnp.where(trunc_t, 0, jnp.minimum(dist_prev + 1, cap))
            return dist_t, dist_t

        init = jnp.full((truncs.shape[1],), 0, dtype=jnp.int32)
        # Scan BACKWARD then reverse output
        _, dists_rev = jax.lax.scan(body, init, (terms[::-1], truncs[::-1]))
        dists = dists_rev[::-1]

        any_next_H = jnp.logical_and(dists >= 0, dists <= H)
        return any_next_H, dists

    trunc_done = dones & (~terminates)
    any_done_next_H, dist_to_next_trunc = _future_trunc_within_H(terminates, trunc_done)
    mask = ~any_done_next_H
    # mask = mask.at[-4:, :].set(False)

    return y, mask, dists_to_next_term, dist_to_next_trunc


dones, terms, indices = load_dones_and_terminations()

print("generated indices: ", indices)
dones = jnp.array(dones[:, 1:2])
terms = jnp.array(terms[:, 1:2])

trunc_done = dones.at[-1, :].set(True) & (~terms)

y, mask, dist, dist_trunc = window_any_done_next_H(dones, terms)

print("terms:\n", terms.T.astype(int))
print("dones:\n", dones.T.astype(int))
print("y:\n", y.T.astype(int))
print("mask:\n", mask.T.astype(int))
print("dist_detect_term:\n", dist.T)
print("dist_detect_trunc:\n", dist_trunc.T)
print("trunc_done:\n", trunc_done.T.astype(int))

neg_valid_mask = (~y) & mask
mean_neg_dist = jnp.sum(dist * neg_valid_mask) / jnp.sum(neg_valid_mask)
print("mean_neg_dist:", mean_neg_dist)
print(dist[neg_valid_mask])

pos_valid_mask = y & mask
mean_pos_dist = jnp.sum(dist * pos_valid_mask) / jnp.sum(pos_valid_mask)
print("mean_pos_dist:", mean_pos_dist)
print(dist[pos_valid_mask])