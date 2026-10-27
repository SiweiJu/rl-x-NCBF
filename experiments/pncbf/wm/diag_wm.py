"""T13 diagnostics: where does the k-step error come from? Splits the per-window squared error at the final step by
the category of the target row (episode step, shock before it, episode end, large roll/pitch) and prints each
category's share of windows and of the MSE per state group (model and persistence).

  python experiments/pncbf/wm/diag_wm.py --wm RUN/wm_final.msgpack --data <shard root> --split train --ks 1,8
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="heldout")
    p.add_argument("--ks", default="1,8")
    p.add_argument("--max_windows", type=int, default=50000)
    p.add_argument("--max_envs_per_shard", type=int, default=0)
    p.add_argument("--groups", default="imu_height,roll_pitch,base_yaw")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    wm, bundle, meta = WE.load_bundle(a.wm)
    data, _ = WD.load_wmdata(a.data, (), a.max_envs_per_shard, which=(a.split,))[a.split]
    data = WD.select_xi_dims(data, np.asarray(meta["schema"]["xi_active"]))
    dev = WD.to_device(data)
    gi = wm.layout.group_index()
    rp = gi.get("roll_pitch")
    out = {}
    for k in [int(x) for x in a.ks.split(",")]:
        starts = WE.sample_starts(WD.data_window_starts(data, k), a.max_windows, 0)
        pred = WE.make_window_predictor(wm, k)
        em, ep = [], []
        for i in range(0, len(starts), 4096):
            s = starts[i:i + 4096]
            r = pred(bundle, dev, jnp.asarray(s, jnp.int32))
            em.append(np.asarray(r[0]))
            ep.append(np.asarray(r[1]))
        em, ep = np.concatenate(em), np.concatenate(ep)
        last = starts + k - 1
        es = data["episode_step"][last].astype(int)
        xp_last = WD.rows_arrays(data, last)[0]
        cats = {
            "ep_step_1_7": (es >= 1) & (es <= 7),
            "ep_step_8_49": (es >= 8) & (es <= 49),
            "ep_step_50+": es >= 50,
            "target_row_done": data["done"][last] > 0,
            "target_row_terminated": data["terminated"][last] > 0,
            "shock_inside": np.array([data["shock_next"][s:s + k - 1].any() if k > 1 else False for s in starts]),
            "shock_after_target": data["shock_next"][last] > 0,
            "abs_roll_or_pitch_gt_1_at_input_of_last_step": (np.abs(xp_last[:, rp]).max(1) > 1.0) if rp is not None else np.zeros(len(starts), bool),
        }
        res = {"n": int(len(starts))}
        tot_m = (em ** 2).mean(0)
        tot_p = (ep ** 2).mean(0)
        for name, m in cats.items():
            row = {"frac_windows": float(m.mean())}
            for g, idx in gi.items():
                tm, tp = tot_m[idx].sum(), tot_p[idx].sum()
                row[g] = {"share_mse_model": float((em[m][:, idx] ** 2).sum() / len(em) / tm) if tm > 0 else 0.0,
                          "rmse_model_in_cat": float(np.sqrt((em[m][:, idx] ** 2).mean())) if m.any() else None,
                          "rmse_persist_in_cat": float(np.sqrt((ep[m][:, idx] ** 2).mean())) if m.any() else None}
            res[name] = row
        out[k] = res
        print(f"== k={k}: {len(starts)} windows")
        groups = [g for g in gi if g != "last_action"]
        print("category".ljust(48) + "frac  " + " ".join(g[:10].rjust(11) for g in groups))
        for name in cats:
            r = res[name]
            print(name[:47].ljust(48) + f"{r['frac_windows']:.3f} " +
                  " ".join(f"{r[g]['share_mse_model']:11.2f}" for g in groups))
        print("  (numbers = share of the model's MSE of that group coming from the category)")
        for g in [x for x in a.groups.split(",") if x in gi]:
            idx = gi[g]
            ae = np.abs(em[:, idx]).max(1)
            q = np.quantile(ae, [0.5, 0.9, 0.99, 0.999])
            top = np.argsort(-ae)[:8]
            print(f"  {g}: |err| quantiles 50/90/99/99.9% = " + " ".join(f"{v:.4g}" for v in q)
                  + f"; top-1% windows hold {np.sort(ae ** 2)[-len(ae) // 100:].sum() / (ae ** 2).sum():.2f} of the squared error")
            xin = WD.rows_arrays(data, starts[top])[0]
            for j, w in enumerate(top):
                r = last[w]
                print(f"    row {r} ep_step {int(data['episode_step'][r])} done {int(data['done'][r])} "
                      f"shock_prev {int(data['shock_next'][r - 1]) if r > 0 else -1} err {em[w, idx].round(4).tolist()} "
                      f"input {xin[j, idx].round(4).tolist()} rp_in {xin[j, rp].round(3).tolist() if rp is not None else ''}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
