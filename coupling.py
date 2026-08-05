"""Coupling between delta_w and delta_s in the trained Stage 1 systems.

Postmortem addendum. Diagnostic only -- it sets no threshold for the new environment.

Tests whether a structural argument from the generative process predicts the *realized*
delta_s in a trained system. delta_s = h_{t+1} - p_s(f(h_t)) depends on g, f and p_s as
well as the environment, so the environment constrains at most one of its determinants.

delta_w is scalar and delta_s is a vector, so a single correlation is underspecified.
Four quantities, all on the held-out set with the §8 split, canonical definitions here so
Protocol v2 §11 can cite them:

  M1  magnitude coupling      corr(|delta_w|, ||delta_s||_2)
  M2  linear predictability   held-out R^2 of a linear map scalar delta_w -> vector delta_s
  M3  signed directional      ||Cov(delta_s, delta_w)||_2 / sqrt(Var(delta_w) tr Cov(delta_s))
  M4  shuffle baseline        M1-M3 recomputed after same-timestep batch permutation of
                              delta_w, so finite-sample structure and time trends cannot
                              masquerade as coupling

Note M2 and M3 are algebraically linked: computed in-sample on the same data,
M3^2 == R^2. M2 is reported held-out, so they differ only by generalisation. They are not
independent evidence and should not be read as such.

    python3 coupling.py
"""

import glob
import json
import os

import numpy as np
import torch

import env
import model
import probe
import stage1

RUNS = os.path.join(stage1._HERE, "runs/stage1")
TRAIN_SEQS, TEST_SEQS = slice(0, 48), slice(48, 64)


@torch.no_grad()
def collect_deltas(agent, x):
    """Replicate the eval forward pass, keeping delta_w (scalar) and delta_s (vector)."""
    h, dw, ds = agent.init_state(x.shape[0], x.device)
    DW, DS = [], []
    for t in range(x.shape[1] - 1):
        m = agent.f_phi(h)
        logit = agent.p_w(torch.cat([h, m], dim=-1)).squeeze(-1)
        h_hat = agent.p_s(m)

        x_next = x[:, t + 1]
        dw_slot = dw if agent.use_dw else torch.zeros_like(dw)
        g_in = torch.cat([x_next.unsqueeze(-1), m, dw_slot, agent._ds_slot(ds)], dim=-1)
        h_new = agent.g(g_in, h)

        dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
        ds = h_new - h_hat
        DW.append(dw.squeeze(-1))
        DS.append(ds)
        h = h_new
    return torch.stack(DW, dim=1), torch.stack(DS, dim=1)


def m1_magnitude(dw, ds):
    a = dw.abs()
    b = ds.norm(dim=-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def m2_r2(dw_tr, ds_tr, dw_te, ds_te):
    """Held-out variance of delta_s explained by a linear map from delta_w alone."""
    A = torch.stack([dw_tr, torch.ones_like(dw_tr)], dim=-1)
    w = torch.linalg.lstsq(A, ds_tr).solution
    pred = torch.stack([dw_te, torch.ones_like(dw_te)], dim=-1) @ w
    ss_res = ((ds_te - pred) ** 2).sum()
    ss_tot = ((ds_te - ds_te.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def m3_directional(dw, ds):
    a = dw - dw.mean()
    B = ds - ds.mean(0, keepdim=True)
    cov = (B * a.unsqueeze(-1)).mean(0)  # Cov(delta_s, delta_w), one entry per dimension
    var_w = (a ** 2).mean()
    tr_cov_s = (B ** 2).mean(0).sum()
    denom = torch.sqrt(var_w * tr_cov_s)
    return float(cov.norm() / denom) if denom > 0 else float("nan")


def metrics(dw, ds):
    """dw (n_seq, T), ds (n_seq, T, h) -> the three coupling scalars on the §8 test split."""
    f1 = lambda t, s: t[s, env.BURN_IN:].reshape(-1)
    f2 = lambda t, s: t[s, env.BURN_IN:].reshape(-1, t.shape[-1])
    dw_tr, dw_te = f1(dw, TRAIN_SEQS), f1(dw, TEST_SEQS)
    ds_tr, ds_te = f2(ds, TRAIN_SEQS), f2(ds, TEST_SEQS)
    return {"m1_magnitude": m1_magnitude(dw_te, ds_te),
            "m2_r2": m2_r2(dw_tr, ds_tr, dw_te, ds_te),
            "m3_directional": m3_directional(dw_te, ds_te)}


def main():
    d = env.heldout()
    x = torch.tensor(d["x"], dtype=torch.float32)
    with open(os.path.join(RUNS, "postmortem.json")) as f:
        pm = {r["tag"]: r for r in json.load(f)["rows"]}

    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "B2_*.json"))):
        with open(path) as f:
            rec = json.load(f)
        agent = model.build(rec["h_dim"], "B2", rec["seed"])
        agent.load_state_dict(
            torch.load(os.path.join(RUNS, "ckpt", rec["tag"] + ".pt"), weights_only=True))
        agent.eval()

        dw, ds = collect_deltas(agent, x)
        real = metrics(dw, ds)
        # Same-timestep permutation across sequences: preserves the marginal of delta_w at
        # every t (so time trends survive) and destroys only its pairing with delta_s.
        shuf = metrics(torch.roll(dw, shifts=1, dims=0), ds)

        p = pm[rec["tag"]]
        row = {"tag": rec["tag"], "h": rec["h_dim"], "lambda": rec["lambda"],
               "seed": rec["seed"], "regret": rec["final_regret"],
               "r2_pz": p["r2_pz"], "r2_pv": p["r2_pv"],
               **real, **{k + "_shuf": v for k, v in shuf.items()}}
        rows.append(row)
        print(f"  {row['tag']}  M1={row['m1_magnitude']:+.3f} ({row['m1_magnitude_shuf']:+.3f})  "
              f"M2={row['m2_r2']:+.4f} ({row['m2_r2_shuf']:+.4f})  "
              f"M3={row['m3_directional']:.4f} ({row['m3_directional_shuf']:.4f})")

    out = {"n_checkpoints": len(rows), "split": {"train": [0, 48], "test": [48, 64],
           "burn_in": env.BURN_IN}, "shuffle": "batch roll by 1, same timestep",
           "rows": rows}
    stage1.atomic_write_json(os.path.join(RUNS, "coupling.json"), out)
    print(f"\nwrote runs/stage1/coupling.json ({len(rows)} checkpoints)")
    return out


if __name__ == "__main__":
    main()
