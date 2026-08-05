"""Does multi-step self-prediction decouple delta_s from delta_w?

Zero training. Uses the existing Stage 1 B2 checkpoints.

Conditional on history, one-step delta_s is a deterministic function of delta_w, because
h_{t+1} depends on the world only through x_{t+1} and p_s predicts exactly one step. At
horizon k, h_{t+k} depends on x_{t+1..t+k} -- k bits -- while delta_w^t still carries one.
So delta_s^(k) should contain information delta_w^t cannot have, and its recoverability
from delta_w^t should fall with k.

For each k, an idealised k-step self-predictor is stood up as the best LINEAR map
m_t -> h_{t+k}, fit on the probe-train split. Its residual is the delta_s^(k) proxy. We
then report the held-out R^2 of a linear map delta_w^t -> that residual, with the same
same-timestep batch-permutation baseline used in coupling.py.

Using the best linear p_s^(k) rather than the trained p_s keeps the comparison across k
internally consistent -- the trained p_s only exists for k=1.

    python3 horizon.py
"""

import glob
import json
import os

import numpy as np
import torch

import coupling
import env
import model
import stage1

RUNS = os.path.join(stage1._HERE, "runs/stage1")
KS = (1, 2, 4, 8, 16, 32)
TRAIN, TEST = slice(0, 48), slice(48, 64)


@torch.no_grad()
def collect(agent, x):
    """h_t, m_t and delta_w^t along the held-out set."""
    h, dw, ds = agent.init_state(x.shape[0], x.device)
    H, M, DW = [], [], []
    for t in range(x.shape[1] - 1):
        m = agent.f_phi(h)
        logit = agent.p_w(torch.cat([h, m], dim=-1)).squeeze(-1)
        h_hat = agent.p_s(m)
        H.append(h)
        M.append(m)

        x_next = x[:, t + 1]
        dw_slot = dw if agent.use_dw else torch.zeros_like(dw)
        g_in = torch.cat([x_next.unsqueeze(-1), m, dw_slot, agent._ds_slot(ds)], dim=-1)
        h_new = agent.g(g_in, h)
        dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
        ds = h_new - h_hat
        DW.append(dw.squeeze(-1))
        h = h_new
    return torch.stack(H, 1), torch.stack(M, 1), torch.stack(DW, 1)


def _fit_predict(A_tr, Y_tr, A_te):
    w = torch.linalg.lstsq(A_tr, Y_tr).solution
    return A_te @ w


def horizon_r2(Hs, Ms, DW, k, burn=env.BURN_IN):
    """Held-out R^2 of delta_w^t -> (h_{t+k} - best_linear(m_t)). Plus shuffle baseline."""
    T = Hs.shape[1]
    lo, hi = burn, T - k
    flat = lambda t, s: t[s, lo:hi].reshape(-1, t.shape[-1])
    y = lambda s: Hs[s, lo + k : hi + k].reshape(-1, Hs.shape[-1])
    d = lambda s: DW[s, lo:hi].reshape(-1)

    m_tr, m_te = flat(Ms, TRAIN), flat(Ms, TEST)
    A_tr = torch.cat([m_tr, torch.ones(len(m_tr), 1)], -1)
    A_te = torch.cat([m_te, torch.ones(len(m_te), 1)], -1)

    # Idealised k-step self-predictor, fit on train only.
    pred_te = _fit_predict(A_tr, y(TRAIN), A_te)
    r_tr = y(TRAIN) - _fit_predict(A_tr, y(TRAIN), A_tr)
    r_te = y(TEST) - pred_te

    # How much of h_{t+k} the self-model can predict at all. If this collapses, delta_s^(k)
    # degenerates into "the state" rather than "the surprise", and a large k buys nothing.
    yt = y(TEST)
    self_r2 = float(1 - (r_te ** 2).sum() / ((yt - yt.mean(0, keepdim=True)) ** 2).sum())

    dw_tr, dw_te = d(TRAIN), d(TEST)
    real = coupling.m2_r2(dw_tr, r_tr, dw_te, r_te)
    sh = DW.roll(1, dims=0)
    shuf = coupling.m2_r2(sh[TRAIN, lo:hi].reshape(-1), r_tr,
                          sh[TEST, lo:hi].reshape(-1), r_te)
    return real, shuf, self_r2


def main():
    d = env.heldout()
    x = torch.tensor(d["x"], dtype=torch.float32)
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "B2_*.json"))):
        with open(path) as f:
            rec = json.load(f)
        agent = model.build(rec["h_dim"], "B2", rec["seed"])
        agent.load_state_dict(
            torch.load(os.path.join(RUNS, "ckpt", rec["tag"] + ".pt"), weights_only=True))
        agent.eval()

        Hs, Ms, DW = collect(agent, x)
        row = {"tag": rec["tag"], "h": rec["h_dim"], "lambda": rec["lambda"],
               "seed": rec["seed"], "regret": rec["final_regret"]}
        for k in KS:
            r, s, sr = horizon_r2(Hs, Ms, DW, k)
            row[f"r2_k{k}"], row[f"r2_k{k}_shuf"], row[f"selfpred_k{k}"] = r, s, sr
        rows.append(row)
        print(f"  {row['tag']}  " + "  ".join(
            f"k={k}:{row[f'r2_k{k}']:+.3f}/{row[f'selfpred_k{k}']:.2f}" for k in KS))

    out = {"n_checkpoints": len(rows), "ks": list(KS),
           "target": "h_{t+k} minus best linear map from m_t (fit on train split)",
           "shuffle": "batch roll by 1, same timestep", "rows": rows}
    stage1.atomic_write_json(os.path.join(RUNS, "horizon.json"), out)
    print(f"\nwrote runs/stage1/horizon.json ({len(rows)} checkpoints)")
    return out


if __name__ == "__main__":
    main()
