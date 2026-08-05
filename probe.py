"""Stage 1 postmortem probe. NOT part of Stage 1 selection.

Tests the diagnosis that this environment's predictive sufficient statistic is
effectively one number -- the belief over z -- with volatility acting only on update
dynamics rather than needing to be separately represented.

Reads the 90 Stage 1 checkpoints. Never retrains, never modifies them. Probe parameters
are separate and no gradient ever reaches a model.

    python3 probe.py
"""

import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

import env
import model
import stage1

_HERE = stage1._HERE
RUNS = os.path.join(_HERE, "runs/stage1")

# PROTOCOL.md §8
PROBE_LR = 1e-2
PROBE_STEPS = 2000
TRAIN_SEQS = slice(0, 48)
TEST_SEQS = slice(48, 64)


def bayes_beliefs(x, cfg=env.EnvConfig()):
    """Predictive beliefs b_{t+1|t} over the 4 latent states. (n_seq, T-1, 4)."""
    n, T = x.shape
    A = env.transition_matrix(cfg)
    e1 = env.emission_p1(cfg)
    b = np.full((n, 4), 0.25)
    out = np.empty((n, T - 1, 4))
    for t in range(T - 1):
        lik = np.where(x[:, t : t + 1] == 1, e1[None, :], 1.0 - e1[None, :])
        b = b * lik
        b /= b.sum(axis=1, keepdims=True)
        b = b @ A
        out[:, t] = b
    return out


@torch.no_grad()
def collect_h(agent, x):
    """Replicate the eval forward pass, keeping h_t. h_t is the state that produces the
    prediction for x_{t+1}, so it aligns with the Bayes predictive belief at t."""
    h, dw, ds = agent.init_state(x.shape[0], x.device)
    H, P = [], []
    for t in range(x.shape[1] - 1):
        m = agent.f_phi(h)
        logit = agent.p_w(torch.cat([h, m], dim=-1)).squeeze(-1)
        h_hat = agent.p_s(m)
        H.append(h)
        P.append(torch.sigmoid(logit))

        x_next = x[:, t + 1]
        dw_slot = dw if agent.use_dw else torch.zeros_like(dw)
        g_in = torch.cat([x_next.unsqueeze(-1), m, dw_slot, agent._ds_slot(ds)], dim=-1)
        h_new = agent.g(g_in, h)
        dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
        ds = h_new - h_hat
        h = h_new
    return torch.stack(H, dim=1), torch.stack(P, dim=1)


def linear_probe(feat_tr, y_tr, feat_te, y_te, h_dim, seed=0):
    """§8: a single linear layer, h -> 2 logits. Linear only -- a nonlinear probe would
    measure the probe's capacity, not h's linear accessibility."""
    torch.manual_seed(seed)
    p = nn.Linear(h_dim, 2)
    opt = torch.optim.AdamW(p.parameters(), lr=PROBE_LR)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(p(feat_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return float((p(feat_te).argmax(-1) == y_te).float().mean())


def linear_r2(feat_tr, y_tr, feat_te, y_te):
    """Closed-form least squares from h (plus bias) to a continuous target."""
    A = torch.cat([feat_tr, torch.ones(len(feat_tr), 1)], dim=-1)
    w = torch.linalg.lstsq(A, y_tr.unsqueeze(-1)).solution
    B = torch.cat([feat_te, torch.ones(len(feat_te), 1)], dim=-1)
    pred = (B @ w).squeeze(-1)
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    d = env.heldout()
    x = torch.tensor(d["x"], dtype=torch.float32)
    bi = env.BURN_IN

    # Targets, computed once -- the held-out set is identical for every checkpoint.
    beliefs = bayes_beliefs(d["x"])
    p_z = torch.tensor(beliefs[..., 1] + beliefs[..., 3], dtype=torch.float32)
    p_v = torch.tensor(beliefs[..., 2] + beliefs[..., 3], dtype=torch.float32)
    z_true = torch.tensor(d["z"][:, 1:].astype(np.int64))
    v_true = torch.tensor(d["v"][:, 1:].astype(np.int64))

    # The belief marginals must reproduce the reference filter exactly.
    assert np.allclose(0.25 + 0.5 * p_z.numpy(), env.bayes_predict(d["x"]), atol=1e-12)

    flat = lambda t, s: t[s, bi:].reshape(-1)
    y = {
        "z_tr": flat(z_true, TRAIN_SEQS), "z_te": flat(z_true, TEST_SEQS),
        "v_tr": flat(v_true, TRAIN_SEQS), "v_te": flat(v_true, TEST_SEQS),
        "pz_tr": flat(p_z, TRAIN_SEQS), "pz_te": flat(p_z, TEST_SEQS),
        "pv_tr": flat(p_v, TRAIN_SEQS), "pv_te": flat(p_v, TEST_SEQS),
    }
    base_z = float(max(y["z_te"].float().mean(), 1 - y["z_te"].float().mean()))
    base_v = float(max(y["v_te"].float().mean(), 1 - y["v_te"].float().mean()))

    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "B2_*.json"))):
        with open(path) as f:
            rec = json.load(f)
        ckpt = os.path.join(RUNS, "ckpt", rec["tag"] + ".pt")
        agent = model.build(rec["h_dim"], "B2", rec["seed"])
        agent.load_state_dict(torch.load(ckpt, weights_only=True))
        agent.eval()

        H, P = collect_h(agent, x)

        # Integrity: the replicated forward must reproduce the recorded final eval.
        pc = P[:, bi:].clamp(1e-12, 1 - 1e-12)
        tgt = x[:, 1:][:, bi:]
        loss = float(-(tgt * pc.log() + (1 - tgt) * (1 - pc).log()).mean())
        recorded = rec["history"][-1]["loss"]
        assert abs(loss - recorded) < 1e-5, f"{rec['tag']}: {loss} vs {recorded}"

        f_tr, f_te = H[TRAIN_SEQS, bi:].reshape(-1, rec["h_dim"]), H[TEST_SEQS, bi:].reshape(-1, rec["h_dim"])
        row = {
            "tag": rec["tag"], "h": rec["h_dim"], "lambda": rec["lambda"], "seed": rec["seed"],
            "regret": rec["final_regret"], "state_velocity": rec["history"][-1]["dh"],
            "eval_loss": loss,
            "acc_z": linear_probe(f_tr, y["z_tr"], f_te, y["z_te"], rec["h_dim"]),
            "acc_v": linear_probe(f_tr, y["v_tr"], f_te, y["v_te"], rec["h_dim"]),
            "r2_pz": linear_r2(f_tr, y["pz_tr"], f_te, y["pz_te"]),
            "r2_pv": linear_r2(f_tr, y["pv_tr"], f_te, y["pv_te"]),
        }
        rows.append(row)
        print(f"  {row['tag']}  regret={row['regret']:.4f}  acc_z={row['acc_z']:.3f}  "
              f"acc_v={row['acc_v']:.3f}  R2(P(z))={row['r2_pz']:.4f}  R2(P(v))={row['r2_pv']:.4f}")

    out = {"baseline_acc_z": base_z, "baseline_acc_v": base_v,
           "n_checkpoints": len(rows), "probe": {"lr": PROBE_LR, "steps": PROBE_STEPS,
           "train_seqs": [0, 48], "test_seqs": [48, 64], "burn_in": bi}, "rows": rows}
    stage1.atomic_write_json(os.path.join(RUNS, "postmortem.json"), out)
    print(f"\nwrote runs/stage1/postmortem.json ({len(rows)} checkpoints)")
    return out


if __name__ == "__main__":
    main()
