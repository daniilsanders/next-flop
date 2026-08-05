"""K-chain calibration. Environment design, NOT evidence.

Every run here is disposable: fresh calibration seeds never reused later, no checkpoint
kept, short training chosen to rank settings rather than establish performance. Its only
output is a frozen (K, eps) for Protocol v2.

SELECTION RULE -- frozen here before any output is inspected (PROTOCOL v2 pre-registration):

  Among (K, epsK) settings, prefer in strict order:
    1. the most ADJACENT h values whose mean regret lies in [0.20, 0.80]
    2. then the lowest mean between-seed SD of regret over those h
    3. then the smaller K

  A setting is only eligible if, over its band:
    (a) mean regret is monotonically decreasing in h across the whole swept range
    (b) the smallest swept h has mean regret >= 0.80  (clearly capacity-limited)
    (c) the largest swept h has mean regret <= 0.05   (approaches Bayes)

Also reported: whether regret tracks the fraction of chain beliefs that cannot fit, which
is the postmortem's prediction and the reason this environment was built.

    python3 calibrate.py --run       # execute the grid
    python3 calibrate.py --analyse   # apply the frozen rule
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

import envk
import modelk
import stage1

OUT = os.path.join(stage1._HERE, "runs/calib")

# Grid (shifted from the original proposal: epsK below 0.25 keeps the regret denominator
# wide -- epsK=1.0 gives 0.03 nats, narrower than the environment that already failed).
KS = (8, 16, 32)
EPSKS = (0.0625, 0.125, 0.25)
HS = (2, 4, 6, 8, 12, 16, 24, 32)
SEEDS = (101, 102, 103)  # calibration-only namespace; never reused in Stage 1 or 2

STEPS = 8000
WINDOW = 128
BATCH = 64
SEQ = 2048
EVAL_EVERY = 1000
LAM = 0.1  # provisional; Stage 1 re-selects lambda under the frozen environment
BAND = (0.20, 0.80)


def cfg_of(K, epsK):
    return envk.EnvKConfig(K=K, eps=epsK / K, p_low=0.1, p_high=0.9)


def bayes_of(cfg):
    d = envk.heldout(cfg)
    return envk.mean_loss(envk.bayes_predict(d["x"], d["k"], cfg), d["x"]), d


def run_one(args):
    K, epsK, h, seed = args
    torch.set_num_threads(1)
    cfg = cfg_of(K, epsK)
    bayes, hold = bayes_of(cfg)
    xh = torch.tensor(hold["x"], dtype=torch.float32)
    kh = torch.tensor(hold["k"].astype(np.int64))

    agent = modelk.build(h, K, seed)
    opt = torch.optim.AdamW(agent.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0)
    rng = np.random.default_rng(seed * 1000 + K * 10 + int(epsK * 100))

    wpb = SEQ // WINDOW
    xb = kb = state = None
    wpos = 0
    hist = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        if xb is None or wpos == wpb:
            d = envk.generate(BATCH, SEQ, rng, cfg)
            xb = torch.tensor(d["x"], dtype=torch.float32)
            kb = torch.tensor(d["k"].astype(np.int64))
            state = agent.init_state(BATCH, xb.device)
            wpos = 0
        for gp in opt.param_groups:
            gp["lr"] = modelk.lr_at(step, warmup=300, total=STEPS)
        state = tuple(s.detach() for s in state)
        lo = wpos * WINDOW
        lw, ls, state, st = agent.forward_window(xb[:, lo:lo + WINDOW + 1],
                                                 kb[:, lo:lo + WINDOW + 1], state)
        opt.zero_grad(set_to_none=True)
        (lw + LAM * ls).backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        wpos += 1

        if (step + 1) % EVAL_EVERY == 0:
            agent.eval()
            s2 = agent.init_state(xh.shape[0], xh.device)
            preds = []
            for pos in range(0, envk.HELDOUT_PRED, WINDOW):
                p, s2 = agent.predict_window(xh[:, pos:pos + WINDOW + 1],
                                             kh[:, pos:pos + WINDOW + 1], s2)
                preds.append(p)
            agent.train()
            ph = torch.cat(preds, 1)[:, envk.BURN_IN:].clamp(1e-12, 1 - 1e-12)
            tg = xh[:, 1:][:, envk.BURN_IN:]
            loss = float(-(tg * ph.log() + (1 - tg) * (1 - ph).log()).mean())
            hist.append({"step": step + 1, "loss": loss,
                         "regret": envk.regret(loss, bayes), "dh": st["dh"]})

    fin = hist[-2:]
    return {"K": K, "epsK": epsK, "h": h, "seed": seed, "bayes_loss": bayes,
            "regret": float(np.mean([r["regret"] for r in fin])),
            "loss": float(np.mean([r["loss"] for r in fin])),
            "dh": hist[-1]["dh"], "runtime_s": time.perf_counter() - t0,
            "history": hist}


def main_run(workers):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    jobs = [j for j in itertools.product(KS, EPSKS, HS, SEEDS)
            if not os.path.exists(os.path.join(OUT, f"K{j[0]}_e{j[1]}_h{j[2]}_s{j[3]}.json"))]
    print(f"calibration: {len(KS)*len(EPSKS)*len(HS)*len(SEEDS)} planned, {len(jobs)} to run")
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            stage1.atomic_write_json(
                os.path.join(OUT, f"K{r['K']}_e{r['epsK']}_h{r['h']}_s{r['seed']}.json"), r)
            print(f"  [{i}/{len(jobs)}] K={r['K']:>2} epsK={r['epsK']:<6} h={r['h']:>2} "
                  f"s={r['seed']} regret={r['regret']:.3f} {r['runtime_s']:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.run:
        raise SystemExit(main_run(a.workers))
    raise SystemExit("use --run (analysis lives in calib_report.py)")
