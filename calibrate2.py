"""Calibration v2 — K-chain environment. Environment design, NOT evidence.

FROZEN BEFORE LAUNCH. Nothing below is changed after any output is inspected.

Calibration v1 (216 runs, seeds 101-103, 8k steps, h ceiling 32) is SUPERSEDED and is
neither reused nor averaged in. It established the mechanism -- monotone regret in h at
all nine settings, ~0.89 chains held per state dimension -- but could not evaluate the
convergence condition fairly: 8k steps undertrains (K=16/h=32 ended at 0.0504 and still
descending, missing the 0.05 ceiling by 0.0004) and h capped at 32 is only two state
dimensions per chain at K=16. Convergence and capacity ceiling were inseparable there.

v2 separates them by training to the Protocol-v2 budget and extending the sweep:
  - if h=32 crosses 0.05 after 30k steps  -> v1's issue was undertraining
  - if only h=48/64 cross                 -> v1's sweep was too narrow
  - if none cross                         -> the environment genuinely fails the ceiling

GRID
  K       16 only
  epsK    {0.0625, 0.125, 0.25}
  h       {2, 4, 6, 8, 12, 16, 24, 32, 48, 64}
  seeds   {201, 202, 203}  -- fresh namespace, never reused in v1, Stage 1 or Stage 2
  steps   30,000, with the optimizer and evaluation schedule intended for Protocol v2

ELIGIBILITY (unchanged from v1 -- no threshold was moved after seeing which setting nearly
passed):
  1. smallest swept h regret >= 0.80
  2. largest swept h regret  <= 0.05
  3. at least three ADJACENT h inside [0.20, 0.80]
  4. regret monotonically decreasing in h

SELECTION among eligible settings, in strict order:
  1. most adjacent h in the band
  2. lowest mean between-seed SD over the band
  3. tie-break: epsK = 0.125, then 0.0625, then 0.25

    python3 calibrate2.py --run
"""

import argparse
import itertools
import os
import time

import numpy as np
import torch

import envk
import modelk
import stage1

OUT = os.path.join(stage1._HERE, "runs/calib2")

K = 16
EPSKS = (0.0625, 0.125, 0.25)
HS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SEEDS = (201, 202, 203)
TIE_BREAK = (0.125, 0.0625, 0.25)  # frozen before launch

# Protocol-v2 training schedule (PROTOCOL.md §4-§5).
STEPS = 30_000
WINDOW = 128
BATCH = 64
SEQ = 4096
EVAL_EVERY = 1000
FINAL_EVALS = 3
PEAK_LR, FLOOR_LR, WARMUP, CLIP = 3e-3, 3e-4, 500, 1.0
LAM = 0.1  # provisional; Stage 1 re-selects lambda under the frozen environment
BAND = (0.20, 0.80)
SMALLEST_H_MIN, LARGEST_H_MAX = 0.80, 0.05


def cfg_of(epsK):
    return envk.EnvKConfig(K=K, eps=epsK / K, p_low=0.1, p_high=0.9)


def run_one(args):
    epsK, h, seed = args
    torch.set_num_threads(1)
    cfg = cfg_of(epsK)
    hold = envk.heldout(cfg)
    bayes = envk.mean_loss(envk.bayes_predict(hold["x"], hold["k"], cfg), hold["x"])
    xh = torch.tensor(hold["x"], dtype=torch.float32)
    kh = torch.tensor(hold["k"].astype(np.int64))

    agent = modelk.build(h, K, seed)
    opt = torch.optim.AdamW(agent.parameters(), lr=PEAK_LR, betas=(0.9, 0.99),
                            eps=1e-8, weight_decay=0.0)
    rng = np.random.default_rng(seed * 1000 + int(epsK * 10000))

    wpb, xb, kb, state, wpos = SEQ // WINDOW, None, None, None, 0
    hist, t0 = [], time.perf_counter()
    for step in range(STEPS):
        if xb is None or wpos == wpb:
            d = envk.generate(BATCH, SEQ, rng, cfg)
            xb = torch.tensor(d["x"], dtype=torch.float32)
            kb = torch.tensor(d["k"].astype(np.int64))
            state = agent.init_state(BATCH, xb.device)
            wpos = 0
        for gp in opt.param_groups:
            gp["lr"] = modelk.lr_at(step, PEAK_LR, FLOOR_LR, WARMUP, STEPS)
        state = tuple(s.detach() for s in state)
        lo = wpos * WINDOW
        lw, ls, state, st = agent.forward_window(xb[:, lo:lo + WINDOW + 1],
                                                 kb[:, lo:lo + WINDOW + 1], state)
        opt.zero_grad(set_to_none=True)
        (lw + LAM * ls).backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP)
        opt.step()
        wpos += 1

        if (step + 1) % EVAL_EVERY == 0:
            agent.eval()
            s2, preds = agent.init_state(xh.shape[0], xh.device), []
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

    fin = hist[-FINAL_EVALS:]
    at25 = next((r["loss"] for r in hist if r["step"] == 25_000), None)
    return {"K": K, "epsK": epsK, "h": h, "seed": seed, "bayes_loss": bayes,
            "regret": float(np.mean([r["regret"] for r in fin])),
            "loss": float(np.mean([r["loss"] for r in fin])),
            "converged": at25 is not None and abs(hist[-1]["loss"] - at25) / at25 < 0.01,
            "dh": hist[-1]["dh"], "runtime_s": time.perf_counter() - t0, "history": hist}


def main_run(workers):
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    planned = list(itertools.product(EPSKS, HS, SEEDS))
    todo = [j for j in planned
            if not os.path.exists(os.path.join(OUT, f"e{j[0]}_h{j[1]}_s{j[2]}.json"))]
    print(f"calibration v2: {len(planned)} planned, {len(todo)} to run "
          f"(K={K}, {STEPS} steps, seeds {list(SEEDS)})", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, todo), 1):
            stage1.atomic_write_json(
                os.path.join(OUT, f"e{r['epsK']}_h{r['h']}_s{r['seed']}.json"), r)
            print(f"  [{i}/{len(todo)}] epsK={r['epsK']:<6} h={r['h']:>2} s={r['seed']} "
                  f"regret={r['regret']:.4f} conv={r['converged']} {r['runtime_s']:.0f}s",
                  flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    raise SystemExit(main_run(a.workers) if a.run else "use --run")
