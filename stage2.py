"""Stage 2 driver. The pilot runs through this exact code path as a configuration.

FROZEN DESIGN DECISIONS (agreed before any Stage 2 result exists):
  delay      d = 9 in BOTH horizon arms. The only intended difference between arms is what
             delta_s summarises, not when feedback arrives.
  aux loss   normalised by the DETACHED target variance, computed identically in both arms.
             Raw and normalised losses are both logged so normalisation cannot hide a scale
             difference.
  effect     0.05 for within-horizon A-vs-B2 and A-vs-B3; 0.03 for the
             difference-of-differences interactions.

ENVIRONMENT, frozen by Calibration v2:
  K = 16, eps = 0.015625 (epsK 0.25), emissions 0.1/0.9, band h in {4, 6, 8, 12}

    python3 stage2.py --config pilot
    python3 stage2.py --config stage2      # not defined until the pilot passes
"""

import argparse
import contextlib
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

import envk
import modelk2
import stage1

_HERE = stage1._HERE
ENV_CFG = envk.EnvKConfig(K=16, eps=0.015625, p_low=0.1, p_high=0.9)


@dataclass(frozen=True)
class S2Config:
    name: str
    runs_dir: str
    conditions: tuple
    horizons: tuple
    h_values: tuple
    seeds: tuple
    lams: tuple = (0.1,)
    delay: int = 9
    normalise_aux: bool = True
    lam: float = 0.1
    steps: int = 30_000
    window: int = 128
    batch: int = 64
    seq: int = 4096
    eval_every: int = 1000
    final_evals: int = 3
    peak_lr: float = 3e-3
    floor_lr: float = 3e-4
    warmup: int = 500
    clip: float = 1.0
    workers: int = 12


# Systems-validity gate, NOT evidence. Never used to select architecture, lambda, h, or
# thresholds; its seeds and checkpoints are never reused in Stage 2.
PILOT = S2Config(
    name="pilot", runs_dir="runs/pilot",
    conditions=("B2", "B3", "A"), horizons=(1, 8), h_values=(8,),
    seeds=(401, 402, 403, 404),  # paired across every cell
)

# Protocol v2 §4: lambda is re-selected under this environment before Stage 2.
# B2 only, so the selection cannot favour the treatment. Own seed namespace.
LAMBDA_SEL = S2Config(
    name="lambda", runs_dir="runs/lambda",
    conditions=("B2",), horizons=(8,), h_values=(4, 6, 8, 12),
    seeds=(501, 502, 503), lams=(0.03, 0.1, 0.3),
)

def _selected_lambda(default=0.1):
    """Read the lambda frozen by the §4 selection, if it has run."""
    p = os.path.join(_HERE, "runs/lambda/lambda_selected.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)["selected_lambda"]
    return default


# The experiment. 2 horizons x 4 conditions x band-h x 10 seeds (power gate: governing
# required n = 4, so the 10-seed minimum applies).
STAGE2 = S2Config(
    name="stage2", runs_dir="runs/stage2",
    conditions=("B1", "B2", "B3", "A"), horizons=(1, 8), h_values=(4, 6, 8, 12),
    seeds=tuple(range(1, 11)),
    # BOTH must carry the selected value: jobs() builds the per-run lambda from `lams`,
    # so setting only `lam` silently leaves every run on the (0.1,) default.
    lams=(_selected_lambda(),), lam=_selected_lambda(),
)

CONFIGS = {c.name: c for c in (PILOT, LAMBDA_SEL, STAGE2)}


def tag(cond, k, h, seed, lam=None):
    base = f"{cond}_k{k}_h{h:03d}_s{seed}"
    return base if lam is None else f"lam{str(lam).replace('.', 'p')}_{base}"


def jobs(cfg):
    multi = len(cfg.lams) > 1
    return [(cfg.name, c, k, h, s, lam, multi) for lam in cfg.lams for k in cfg.horizons
            for h in cfg.h_values for c in cfg.conditions for s in cfg.seeds]


def record_path(cfg, cond, k, h, seed, lam=None):
    return os.path.join(_HERE, cfg.runs_dir, tag(cond, k, h, seed, lam) + ".json")


# ------------------------------------------------------------------ training

def _bayes(cfg_env):
    d = envk.heldout(cfg_env)
    return envk.mean_loss(envk.bayes_predict(d["x"], d["k"], cfg_env), d["x"]), d


def train_one(cond, k, h, seed, cfg: S2Config, lam=None):
    lam = cfg.lam if lam is None else lam
    bayes, hold = _bayes(ENV_CFG)
    xh = torch.tensor(hold["x"], dtype=torch.float32)
    kh = torch.tensor(hold["k"].astype(np.int64))

    agent = modelk2.build(h, ENV_CFG.K, cond, seed, horizon=k, delay=cfg.delay,
                          normalise_aux=cfg.normalise_aux)
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.peak_lr, betas=(0.9, 0.99),
                            eps=1e-8, weight_decay=0.0)
    rng = np.random.default_rng(seed * 1000 + k)  # paired: same stream for every condition

    wpb = cfg.seq // cfg.window
    xb = kb = state = None
    wpos, hist, failed = 0, [], None
    n_clipped, n_steps, first_consume = 0, 0, None

    for step in range(cfg.steps):
        if xb is None or wpos == wpb:
            d = envk.generate(cfg.batch, cfg.seq, rng, ENV_CFG)
            xb = torch.tensor(d["x"], dtype=torch.float32)
            kb = torch.tensor(d["k"].astype(np.int64))
            state = agent.init_state(cfg.batch, xb.device)
            wpos = 0
        for gp in opt.param_groups:
            gp["lr"] = modelk2_lr(step, cfg)
        state = agent.detach_state(state)
        lo = wpos * cfg.window
        lw, ls, state, log, _ = agent.forward_window(
            xb[:, lo:lo + cfg.window + 1], kb[:, lo:lo + cfg.window + 1], state)
        loss = lw + lam * ls

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.clip))
        opt.step()
        wpos += 1
        n_steps += 1
        n_clipped += int(gn > cfg.clip)
        if first_consume is None and log["first_consume"] is not None:
            first_consume = log["first_consume"]

        if not np.isfinite(loss.item()):
            failed = f"non-finite training loss at step {step}"
            break

        if (step + 1) % cfg.eval_every == 0:
            ev, ok = evaluate(agent, xh, kh, cfg, bayes)
            if not ok:
                failed = f"non-finite eval loss at step {step + 1}"
                break
            hist.append({"step": step + 1, **ev, "grad_norm": gn,
                         "clip_rate": n_clipped / max(1, n_steps),
                         "aux_raw": log["aux_raw"], "aux_norm": float(ls),
                         "ds_norm": log["ds_norm"], "dh": log["dh"],
                         "h_norm": log["h_norm"],
                         "n_ds_consumed": log["n_ds_consumed"]})

    if failed is None:
        fin = hist[-cfg.final_evals:]
        at = next((r["loss"] for r in hist if r["step"] == 25_000), None)
        out = {"final_loss": float(np.mean([r["loss"] for r in fin])),
               "final_regret": float(np.mean([r["regret"] for r in fin])),
               "converged": at is not None and abs(hist[-1]["loss"] - at) / at < 0.01}
    else:
        out = {"final_loss": float("nan"), "final_regret": float("nan"), "converged": False}

    torch.save(agent.state_dict(),
               os.path.join(_HERE, cfg.runs_dir, "ckpt", tag(cond, k, h, seed, lam) + ".pt"))
    return {"condition": cond, "horizon": k, "h_dim": h, "seed": seed,
            "delay": cfg.delay, "normalise_aux": cfg.normalise_aux, "lam": lam,
            "bayes_loss": bayes, "first_consume": first_consume,
            "expected_first_consume": cfg.delay,
            "n_params": sum(p.numel() for p in agent.parameters()),
            "failed": failed, "history": hist, **out}


def modelk2_lr(step, cfg):
    import math
    if step < cfg.warmup:
        return cfg.peak_lr * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.floor_lr + 0.5 * (cfg.peak_lr - cfg.floor_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(agent, xh, kh, cfg, bayes):
    agent.eval()
    st = agent.init_state(xh.shape[0], xh.device)
    preds = []
    n_pred = xh.shape[1] - 1
    for pos in range(0, n_pred, cfg.window):
        w = min(cfg.window, n_pred - pos)
        p, st = agent.probs(xh[:, pos:pos + w + 1], kh[:, pos:pos + w + 1], st)
        preds.append(p)
    agent.train()
    ph = torch.cat(preds, 1)[:, envk.BURN_IN:].clamp(1e-12, 1 - 1e-12)
    tg = xh[:, 1:][:, envk.BURN_IN:]
    loss = float(-(tg * ph.log() + (1 - tg) * (1 - ph).log()).mean())
    if not np.isfinite(loss):
        return {}, False
    return {"loss": loss, "regret": envk.regret(loss, bayes)}, True


# ------------------------------------------------------------------ driver

def run_one(args):
    torch.set_num_threads(1)
    cfg_name, cond, k, h, seed, lam, multi = args
    cfg = CONFIGS[cfg_name]
    t = tag(cond, k, h, seed, lam if multi else None)
    log_path = os.path.join(_HERE, cfg.runs_dir, "logs", t + ".log")
    ckpt_dir = os.path.join(_HERE, cfg.runs_dir, "ckpt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    prov = stage1.provenance()
    started, t0 = time.time(), time.perf_counter()
    with open(log_path, "w", buffering=1) as lf:
        with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            print(f"{t}  commit={prov['git_commit'][:8]}")
            try:
                rec = train_one(cond, k, h, seed, cfg, lam)
                status = "failed" if rec["failed"] else "ok"
                if status == "ok" and not all(
                        np.isfinite(v) for v in (rec["final_loss"], rec["final_regret"])):
                    status, rec["failed"] = "failed", "non-finite final value"
            except Exception:
                traceback.print_exc()
                rec = {"condition": cond, "horizon": k, "h_dim": h, "seed": seed,
                       "final_loss": float("nan"), "final_regret": float("nan"),
                       "history": [], "failed": None, "converged": False,
                       "traceback": traceback.format_exc()}
                status = "error"
            print("done:", status)

    rec.update({"tag": t, "status": status, "provenance": prov, "config": cfg.name,
                "started": started, "ended": time.time(),
                "runtime_s": time.perf_counter() - t0,
                "log": os.path.relpath(log_path, _HERE)})
    stage1.atomic_write_json(record_path(cfg, cond, k, h, seed, lam if multi else None), rec)
    return t, status, rec["final_regret"], rec["runtime_s"]


def main(cfg, workers=None):
    import multiprocessing as mp
    out_dir = os.path.join(_HERE, cfg.runs_dir)
    os.makedirs(out_dir, exist_ok=True)
    planned = jobs(cfg)
    todo = [j for j in planned
            if not os.path.exists(record_path(cfg, j[1], j[2], j[3], j[4],
                                                j[5] if j[6] else None))]

    stage1.atomic_write_json(os.path.join(out_dir, "manifest.json"), {
        "config": cfg.name, "written": time.time(), "provenance": stage1.provenance(),
        "env": asdict(ENV_CFG), "run_cfg": asdict(cfg),
        "n_planned": len(planned),
        "planned": [tag(c, k, h, s, lm if mu else None)
                    for _, c, k, h, s, lm, mu in planned],
        "workers": workers or cfg.workers, "hostname": socket.gethostname()})

    print(f"stage2[{cfg.name}] planned={len(planned)} todo={len(todo)} "
          f"workers={workers or cfg.workers}", flush=True)
    if not todo:
        print("nothing to do")
        return 0
    with mp.get_context("spawn").Pool(workers or cfg.workers) as pool:
        for i, (t, st, reg, secs) in enumerate(pool.imap_unordered(run_one, todo), 1):
            r = "  nan" if not np.isfinite(reg) else f"{reg:6.4f}"
            print(f"  [{i}/{len(todo)}] {t:<22} {st:6s} regret={r} {secs:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot", choices=sorted(CONFIGS))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    sys.exit(main(CONFIGS[a.config], a.workers))
