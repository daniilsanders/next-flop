"""Minimal training loop for Experiment 1. Implements PROTOCOL.md §4, §5, §10, §11.

One function trains one (h_dim, condition, seed, lambda) run. The Stage-1 driver,
lambda/band selection, probes, and statistics are NOT here yet.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import env
import model

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RunCfg:
    """PROTOCOL.md §4-§5. Values are frozen; do not override in experiment runs."""

    seq_len: int = 4096  # predictions per sequence
    window: int = 128  # truncated-BPTT window
    batch: int = 64
    total_steps: int = 30_000
    eval_every: int = 1000
    peak_lr: float = 3e-3
    floor_lr: float = 3e-4
    warmup: int = 500
    grad_clip: float = 1.0
    burn_in: int = env.BURN_IN
    final_evals: int = 3  # mean of the last N evals is the reported value
    abort_check_step: int = 10_000  # §10 divergence abort
    converge_check_step: int = 25_000  # §10 convergence gate reference point
    env_cfg: env.EnvConfig = field(default_factory=env.EnvConfig)


def _batch(rng, cfg: RunCfg, device):
    d = env.generate(cfg.batch, cfg.seq_len, rng, cfg.env_cfg)
    return torch.tensor(d["x"], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(agent, x_heldout, cfg: RunCfg, bayes_loss: float):
    """Full held-out pass. Identical set for every condition, h, and seed."""
    agent.eval()
    state = agent.init_state(x_heldout.shape[0], x_heldout.device)
    preds = []
    # Length comes from the held-out data itself, never from cfg.seq_len -- the training
    # sequence length and the evaluation set are independent and only coincide by default.
    n_pred = x_heldout.shape[1] - 1
    for pos in range(0, n_pred, cfg.window):
        w = min(cfg.window, n_pred - pos)
        p, state = agent.predict_window(x_heldout[:, pos : pos + w + 1], state)
        preds.append(p)
    agent.train()

    p_hat = torch.cat(preds, dim=1)[:, cfg.burn_in :]
    target = x_heldout[:, 1:][:, cfg.burn_in :]
    p = p_hat.clamp(1e-12, 1 - 1e-12)
    loss = float(-(target * p.log() + (1 - target) * (1 - p).log()).mean())
    return loss, env.regret(loss, bayes_loss)


def run(h_dim, condition, seed, lam, cfg: RunCfg = RunCfg(), device="cpu", log=None):
    """Train one run. Returns the record written to runs/<tag>.json."""
    with open(os.path.join(_HERE, "reference.json")) as f:
        bayes_loss = json.load(f)["bayes_loss"]

    agent = model.build(h_dim, condition, seed).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.peak_lr, betas=(0.9, 0.99),
                            eps=1e-8, weight_decay=0.0)

    # Seed s fixes both init and the training stream, identically across conditions.
    rng = np.random.default_rng(seed)
    x_heldout = torch.tensor(env.heldout(cfg.env_cfg)["x"], dtype=torch.float32, device=device)

    windows_per_batch = cfg.seq_len // cfg.window
    xb, state, wpos = None, None, 0
    history, failed = [], None

    for step in range(cfg.total_steps):
        if xb is None or wpos == windows_per_batch:
            xb = _batch(rng, cfg, device)
            state = agent.init_state(cfg.batch, device)
            wpos = 0

        for gp in opt.param_groups:
            gp["lr"] = model.lr_at(step, cfg.peak_lr, cfg.floor_lr, cfg.warmup, cfg.total_steps)

        state = tuple(s.detach() for s in state)  # TBPTT boundary
        lo = wpos * cfg.window
        loss_w, loss_s, state, stats = agent.forward_window(
            xb[:, lo : lo + cfg.window + 1], state
        )
        loss = loss_w + lam * loss_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip))
        opt.step()
        wpos += 1

        if not np.isfinite(loss.item()):
            failed = f"nan at step {step}"
            break

        if (step + 1) % cfg.eval_every == 0:
            ev_loss, ev_regret = evaluate(agent, x_heldout, cfg, bayes_loss)
            # §10: a run is FAILED if loss is NaN. That includes eval loss -- weights can
            # diverge while the training loss stays large-but-finite.
            if not np.isfinite(ev_loss):
                failed = f"non-finite eval loss at step {step + 1}"
                break
            rec = {"step": step + 1, "loss": ev_loss, "regret": ev_regret,
                   "grad_norm": gnorm, **stats}
            history.append(rec)
            if log:
                log(rec)
            # §10 divergence abort
            if step + 1 == cfg.abort_check_step and ev_loss > env.LN2:
                failed = f"worse than frozen at {cfg.abort_check_step}"
                break

    if failed is None:
        finals = history[-cfg.final_evals :]
        final_loss = float(np.mean([r["loss"] for r in finals]))
        final_regret = float(np.mean([r["regret"] for r in finals]))
        # §10 convergence gate
        ref = next((r["loss"] for r in history if r["step"] == cfg.converge_check_step), None)
        converged = ref is not None and abs(history[-1]["loss"] - ref) / ref < 0.01
    else:
        final_loss = final_regret = float("nan")
        converged = False

    return {
        "h_dim": h_dim, "condition": condition, "seed": seed, "lambda": lam,
        "n_params": sum(p.numel() for p in agent.parameters()),
        "final_loss": final_loss, "final_regret": final_regret,
        "converged": converged, "failed": failed, "history": history,
        "state_dict": agent.state_dict(),  # kept for the post-hoc probes (§8)
    }
