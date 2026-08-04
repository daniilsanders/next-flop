"""Stage 1 driver. Execution only -- no analysis, no interpretation, no tuning.

Runs the 90 B2 (h, seed, lambda) jobs of PROTOCOL.md §5-§6.1 and writes one immutable
record per run. Selection happens in analyze1.py, separately and afterwards.

    python3 stage1.py --config smoke      # tiny end-to-end test
    python3 stage1.py --config protocol   # the real sweep

Resume is safe: a job with a valid record is skipped, never rerun. A job that FAILED
under the protocol (nan, divergence abort) is a result, not a gap -- it is recorded and
never retried. A job whose process died leaves no record at all (writes are atomic) and
is simply picked up again.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import env
from train import RunCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ("PROTOCOL.md", "env.py", "model.py", "train.py", "stage1.py")


@dataclass(frozen=True)
class Stage1Config:
    name: str
    h_values: tuple
    seeds: tuple
    lambdas: tuple
    runs_dir: str
    workers: int
    run_cfg: RunCfg = field(default_factory=RunCfg)


PROTOCOL = Stage1Config(
    name="protocol",
    h_values=(2, 3, 4, 6, 8, 12, 16, 24, 32, 64),  # §3
    seeds=(1, 2, 3),  # §5
    lambdas=(0.03, 0.1, 0.3),  # §6.1
    runs_dir="runs/stage1",
    workers=8,
)

# Differs from PROTOCOL only in configuration. Same code path, start to finish.
SMOKE = Stage1Config(
    name="smoke",
    h_values=(2, 4),
    seeds=(1, 2),
    lambdas=(0.03, 0.1),
    runs_dir="runs/smoke",
    workers=4,
    run_cfg=RunCfg(seq_len=512, window=64, batch=8, total_steps=200, eval_every=50,
                   warmup=20, burn_in=64, abort_check_step=100, converge_check_step=100),
)

# Absurd learning rate -> nan. Exercises the real failure path with no test-only code.
SMOKE_FAIL = replace(SMOKE, name="smoke_fail", runs_dir="runs/smoke_fail",
                     run_cfg=replace(SMOKE.run_cfg, peak_lr=1e9, floor_lr=1e9))

# Undertrained on purpose so both h land inside [0.20, 0.80] and the analysis reaches
# the power gate. Exercises the continue branch that SMOKE (which stops on an empty
# band) does not.
SMOKE_BAND = replace(SMOKE, name="smoke_band", runs_dir="runs/smoke_band",
                     run_cfg=replace(SMOKE.run_cfg, peak_lr=1e-3, floor_lr=1e-4))

CONFIGS = {c.name: c for c in (PROTOCOL, SMOKE, SMOKE_FAIL, SMOKE_BAND)}


# ------------------------------------------------------------------ provenance

def _sha(path):
    with open(os.path.join(_HERE, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _git(*args, default="unavailable"):
    try:
        return subprocess.check_output(["git", "-C", _HERE, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def provenance():
    import torch
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", default="")),
        "protocol_sha": _sha("PROTOCOL.md"),
        "code_sha": hashlib.sha256(
            "".join(_sha(f) for f in CODE_FILES if f != "PROTOCOL.md").encode()
        ).hexdigest()[:16],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


# ------------------------------------------------------------------ enumeration

def tag(h, lam, seed):
    return f"B2_h{h:03d}_lam{str(lam).replace('.', 'p')}_s{seed:02d}"


def jobs(cfg: Stage1Config):
    """Deterministic order: h, then lambda, then seed."""
    return [(h, lam, seed) for h in cfg.h_values for lam in cfg.lambdas for seed in cfg.seeds]


def record_path(cfg, h, lam, seed):
    return os.path.join(_HERE, cfg.runs_dir, tag(h, lam, seed) + ".json")


# ------------------------------------------------------------------ io

def atomic_write_text(path, text):
    """Write via a temp file in the same directory, then rename. An interrupted write
    leaves no file at all rather than a plausible-looking partial one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


REQUIRED = ("tag", "status", "h_dim", "lambda", "seed", "condition", "final_loss",
            "final_regret", "history", "provenance", "started", "ended", "runtime_s",
            "final_three")


def validate_record(rec, cfg: Stage1Config, prov):
    """Returns (ok, reason). A record must be self-consistent AND from this code version."""
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        return False, f"missing keys: {missing}"
    if rec["status"] not in ("ok", "failed"):
        return False, f"bad status {rec['status']!r}"
    if rec["condition"] != "B2":
        return False, "Stage 1 is B2 only"
    for k in ("protocol_sha", "code_sha"):
        if rec["provenance"].get(k) != prov[k]:
            return False, f"{k} mismatch (record {rec['provenance'].get(k)}, now {prov[k]})"
    if rec["status"] == "failed":
        return (True, "") if rec.get("failed") else (False, "failed without a reason")

    n_expected = cfg.run_cfg.total_steps // cfg.run_cfg.eval_every
    if len(rec["history"]) != n_expected:
        return False, f"history {len(rec['history'])} != {n_expected} evals"
    if len(rec["final_three"]) != cfg.run_cfg.final_evals:
        return False, "final_three wrong length"
    for v in (rec["final_loss"], rec["final_regret"], *rec["final_three"]):
        if not np.isfinite(v):
            return False, "non-finite final value"
    return True, ""


# ------------------------------------------------------------------ worker

def run_one(args):
    """Executed in a worker process. Never raises -- errors become records."""
    import torch
    torch.set_num_threads(1)  # workers are parallel; let each have one core

    cfg_name, h, lam, seed = args
    cfg = CONFIGS[cfg_name]
    t = tag(h, lam, seed)
    path = record_path(cfg, h, lam, seed)
    log_path = os.path.join(_HERE, cfg.runs_dir, "logs", t + ".log")
    ckpt_dir = os.path.join(_HERE, cfg.runs_dir, "ckpt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    prov = provenance()
    started = time.time()
    t0 = time.perf_counter()

    with open(log_path, "w", buffering=1) as lf:
        with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            print(f"{t}  h={h} lambda={lam} seed={seed}  commit={prov['git_commit'][:8]}")
            try:
                import train
                out = train.run(h, "B2", seed, lam, cfg.run_cfg,
                                log=lambda r: print(json.dumps(r)))
                sd = out.pop("state_dict")
                torch.save(sd, os.path.join(ckpt_dir, t + ".pt"))
                status = "failed" if out["failed"] else "ok"
                finals = [r["regret"] for r in out["history"][-cfg.run_cfg.final_evals:]]
                # Second net: a record must never be written as "ok" with non-finite
                # finals. If it is, that is a failed run, not a valid one.
                if status == "ok" and not all(
                    np.isfinite(v) for v in (out["final_loss"], out["final_regret"], *finals)
                ):
                    status = "failed"
                    out["failed"] = "non-finite final value"
                rec = {**out, "tag": t, "status": status, "final_three": finals}
            except Exception:
                traceback.print_exc()
                rec = {"tag": t, "status": "error", "h_dim": h, "lambda": lam, "seed": seed,
                       "condition": "B2", "final_loss": float("nan"),
                       "final_regret": float("nan"), "history": [], "final_three": [],
                       "failed": None, "converged": False, "n_params": None,
                       "traceback": traceback.format_exc()}
            print(f"done: {rec['status']}")

    rec.update({"provenance": prov, "started": started, "ended": time.time(),
                "runtime_s": time.perf_counter() - t0, "config": cfg.name,
                "log": os.path.relpath(log_path, _HERE)})
    atomic_write_json(path, rec)
    return t, rec["status"], rec["final_regret"], rec["runtime_s"]


# ------------------------------------------------------------------ driver

def main(cfg: Stage1Config, workers=None):
    import multiprocessing as mp

    prov = provenance()
    planned = jobs(cfg)
    out_dir = os.path.join(_HERE, cfg.runs_dir)
    os.makedirs(out_dir, exist_ok=True)

    todo, done, corrupt = [], [], []
    for h, lam, seed in planned:
        p = record_path(cfg, h, lam, seed)
        if not os.path.exists(p):
            todo.append((cfg.name, h, lam, seed))
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
            ok, why = validate_record(rec, cfg, prov)
        except Exception as e:
            ok, why = False, f"unreadable: {e}"
        (done if ok else corrupt).append((tag(h, lam, seed), why))

    manifest = {
        "config": cfg.name,
        "written": time.time(),
        "provenance": prov,
        "run_cfg": asdict(cfg.run_cfg),
        "h_values": list(cfg.h_values),
        "seeds": list(cfg.seeds),
        "lambdas": list(cfg.lambdas),
        "n_planned": len(planned),
        "planned": [tag(h, lam, s) for h, lam, s in planned],
        "already_done": [t for t, _ in done],
        "corrupt_at_start": [{"tag": t, "reason": w} for t, w in corrupt],
    }
    atomic_write_json(os.path.join(out_dir, "manifest.json"), manifest)

    print(f"stage1[{cfg.name}]  planned={len(planned)}  done={len(done)}  "
          f"todo={len(todo)}  corrupt={len(corrupt)}")
    if corrupt:
        print("  CORRUPT RECORDS PRESENT -- not overwritten, not rerun. Inspect and remove:")
        for t, why in corrupt:
            print(f"    {t}: {why}")
        return 2
    if not todo:
        print("  nothing to do")
        return 0

    n_workers = workers or cfg.workers
    t_start = time.perf_counter()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for i, (t, status, regret, secs) in enumerate(pool.imap_unordered(run_one, todo), 1):
            r = "  nan" if not np.isfinite(regret) else f"{regret:5.3f}"
            print(f"  [{i}/{len(todo)}] {t}  {status:6s} regret={r}  {secs:6.1f}s")

    print(f"stage1[{cfg.name}] finished in {time.perf_counter()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    sys.exit(main(CONFIGS[a.config], a.workers))
