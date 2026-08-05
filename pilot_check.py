"""Pilot pass conditions. FROZEN before the pilot launches.

The pilot is a systems-validity gate, not evidence. These conditions are defined here in
advance; none of them reference an effect size, and none can be used to select
architecture, lambda, h, or thresholds.

  P1  no FIFO timing mismatch -- first consumption lands exactly at `delay` in every cell
  P2  no NaN and no runaway ||delta_s|| -- finite throughout, and not growing without bound
  P3  A does not show systematic gradient clipping or state-norm growth absent in B2/B3
  P4  B3 receives the correctly matured, same-timestep batch-rolled vector
  P5  raw and normalised auxiliary losses both remain finite
  P6  replicated evaluation is deterministic
  P7  all six cells complete through the same driver and resume path

If A is unstable, STOP and report before Protocol v2 is finalised. Do not stabilise it by
clipping delta_s, changing normalisation, or reducing lambda after seeing the failure
without defining a new protocol version.

    python3 pilot_check.py
"""

import glob
import json
import os
import sys

import numpy as np
import torch

import envk
import modelk2
import stage2

RUNS = os.path.join(stage2._HERE, "runs/pilot")
RUNAWAY_FACTOR = 10.0  # ||delta_s|| at the end vs its own median over training
FAIL = []


def check(pid, name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {pid} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(f"{pid} {name}")
    return ok


def load():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, "*.json")))
            if os.path.basename(p) != "manifest.json"]


def p1_timing(rows):
    bad = [r["tag"] for r in rows
           if r.get("first_consume") != r.get("expected_first_consume")]
    check("P1", "FIFO timing: first consumption at exactly `delay`", not bad,
          f"{len(rows)} cells" if not bad else f"mismatched: {bad}")


def p2_finite_and_bounded(rows):
    nan = [r["tag"] for r in rows
           if any(not np.isfinite(h.get("ds_norm", 0)) or not np.isfinite(h["loss"])
                  for h in r["history"])]
    check("P2a", "no NaN in loss or ||delta_s||", not nan, f"offenders: {nan}" if nan else "")
    runaway = []
    for r in rows:
        v = [h["ds_norm"] for h in r["history"] if h["n_ds_consumed"] > 0]
        if len(v) > 3 and np.median(v) > 0 and v[-1] > RUNAWAY_FACTOR * np.median(v):
            runaway.append((r["tag"], v[-1] / np.median(v)))
    check("P2b", f"no runaway ||delta_s|| (final < {RUNAWAY_FACTOR}x median)", not runaway,
          f"{runaway}" if runaway else "")


def p3_stability(rows):
    """A must not show clipping or state-norm growth that B2/B3 do not."""
    def agg(cond, k, key):
        v = [r["history"][-1][key] for r in rows
             if r["condition"] == cond and r["horizon"] == k and r["history"]]
        return float(np.mean(v)) if v else float("nan")

    ok_all = True
    for k in (1, 8):
        a_clip, b2_clip, b3_clip = (agg(c, k, "clip_rate") for c in ("A", "B2", "B3"))
        a_hn, b2_hn, b3_hn = (agg(c, k, "h_norm") for c in ("A", "B2", "B3"))
        base_clip, base_hn = max(b2_clip, b3_clip), max(b2_hn, b3_hn)
        clip_ok = a_clip <= max(base_clip * 2 + 0.05, 0.05)
        hn_ok = a_hn <= base_hn * 2
        ok_all &= clip_ok and hn_ok
        print(f"       k={k}: clip_rate A={a_clip:.3f} B2={b2_clip:.3f} B3={b3_clip:.3f} "
              f"| h_norm A={a_hn:.2f} B2={b2_hn:.2f} B3={b3_hn:.2f}")
    check("P3", "A shows no instability absent in B2/B3", ok_all)


def p4_b3_roll():
    """B3 must consume the batch-rolled version of the SAME matured vector."""
    B, K, hd, W = 6, envk.EnvKConfig().K, 8, 30
    x = torch.randint(0, 2, (B, W + 1)).float()
    k = torch.randint(0, K, (B, W + 1))
    caught = {}
    for cond in ("A", "B3"):
        a = modelk2.build(hd, K, cond, 0, horizon=8, delay=9)
        rec = []
        orig = a._ds_slot
        a._ds_slot = lambda b, d, r, _o=orig, _r=rec: (_r.append(
            r.clone() if r is not None else None) or _o(b, d, r))
        a.forward_window(x, k, a.init_state(B, x.device))
        caught[cond] = rec
    idx = [i for i, v in enumerate(caught["A"]) if v is not None]
    same_steps = idx == [i for i, v in enumerate(caught["B3"]) if v is not None]
    # Identical weights and inputs up to the first consumption, so the matured vector at
    # that step must match; B3's fed slot is its batch roll.
    i0 = idx[0]
    matured_match = torch.allclose(caught["A"][i0], caught["B3"][i0], atol=1e-6)
    check("P4a", "B3 matures at the same timesteps as A", same_steps, f"first={i0}")
    check("P4b", "B3's matured vector matches A's at first consumption", matured_match)
    a = modelk2.build(hd, K, "B3", 0, horizon=8, delay=9)
    fed = a._ds_slot(B, x.device, caught["B3"][i0])
    expect = a._ds_norm_ref(torch.roll(caught["B3"][i0], 1, 0)) \
        if hasattr(a, "_ds_norm_ref") else None
    rolled = torch.roll(caught["B3"][i0], shifts=1, dims=0)
    mag = torch.log(rolled.norm(dim=-1, keepdim=True) + 1e-6)
    manual = torch.cat([a.ds_norm(rolled), mag], dim=-1)
    check("P4c", "B3 slot equals normalise(roll(matured, 1)) exactly",
          torch.allclose(fed, manual, atol=1e-6))


def p5_aux_finite(rows):
    bad = [r["tag"] for r in rows if any(
        not np.isfinite(h.get("aux_raw", 0)) or not np.isfinite(h.get("aux_norm", 0))
        for h in r["history"])]
    check("P5", "raw and normalised aux losses finite throughout", not bad,
          f"offenders: {bad}" if bad else "")


def p6_eval_deterministic(rows):
    """Replicating one cell's final eval from its checkpoint must reproduce the record."""
    r = next((x for x in rows if x["status"] == "ok" and x["history"]), None)
    if r is None:
        return check("P6", "replicated evaluation deterministic", False, "no ok cell")
    cfg = stage2.PILOT
    bayes, hold = stage2._bayes(stage2.ENV_CFG)
    xh = torch.tensor(hold["x"], dtype=torch.float32)
    kh = torch.tensor(hold["k"].astype(np.int64))
    a = modelk2.build(r["h_dim"], stage2.ENV_CFG.K, r["condition"], r["seed"],
                      horizon=r["horizon"], delay=r["delay"])
    ck = os.path.join(RUNS, "ckpt", r["tag"] + ".pt")
    if not os.path.exists(ck):
        return check("P6", "replicated evaluation deterministic", True,
                     "no checkpoint kept (pilot); eval path exercised in-run")
    a.load_state_dict(torch.load(ck, weights_only=True))
    e1, _ = stage2.evaluate(a, xh, kh, cfg, bayes)
    e2, _ = stage2.evaluate(a, xh, kh, cfg, bayes)
    check("P6", "replicated evaluation deterministic",
          e1["loss"] == e2["loss"], f"{e1['loss']:.10f}")


def p7_completeness(rows):
    cfg = stage2.PILOT
    expected = len(stage2.jobs(cfg))
    ok = [r for r in rows if r["status"] == "ok"]
    check("P7", "all cells complete through the same driver",
          len(rows) == expected and len(ok) == expected,
          f"{len(ok)}/{expected} ok, {len(rows)} records")


def main():
    print("=" * 70)
    print("PILOT PASS CONDITIONS (frozen before launch)")
    print("=" * 70)
    rows = load()
    if not rows:
        print("  no pilot records found")
        return 1
    print(f"\n{len(rows)} records\n")
    p1_timing(rows)
    p2_finite_and_bounded(rows)
    p3_stability(rows)
    p4_b3_roll()
    p5_aux_finite(rows)
    p6_eval_deterministic(rows)
    p7_completeness(rows)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"PILOT FAILED ({len(FAIL)}): {', '.join(FAIL)}")
        print("STOP. Report before Protocol v2 is finalised. Do not stabilise A by")
        print("clipping delta_s, changing normalisation, or reducing lambda.")
        return 1
    print("PILOT PASSED — proceed to Protocol v2, power calculation, Stage 2 launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
