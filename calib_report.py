"""Apply the frozen calibration selection rule and render CALIBRATION_REPORT.md.

Separate from calibrate.py so the report can be regenerated without re-running the grid.

The selection rule is frozen in calibrate.py's docstring. The one operational number under
discussion is the eligibility threshold for the largest swept h: it was frozen at 0.05
assuming converged runs, but calibration is deliberately short (8k steps) and settings can
sit just above it while still clearly declining. --max-h-threshold makes that choice
explicit and recorded rather than silent.

    python3 calib_report.py                        # rule exactly as frozen
    python3 calib_report.py --max-h-threshold 0.10 # disclosed pre-analysis amendment
"""

import argparse
import glob
import json
import os

import numpy as np

import calibrate
import stage1

OUT = calibrate.OUT
BAND_LO, BAND_HI = calibrate.BAND
SMALLEST_H_MIN = 0.80  # smallest swept h must be clearly capacity-limited


def load():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    return [r for r in rows if "regret" in r]


def grid(rows):
    """(K, epsK, h) -> (mean regret, seed sd, n)."""
    g = {}
    for K in calibrate.KS:
        for e in calibrate.EPSKS:
            for h in calibrate.HS:
                v = [r["regret"] for r in rows
                     if r["K"] == K and r["epsK"] == e and r["h"] == h]
                if v:
                    g[(K, e, h)] = (float(np.mean(v)),
                                    float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                                    len(v))
    return g


def longest_adjacent_band(g, K, e):
    """Longest run of CONSECUTIVE swept h whose mean regret lies in the band."""
    best, cur = [], []
    for h in calibrate.HS:
        if (K, e, h) in g and BAND_LO <= g[(K, e, h)][0] <= BAND_HI:
            cur.append(h)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return best


def evaluate(g, K, e, max_h_threshold):
    hs = [h for h in calibrate.HS if (K, e, h) in g]
    if not hs:
        return None
    means = [g[(K, e, h)][0] for h in hs]
    band = longest_adjacent_band(g, K, e)
    checks = {
        "monotone_decreasing": all(b <= a + 1e-9 for a, b in zip(means, means[1:])),
        "smallest_h_capacity_limited": means[0] >= SMALLEST_H_MIN,
        "largest_h_approaches_bayes": means[-1] <= max_h_threshold,
    }
    sds = [g[(K, e, h)][1] for h in band]
    return {"K": K, "epsK": e, "band": band, "n_adjacent": len(band),
            "mean_seed_sd": float(np.mean(sds)) if sds else float("nan"),
            "regret_by_h": {str(h): g[(K, e, h)][0] for h in hs},
            "sd_by_h": {str(h): g[(K, e, h)][1] for h in hs},
            "smallest_h_regret": means[0], "largest_h_regret": means[-1],
            "checks": checks, "eligible": all(checks.values()) and len(band) >= 3}


def select(cands):
    """Frozen order: most adjacent h in band -> lowest mean seed SD -> smaller K."""
    ok = [c for c in cands if c["eligible"]]
    if not ok:
        return None
    return sorted(ok, key=lambda c: (-c["n_adjacent"], c["mean_seed_sd"], c["K"]))[0]


def chains_per_dim(g, K, e):
    """The postmortem's prediction: regret should track the fraction of beliefs that do
    not fit, i.e. (1-regret)*K chains held should be roughly proportional to h."""
    out = {}
    for h in calibrate.HS:
        if (K, e, h) in g:
            held = (1 - g[(K, e, h)][0]) * K
            out[h] = {"held": held, "per_dim": held / h, "saturated": held > 0.95 * K}
    return out


def main(max_h_threshold):
    rows = load()
    g = grid(rows)
    cands = [c for c in (evaluate(g, K, e, max_h_threshold)
                         for K in calibrate.KS for e in calibrate.EPSKS) if c]
    chosen = select(cands)

    L = []
    w = L.append
    w("# Calibration Report — K-chain environment\n")
    w("**Environment design, not evidence.** All runs here are disposable: calibration-only "
      f"seeds {list(calibrate.SEEDS)} never reused in Stage 1 or 2, no checkpoints kept, "
      f"{calibrate.STEPS} training steps chosen to rank settings rather than establish "
      "performance. The only output that survives is a frozen (K, ε).\n")
    w(f"- runs: {len(rows)} of "
      f"{len(calibrate.KS)*len(calibrate.EPSKS)*len(calibrate.HS)*len(calibrate.SEEDS)}")
    w(f"- band: [{BAND_LO}, {BAND_HI}]  ·  smallest-h floor: {SMALLEST_H_MIN}"
      f"  ·  largest-h ceiling: **{max_h_threshold}**"
      + ("  *(as frozen)*" if abs(max_h_threshold - 0.05) < 1e-9 else
         "  *(disclosed amendment — 0.05 was frozen assuming converged runs; calibration "
         "is deliberately short)*"))
    w("- rule: most adjacent h in band → lowest mean seed SD → smaller K\n")

    w("## Regret by setting\n")
    w("| K | εK | " + " | ".join(f"h={h}" for h in calibrate.HS) +
      " | band | adj | seed SD | eligible |")
    w("|---|---|" + "---|" * (len(calibrate.HS) + 4))
    for c in cands:
        cells = []
        for h in calibrate.HS:
            k = str(h)
            if k not in c["regret_by_h"]:
                cells.append("—")
                continue
            m = c["regret_by_h"][k]
            cells.append(f"**{m:.3f}**" if BAND_LO <= m <= BAND_HI else f"{m:.3f}")
        flag = "✅" if c["eligible"] else "—"
        w(f"| {c['K']} | {c['epsK']} | " + " | ".join(cells) +
          f" | {c['band'] or '—'} | {c['n_adjacent']} | {c['mean_seed_sd']:.4f} | {flag} |")
    w("\n*bold = inside the capacity band*\n")

    w("## Eligibility detail\n")
    w("| K | εK | monotone | smallest h ≥ 0.80 | largest h ≤ ceiling | ≥3 adjacent |")
    w("|---|---|---|---|---|---|")
    for c in cands:
        ch = c["checks"]
        w(f"| {c['K']} | {c['epsK']} | {'✓' if ch['monotone_decreasing'] else '✗'} "
          f"| {'✓' if ch['smallest_h_capacity_limited'] else '✗'} ({c['smallest_h_regret']:.3f}) "
          f"| {'✓' if ch['largest_h_approaches_bayes'] else '✗'} ({c['largest_h_regret']:.3f}) "
          f"| {'✓' if c['n_adjacent'] >= 3 else '✗'} ({c['n_adjacent']}) |")
    w("")

    w("## Does regret track the fraction of beliefs that do not fit?\n")
    w("The postmortem's prediction, and the reason this environment was built. Chains "
      "effectively held = (1 − regret)·K.\n")
    for c in cands:
        cd = chains_per_dim(g, c["K"], c["epsK"])
        unsat = [v["per_dim"] for v in cd.values() if not v["saturated"]]
        w(f"- **K={c['K']}, εK={c['epsK']}** — chains per state dimension: " +
          ", ".join(f"h={h}:{v['per_dim']:.2f}" + ("*" if v["saturated"] else "")
                    for h, v in cd.items()) +
          (f"  → mean {np.mean(unsat):.2f} before saturation" if unsat else ""))
    w("\n*\\* saturated: holding >95% of chains, so per-dimension figure is floored by K*\n")

    w("## Selection\n")
    if chosen:
        w(f"**K = {chosen['K']}, εK = {chosen['epsK']}** "
          f"(ε = {chosen['epsK']/chosen['K']:.6g}), emissions 0.1/0.9\n")
        w(f"- band: h ∈ {chosen['band']} ({chosen['n_adjacent']} adjacent)")
        w(f"- mean between-seed SD over the band: {chosen['mean_seed_sd']:.4f}")
        w(f"- smallest h regret {chosen['smallest_h_regret']:.3f}, "
          f"largest {chosen['largest_h_regret']:.3f}\n")
        w("Frozen for Protocol v2. Stage 1 re-selects λ under this environment; calibration's "
          "provisional λ is not carried forward.\n")
    else:
        w("**No eligible setting.** No (K, εK) satisfied all three conditions with ≥3 "
          "adjacent h in the band. The environment is not ready to freeze; report and stop "
          "rather than widening the band.\n")

    result = {"max_h_threshold": max_h_threshold, "n_runs": len(rows),
              "candidates": cands, "chosen": chosen}
    stage1.atomic_write_json(os.path.join(OUT, "calibration.json"), result)
    txt = "\n".join(L) + "\n"
    stage1.atomic_write_text(os.path.join(OUT, "CALIBRATION_REPORT.md"), txt)
    print(txt)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-h-threshold", type=float, default=0.05)
    main(ap.parse_args().max_h_threshold)
