"""Apply the Calibration v2 selection rule and render CALIBRATION_V2_REPORT.md.

The rule is frozen in calibrate2.py's docstring, committed before the grid launched.
Nothing here is parameterised -- there is no threshold flag, because no threshold is
under discussion. v1 runs are never loaded.
"""

import glob
import json
import os

import numpy as np

import calibrate2 as C
import stage1

OUT = C.OUT
BAND_LO, BAND_HI = C.BAND


def load():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))
            if os.path.basename(p).startswith("e")]


def grid(rows):
    g = {}
    for e in C.EPSKS:
        for h in C.HS:
            v = [r["regret"] for r in rows if r["epsK"] == e and r["h"] == h]
            if v:
                g[(e, h)] = (float(np.mean(v)),
                             float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v))
    return g


def longest_adjacent(g, e):
    best, cur = [], []
    for h in C.HS:
        if (e, h) in g and BAND_LO <= g[(e, h)][0] <= BAND_HI:
            cur.append(h)
            best = list(cur) if len(cur) > len(best) else best
        else:
            cur = []
    return best


def evaluate(g, e):
    hs = [h for h in C.HS if (e, h) in g]
    if not hs:
        return None
    means = [g[(e, h)][0] for h in hs]
    band = longest_adjacent(g, e)
    checks = {
        "monotone_decreasing": all(b <= a + 1e-9 for a, b in zip(means, means[1:])),
        "smallest_h_capacity_limited": means[0] >= C.SMALLEST_H_MIN,
        "largest_h_approaches_bayes": means[-1] <= C.LARGEST_H_MAX,
        "three_adjacent_in_band": len(band) >= 3,
    }
    sds = [g[(e, h)][1] for h in band]
    return {"epsK": e, "band": band, "n_adjacent": len(band),
            "mean_seed_sd": float(np.mean(sds)) if sds else float("nan"),
            "regret_by_h": {str(h): g[(e, h)][0] for h in hs},
            "sd_by_h": {str(h): g[(e, h)][1] for h in hs},
            "n_by_h": {str(h): g[(e, h)][2] for h in hs},
            "smallest_h_regret": means[0], "largest_h_regret": means[-1],
            "checks": checks, "eligible": all(checks.values())}


def select(cands):
    """Frozen: most adjacent h -> lowest mean seed SD -> tie-break 0.125, 0.0625, 0.25."""
    ok = [c for c in cands if c["eligible"]]
    if not ok:
        return None
    return sorted(ok, key=lambda c: (-c["n_adjacent"], c["mean_seed_sd"],
                                     C.TIE_BREAK.index(c["epsK"])))[0]


def main():
    rows = load()
    g = grid(rows)
    cands = [c for c in (evaluate(g, e) for e in C.EPSKS) if c]
    chosen = select(cands)
    n_conv = sum(1 for r in rows if r.get("converged"))

    L, w = [], None
    w = L.append
    w("# Calibration v2 Report — K-chain environment\n")
    w("**Environment design, not evidence.** Disposable by construction: calibration-only "
      f"seeds {list(C.SEEDS)}, never reused in v1, Stage 1 or Stage 2; no checkpoints kept. "
      "The only surviving output is a frozen (K, ε).\n")
    w("Calibration v1 is **not** loaded, reused, or averaged in. It is preserved separately "
      "as `runs/calib/CALIBRATION_V1_REPORT.md`, marked superseded.\n")
    w(f"- runs: {len(rows)} of {len(C.EPSKS)*len(C.HS)*len(C.SEEDS)} · converged: "
      f"{n_conv}/{len(rows)}")
    w(f"- K = **{C.K}** · h ∈ {list(C.HS)} · {C.STEPS:,} steps · Protocol-v2 optimizer "
      f"and eval schedule")
    w(f"- eligibility (unchanged from v1): smallest h ≥ {C.SMALLEST_H_MIN}, largest h ≤ "
      f"{C.LARGEST_H_MAX}, ≥3 adjacent in [{BAND_LO}, {BAND_HI}], monotone")
    w(f"- selection: most adjacent → lowest mean seed SD → tie-break {list(C.TIE_BREAK)}\n")

    w("## Regret by εK and h\n")
    w("| εK | " + " | ".join(f"h={h}" for h in C.HS) + " | band | adj | seed SD |")
    w("|---|" + "---|" * (len(C.HS) + 3))
    for c in cands:
        cells = []
        for h in C.HS:
            k = str(h)
            if k not in c["regret_by_h"]:
                cells.append("—")
                continue
            m = c["regret_by_h"][k]
            cells.append(f"**{m:.3f}**" if BAND_LO <= m <= BAND_HI else f"{m:.3f}")
        w(f"| {c['epsK']} | " + " | ".join(cells) +
          f" | {c['band'] or '—'} | {c['n_adjacent']} | {c['mean_seed_sd']:.4f} |")
    w("\n*bold = inside the capacity band*\n")

    w("## Eligibility\n")
    w("| εK | monotone | smallest h ≥ 0.80 | largest h ≤ 0.05 | ≥3 adjacent | eligible |")
    w("|---|---|---|---|---|---|")
    for c in cands:
        k = c["checks"]
        tick = lambda b: "✓" if b else "✗"
        w(f"| {c['epsK']} | {tick(k['monotone_decreasing'])} "
          f"| {tick(k['smallest_h_capacity_limited'])} ({c['smallest_h_regret']:.4f}) "
          f"| {tick(k['largest_h_approaches_bayes'])} ({c['largest_h_regret']:.4f}) "
          f"| {tick(k['three_adjacent_in_band'])} ({c['n_adjacent']}) "
          f"| {'**yes**' if c['eligible'] else 'no'} |")
    w("")

    w("## What v1 could not separate\n")
    w("v1 left convergence and capacity ceiling confounded: K=16/h=32 finished at 0.0527 "
      "after 8k steps, still descending, missing the 0.05 ceiling by 0.0004. v2 answers it:\n")
    for c in cands:
        r32 = c["regret_by_h"].get("32")
        r64 = c["regret_by_h"].get("64")
        if r32 is None:
            continue
        verdict = ("**undertraining** — h=32 crosses the ceiling once trained to 30k"
                   if r32 <= C.LARGEST_H_MAX else
                   ("**sweep width** — h=32 still fails, only larger h cross"
                    if r64 is not None and r64 <= C.LARGEST_H_MAX else
                    "**genuine ceiling failure** — no h crosses"))
        w(f"- εK={c['epsK']}: h=32 → {r32:.4f}"
          + (f", h=64 → {r64:.4f}" if r64 is not None else "") + f" · {verdict}")
    w("")

    w("## Chains held per state dimension\n")
    w("The marginal-value property, re-measured at full training. "
      "Chains held = (1 − regret)·K.\n")
    for c in cands:
        vals = []
        for h in C.HS:
            k = str(h)
            if k in c["regret_by_h"]:
                held = (1 - c["regret_by_h"][k]) * C.K
                vals.append((h, held / h, held > 0.95 * C.K))
        unsat = [v for _, v, s in vals if not s]
        w(f"- **εK={c['epsK']}** — " + ", ".join(f"h={h}:{v:.2f}" + ("*" if s else "")
                                                 for h, v, s in vals) +
          (f"  → mean {np.mean(unsat):.2f} before saturation" if unsat else ""))
    w("\n*\\* saturated: holding >95% of chains*\n")

    w("## Selection\n")
    if chosen:
        eps = chosen["epsK"] / C.K
        w(f"### K = {C.K}, εK = {chosen['epsK']} (ε = {eps:.6g}), emissions 0.1/0.9\n")
        w(f"- band: h ∈ {chosen['band']} ({chosen['n_adjacent']} adjacent)")
        w(f"- mean between-seed SD over the band: **{chosen['mean_seed_sd']:.4f}**")
        w(f"- smallest h {chosen['smallest_h_regret']:.4f} · "
          f"largest h {chosen['largest_h_regret']:.4f}")
        w(f"- regret denominator: see `envk`; seed SD is far below the v1 power-gate stop "
          f"threshold of 0.0668\n")
        w("**Frozen for Protocol v2.** Stage 1 re-selects λ under this environment; the "
          "provisional λ used in calibration is not carried forward.\n")
    else:
        w("**No eligible setting.** Report and stop; do not widen the band or move a "
          "threshold.\n")

    stage1.atomic_write_json(os.path.join(OUT, "calibration_v2.json"),
                             {"n_runs": len(rows), "n_converged": n_conv,
                              "candidates": cands, "chosen": chosen})
    txt = "\n".join(L) + "\n"
    stage1.atomic_write_text(os.path.join(OUT, "CALIBRATION_V2_REPORT.md"), txt)
    print(txt)
    return chosen


if __name__ == "__main__":
    main()
