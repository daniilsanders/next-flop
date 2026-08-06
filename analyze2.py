"""Stage 2 analysis. Applies PROTOCOL_V2.md §8 exactly. No judgement, no new thresholds.

Improvement is defined so positive means the treatment is BETTER (regret is lower is
better):  improvement = regret_control - regret_A.

    python3 analyze2.py
"""

import glob
import json
import os

import numpy as np
from scipy import stats

import stage1
import stage2

RUNS = os.path.join(stage2._HERE, "runs/stage2")
BAND = stage2.STAGE2.h_values
WITHIN_EFFECT = 0.05  # §7: A vs B2, A vs B3
INTERACTION_EFFECT = 0.03  # §7: difference-of-differences


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def load():
    return [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "*.json"))
            if "manifest" not in p and "power_v2" not in p]


def cell(rows, cond, k, h):
    return {r["seed"]: r["final_regret"] for r in rows
            if r["condition"] == cond and r["horizon"] == k and r["h_dim"] == h}


def paired(rows, treat, ctrl, k, h):
    """regret_ctrl - regret_A, paired by seed. Positive => treatment better."""
    a, c = cell(rows, treat, k, h), cell(rows, ctrl, k, h)
    seeds = sorted(set(a) & set(c))
    return np.array([c[s] - a[s] for s in seeds]), seeds


def within_horizon(rows, ctrl, k):
    """§8 primary rule against one control at one horizon."""
    out, praw = [], []
    for h in BAND:
        d, seeds = paired(rows, "A", ctrl, k, h)
        p = stats.wilcoxon(d, alternative="two-sided").pvalue if np.any(d != 0) else 1.0
        praw.append(p)
        out.append({"h": h, "n": len(seeds), "mean_improvement": float(d.mean()),
                    "sd": float(d.std(ddof=1)), "p_raw": float(p)})
    for r, pa in zip(out, holm(praw)):
        r["p_holm"] = float(pa)
    n_pos = sum(1 for r in out if r["mean_improvement"] > 0)
    declared = any(r["p_holm"] < 0.05 and r["mean_improvement"] >= WITHIN_EFFECT
                   for r in out) and n_pos >= len(BAND) / 2
    return {"control": ctrl, "horizon": k, "per_h": out,
            "n_positive_sign": n_pos, "declared": declared}


def interaction(rows, ctrl):
    """§8 secondary: (A-ctrl)_{k=8} > (A-ctrl)_{k=1}, paired by seed."""
    out, praw = [], []
    for h in BAND:
        d8, s8 = paired(rows, "A", ctrl, 8, h)
        d1, s1 = paired(rows, "A", ctrl, 1, h)
        assert s8 == s1
        dd = d8 - d1
        p = stats.wilcoxon(dd, alternative="two-sided").pvalue if np.any(dd != 0) else 1.0
        praw.append(p)
        out.append({"h": h, "mean_dd": float(dd.mean()), "sd": float(dd.std(ddof=1)),
                    "improvement_k8": float(d8.mean()), "improvement_k1": float(d1.mean()),
                    "p_raw": float(p)})
    for r, pa in zip(out, holm(praw)):
        r["p_holm"] = float(pa)
    declared = any(r["p_holm"] < 0.05 and r["mean_dd"] >= INTERACTION_EFFECT for r in out)
    return {"control": ctrl, "per_h": out, "declared": declared}


def interpret(a_b2, a_b3, i_b2, i_b3):
    """§8 interpretation table, fixed before results existed."""
    if not a_b2["declared"] and not a_b3["declared"]:
        return ("NULL", "delta_s feedback does nothing on a calibrated task where it had "
                "room to succeed.")
    if a_b2["declared"] and not a_b3["declared"]:
        return ("CONFOUND", "A structured extra recurrent input helps; self-ness does not.")
    if a_b2["declared"] and a_b3["declared"]:
        if i_b2["declared"] or i_b3["declared"]:
            return ("HYPOTHESIS SUPPORTED",
                    "Self-prediction error helps, and the advantage grows with horizon.")
        return ("PARTIAL", "Self-prediction error helps, but not because of the horizon.")
    return ("UNEXPECTED", "A beats B3 but not B2; report as-is.")


def main():
    rows = load()
    ok = [r for r in rows if r["status"] == "ok"]
    a_b2 = within_horizon(ok, "B2", 8)
    a_b3 = within_horizon(ok, "B3", 8)
    a_b2_k1 = within_horizon(ok, "B2", 1)
    i_b2, i_b3 = interaction(ok, "B2"), interaction(ok, "B3")
    verdict, gloss = interpret(a_b2, a_b3, i_b2, i_b3)

    L, w = [], None
    w = L.append
    w("# Stage 2 Results\n")
    w("Decision rule applied exactly as frozen in `PROTOCOL_V2.md` §8. Wilcoxon "
      "signed-rank, two-sided, paired by seed; Holm–Bonferroni across band-h. Improvement "
      "is `regret_control − regret_A`, so **positive means A is better**.\n")
    p = ok[0]["provenance"]
    w(f"- runs: {len(ok)}/320 ok, {len(rows) - len(ok)} failed · all converged · "
      f"λ = {ok[0]['lam']} · FIFO mismatches "
      f"{sum(1 for r in ok if r['first_consume'] != r['expected_first_consume'])}")
    w(f"- commit `{p['git_commit'][:12]}` · band h ∈ {list(BAND)} · 10 paired seeds\n")

    w("## Cell means (regret, lower is better)\n")
    w("| h | " + " | ".join(f"{c} k=1 | {c} k=8" for c in ("B1", "B2", "B3", "A")) + " |")
    w("|---|" + "---|" * 8)
    for h in BAND:
        cells = []
        for c in ("B1", "B2", "B3", "A"):
            for k in (1, 8):
                v = list(cell(ok, c, k, h).values())
                cells.append(f"{np.mean(v):.4f}")
        w(f"| {h} | " + " | ".join(cells) + " |")
    w("")

    for name, res, thr in (("PRIMARY — A vs B2 at k=8", a_b2, WITHIN_EFFECT),
                           ("GATE — A vs B3 at k=8", a_b3, WITHIN_EFFECT),
                           ("reference — A vs B2 at k=1 (known-negative arm)",
                            a_b2_k1, WITHIN_EFFECT)):
        w(f"## {name}\n")
        w("| h | mean improvement | sd | p (raw) | p (Holm) | ≥ threshold |")
        w("|---|---|---|---|---|---|")
        for r in res["per_h"]:
            w(f"| {r['h']} | {r['mean_improvement']:+.4f} | {r['sd']:.4f} "
              f"| {r['p_raw']:.4f} | {r['p_holm']:.4f} "
              f"| {'yes' if r['mean_improvement'] >= thr else 'no'} |")
        w(f"\n**Declared: {res['declared']}** "
          f"(needs Holm p < 0.05 AND improvement ≥ {thr} at ≥1 h, "
          f"AND same sign at ≥ half the band; positive at "
          f"{res['n_positive_sign']}/{len(BAND)})\n")

    w("## Interactions (secondary family, threshold 0.03)\n")
    for res in (i_b2, i_b3):
        w(f"### (A − {res['control']})_k=8  >  (A − {res['control']})_k=1\n")
        w("| h | improvement k=8 | improvement k=1 | difference | p (Holm) |")
        w("|---|---|---|---|---|")
        for r in res["per_h"]:
            w(f"| {r['h']} | {r['improvement_k8']:+.4f} | {r['improvement_k1']:+.4f} "
              f"| {r['mean_dd']:+.4f} | {r['p_holm']:.4f} |")
        w(f"\n**Declared: {res['declared']}**\n")

    w("## Verdict\n")
    w(f"### {verdict}\n")
    w(f"{gloss}\n")

    out = {"verdict": verdict, "gloss": gloss, "a_vs_b2_k8": a_b2, "a_vs_b3_k8": a_b3,
           "a_vs_b2_k1": a_b2_k1, "interaction_b2": i_b2, "interaction_b3": i_b3,
           "n_ok": len(ok), "provenance": stage1.provenance()}
    stage1.atomic_write_json(os.path.join(RUNS, "results.json"), out)
    txt = "\n".join(L) + "\n"
    stage1.atomic_write_text(os.path.join(RUNS, "STAGE2_RESULTS.md"), txt)
    print(txt)
    return out


if __name__ == "__main__":
    main()
