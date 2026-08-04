"""Stage 1 analysis. Separate from execution, deterministic, no training.

Applies PROTOCOL.md §6.1 (lambda), §6.2 (capacity band), §6.4 (escape hatch condition),
and §6.3 (power gate) in that fixed order, then writes selection.json, power.json, and a
human-readable report.

Does NOT start Stage 2. Terminates after producing and validating the artifacts.

    python3 analyze1.py --config protocol
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import env
import power
import stage1

_HERE = stage1._HERE

# Frozen windows (PROTOCOL.md §6)
LAMBDA_WINDOW = (0.05, 0.95)  # §6.1 step 2
BAND_WINDOW = (0.20, 0.80)  # §6.2


def load(cfg):
    """Load every planned record. Returns (records_by_tag, manifest, problems)."""
    out_dir = os.path.join(_HERE, cfg.runs_dir)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)

    recs, problems = {}, []
    for h, lam, seed in stage1.jobs(cfg):
        t = stage1.tag(h, lam, seed)
        p = stage1.record_path(cfg, h, lam, seed)
        if not os.path.exists(p):
            problems.append((t, "missing"))
            continue
        with open(p) as f:
            rec = json.load(f)
        ok, why = stage1.validate_record(rec, cfg, rec.get("provenance", {}))
        if not ok:
            problems.append((t, why))
            continue
        if rec["status"] != "ok":
            problems.append((t, f"status={rec['status']}: {rec.get('failed')}"))
        recs[t] = rec
    return recs, manifest, problems


def regret_grid(cfg, recs):
    """{(lam, h): (mean, sd, n, [values])} over seeds, excluding non-ok runs."""
    grid = {}
    for lam in cfg.lambdas:
        for h in cfg.h_values:
            vals = []
            for seed in cfg.seeds:
                r = recs.get(stage1.tag(h, lam, seed))
                if r and r["status"] == "ok" and np.isfinite(r["final_regret"]):
                    vals.append(r["final_regret"])
            if vals:
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
                grid[(lam, h)] = (float(np.mean(vals)), sd, len(vals), vals)
    return grid


def select_lambda(cfg, grid):
    """§6.1 steps 2-4."""
    lo, hi = LAMBDA_WINDOW
    table = []
    for lam in cfg.lambdas:
        included = [h for h in cfg.h_values
                    if (lam, h) in grid and lo <= grid[(lam, h)][0] <= hi]
        agg = float(np.mean([grid[(lam, h)][0] for h in included])) if included else None
        table.append({"lambda": lam, "n_h_included": len(included),
                      "h_included": included, "aggregate_regret": agg})

    valid = [r for r in table if r["aggregate_regret"] is not None]
    if not valid:
        return None, table, "no lambda has any h inside the [0.05, 0.95] window"

    best = min(r["aggregate_regret"] for r in valid)
    tied = [r["lambda"] for r in valid if r["aggregate_regret"] - best < 0.01]
    chosen = 0.1 if 0.1 in tied else min(tied)  # §6.1 step 4
    return chosen, table, ("tie-break applied" if len(tied) > 1 else "unique minimum")


def find_band(cfg, grid, lam):
    """§6.2, using the selected lambda only."""
    lo, hi = BAND_WINDOW
    return [h for h in cfg.h_values if (lam, h) in grid and lo <= grid[(lam, h)][0] <= hi]


def hatch_check(cfg, grid, lam):
    """§6.4. Reports whether the pre-registered condition is met. Never executes it."""
    used = abs(cfg.run_cfg.env_cfg.p_low - 0.25) > 1e-12
    means = [grid[(lam, h)][0] for h in cfg.h_values if (lam, h) in grid]
    if not means:
        return {"condition": "no data", "eligible": False, "already_used": used}
    if max(means) < BAND_WINDOW[0]:
        cond, action = "task too easy (no h reaches regret 0.20)", "weaken emissions to 0.35/0.65"
    elif min(means) > BAND_WINDOW[1]:
        cond, action = "task too hard (all h exceed regret 0.80)", "strengthen emissions to 0.15/0.85"
    else:
        return {"condition": "not met", "eligible": False, "already_used": used}
    return {"condition": cond, "eligible": not used, "already_used": used,
            "action": action if not used else "escape hatch already used once -- STOP"}


def analyse(cfg):
    recs, manifest, problems = load(cfg)
    grid = regret_grid(cfg, recs)
    lam, lam_table, lam_note = select_lambda(cfg, grid)

    result = {
        "config": cfg.name,
        "written": time.time(),
        "provenance": stage1.provenance(),
        "manifest_planned": manifest["n_planned"],
        "records_ok": len(recs),
        "problems": [{"tag": t, "reason": w} for t, w in problems],
        "lambda_window": list(LAMBDA_WINDOW),
        "band_window": list(BAND_WINDOW),
        "lambda_table": lam_table,
        "lambda_note": lam_note,
        "selected_lambda": lam,
    }

    if lam is None:
        result.update({"band": [], "decision": "stop",
                       "message": "no valid lambda aggregate; " + lam_note,
                       "hatch": hatch_check(cfg, grid, cfg.lambdas[0])})
        return result, grid, None

    band = find_band(cfg, grid, lam)
    hatch = hatch_check(cfg, grid, lam)
    result.update({"band": band, "hatch": hatch})

    if len(band) < 2:  # §6.2
        if hatch["eligible"]:
            result.update({"decision": "escape_hatch",
                           "message": f"fewer than 2 h in band; {hatch['condition']}; "
                                      f"pre-registered action: {hatch['action']}"})
        else:
            result.update({"decision": "stop", "message": "no measurable capacity band"})
        return result, grid, None

    # §6.3 power gate, on the band, at the selected lambda.
    sigma = {h: grid[(lam, h)][1] for h in band}
    thin = [h for h in band if grid[(lam, h)][2] < len(cfg.seeds)]
    if any(not np.isfinite(s) for s in sigma.values()):
        result.update({"decision": "stop",
                       "message": "cannot estimate seed SD (a band h has fewer than 2 ok seeds)"})
        return result, grid, None

    gate = power.gate(sigma)
    gate.update({
        "config": cfg.name, "written": time.time(), "provenance": stage1.provenance(),
        "selected_lambda": lam, "band": band,
        "seeds_per_h": {str(h): grid[(lam, h)][2] for h in band},
        "incomplete_h": thin,
    })
    result["decision"] = "stop" if gate["decision"] == "stop" else "continue"
    result["stage2_seeds"] = gate["stage2_seeds"]
    result["message"] = gate["message"] or f"proceed to Stage 2 with {gate['stage2_seeds']} seeds"
    return result, grid, gate


# ------------------------------------------------------------------ report

def report(cfg, result, grid, gate):
    L = []
    w = L.append
    w("# Stage 1 Report\n")
    w(f"- config: `{cfg.name}`")
    p = result["provenance"]
    w(f"- commit: `{p['git_commit'][:12]}`{' **DIRTY**' if p['git_dirty'] else ''}")
    w(f"- protocol sha: `{p['protocol_sha']}`  code sha: `{p['code_sha']}`")
    w(f"- host: {p['hostname']}  torch {p['torch']}")
    w(f"- runs planned {result['manifest_planned']}, valid {result['records_ok']}, "
      f"problems {len(result['problems'])}\n")

    if result["problems"]:
        w("## Problems\n")
        for pr in result["problems"]:
            w(f"- `{pr['tag']}` — {pr['reason']}")
        w("")

    w("## §6.1 lambda selection\n")
    w(f"Aggregate = mean regret over h with mean regret in "
      f"[{LAMBDA_WINDOW[0]}, {LAMBDA_WINDOW[1]}].\n")
    w("| lambda | h included | n | aggregate regret |")
    w("|---|---|---|---|")
    for r in result["lambda_table"]:
        agg = "—" if r["aggregate_regret"] is None else f"{r['aggregate_regret']:.4f}"
        sel = " ⬅ **selected**" if r["lambda"] == result["selected_lambda"] else ""
        w(f"| {r['lambda']} | {r['h_included'] or '—'} | {r['n_h_included']} | {agg}{sel} |")
    w(f"\n{result['lambda_note']}. Selected λ = **{result['selected_lambda']}**\n")

    w("## Regret by hidden size (mean ± sd over seeds)\n")
    w("| h | " + " | ".join(f"λ={l}" for l in cfg.lambdas) + " |")
    w("|---" * (len(cfg.lambdas) + 1) + "|")
    for h in cfg.h_values:
        cells = []
        for lam in cfg.lambdas:
            if (lam, h) not in grid:
                cells.append("—")
                continue
            m, sd, n, _ = grid[(lam, h)]
            s = f"{m:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{m:.3f} ± n/a"
            cells.append(s + (f" (n={n})" if n != len(cfg.seeds) else ""))
        w(f"| {h} | " + " | ".join(cells) + " |")
    w("")

    w("## §6.2 capacity band\n")
    w(f"Window [{BAND_WINDOW[0]}, {BAND_WINDOW[1]}] at the selected λ. "
      f"Band = **{result['band'] or 'empty'}**\n")
    hatch = result.get("hatch", {})
    if hatch:
        w(f"Escape hatch (§6.4): condition *{hatch['condition']}*, "
          f"eligible={hatch['eligible']}, already used={hatch['already_used']}")
        if hatch.get("action"):
            w(f"→ {hatch['action']}")
        w("")

    if gate:
        w("## §6.3 power gate\n")
        w(f"*{gate['_note']}*\n")
        w(f"σ_plan = √2 · σ_B2. {gate['safety_factor_rationale']}\n")
        w("| h | σ_B2 | σ_plan | required n | MDE @10 | MDE @30 |")
        w("|---|---|---|---|---|---|")
        for r in gate["per_h"]:
            w(f"| {r['h']} | {r['sigma_b2']:.4f} | {r['sigma_delta_plan']:.4f} | "
              f"{r['required_n']} | {r['mde_at_min_seeds']:.4f} | {r['mde_at_max_seeds']:.4f} |")
        w(f"\nMax required n = {gate['max_required_n']}, rounded up to "
          f"{gate['rounded_up_even']} → **{gate['decision']}**")
        if gate["incomplete_h"]:
            w(f"\n**WARNING** — h {gate['incomplete_h']} have fewer than "
              f"{len(cfg.seeds)} ok seeds; their σ is estimated from a short sample.")
        w("")

    w("## Decision\n")
    w(f"**{result['decision'].upper()}** — {result['message']}\n")
    if result["decision"] == "continue":
        w(f"Stage 2 runs at h {result['band']}, λ = {result['selected_lambda']}, "
          f"{result['stage2_seeds']} paired seeds per condition.")
    w("\nStage 1 ends here. Stage 2 is not started automatically.")
    return "\n".join(L) + "\n"


def main(cfg):
    result, grid, gate = analyse(cfg)
    out = os.path.join(_HERE, cfg.runs_dir)

    stage1.atomic_write_json(os.path.join(out, "selection.json"), result)
    if gate:
        stage1.atomic_write_json(os.path.join(out, "power.json"), gate)
    text = report(cfg, result, grid, gate)
    stage1.atomic_write_text(os.path.join(out, "STAGE1_REPORT.md"), text)

    print(text)
    print(f"wrote {os.path.relpath(out, _HERE)}/selection.json"
          + (", power.json" if gate else "") + ", STAGE1_REPORT.md")
    return 0 if result["decision"] in ("continue", "stop", "escape_hatch") else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(stage1.CONFIGS))
    a = ap.parse_args()
    sys.exit(main(stage1.CONFIGS[a.config]))
