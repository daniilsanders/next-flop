"""Stage 1 analysis. Separate from execution, deterministic, no training.

Applies PROTOCOL.md §6.1 (lambda), §6.2 (capacity band), §6.4 (escape hatch condition),
and §6.3 (power gate) in that fixed order, then writes selection.json, power.json, and a
human-readable report.

Does NOT start Stage 2. Terminates after producing and validating the artifacts.

    python3 analyze1.py --config protocol
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

import env
import power
import stage1

_HERE = stage1._HERE

# Frozen windows (PROTOCOL.md §6)
LAMBDA_WINDOW = (0.05, 0.95)  # §6.1 step 2
BAND_WINDOW = (0.20, 0.80)  # §6.2


REWRITE_TOLERANCE_S = 5.0  # a record's mtime should match the moment it was written


def scan(cfg):
    """Inspect every planned run on disk. Returns (rows, manifest).

    One row per planned job whether or not it exists, each carrying the file checksum
    and the facts needed for the completion manifest and integrity checks.
    """
    out_dir = os.path.join(_HERE, cfg.runs_dir)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)

    rows = []
    for h, lam, seed in stage1.jobs(cfg):
        t = stage1.tag(h, lam, seed)
        p = stage1.record_path(cfg, h, lam, seed)
        row = {"tag": t, "h": h, "lambda": lam, "seed": seed, "exists": os.path.exists(p),
               "valid": False, "status": None, "reason": None, "sha256": None,
               "mtime": None, "ended": None, "rewritten_after_write": None,
               "final_regret": None, "runtime_s": None, "used_in_analysis": False,
               "record": None}
        if not row["exists"]:
            row["reason"] = "missing"
            rows.append(row)
            continue

        with open(p, "rb") as f:
            raw = f.read()
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        row["mtime"] = os.path.getmtime(p)
        try:
            rec = json.loads(raw)
        except Exception as e:
            row["reason"] = f"unreadable: {e}"
            rows.append(row)
            continue

        ok, why = stage1.validate_record(rec, cfg, rec.get("provenance", {}))
        row.update({"valid": ok, "reason": why or None, "status": rec.get("status"),
                    "ended": rec.get("ended"), "final_regret": rec.get("final_regret"),
                    "runtime_s": rec.get("runtime_s"), "record": rec})
        if row["ended"]:
            # os.replace preserves the temp file's mtime, so mtime is the write moment.
            row["rewritten_after_write"] = (row["mtime"] - row["ended"]) > REWRITE_TOLERANCE_S
        row["used_in_analysis"] = bool(ok and rec.get("status") == "ok")
        rows.append(row)
    return rows, manifest


def load(cfg):
    """Records usable by the analysis, plus the manifest and the problem list."""
    rows, manifest = scan(cfg)
    recs, problems = {}, []
    for r in rows:
        if not r["exists"]:
            problems.append((r["tag"], "missing"))
        elif not r["valid"]:
            problems.append((r["tag"], r["reason"]))
        else:
            if r["status"] != "ok":
                problems.append((r["tag"],
                                 f"status={r['status']}: {r['record'].get('failed')}"))
            recs[r["tag"]] = r["record"]
    return recs, manifest, problems, rows


def _sha_at_commit(commit, path):
    import subprocess
    try:
        raw = subprocess.check_output(["git", "-C", _HERE, "show", f"{commit}:{path}"],
                                      stderr=subprocess.DEVNULL)
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return None


def verify_code_matched_commit(mp):
    """Was the code that actually ran identical to the training commit?

    `git_dirty` counts any modified or untracked file, including ones with no bearing on
    results. This reconstructs protocol_sha and code_sha from the blobs at the training
    commit and compares them to what the driver recorded, which answers the question the
    dirty flag only gestures at.
    """
    commit = mp.get("git_commit")
    per_file = {f: _sha_at_commit(commit, f) for f in stage1.CODE_FILES}
    if any(v is None for v in per_file.values()):
        return {"checked": False, "reason": "could not read blobs at the training commit",
                "per_file": per_file}
    code = hashlib.sha256(
        "".join(per_file[f] for f in stage1.CODE_FILES if f != "PROTOCOL.md").encode()
    ).hexdigest()[:16]
    return {
        "checked": True,
        "per_file_sha_at_commit": per_file,
        "protocol_sha_matches": per_file["PROTOCOL.md"] == mp.get("protocol_sha"),
        "code_sha_matches": code == mp.get("code_sha"),
        "all_result_determining_files_matched_commit":
            per_file["PROTOCOL.md"] == mp.get("protocol_sha") and code == mp.get("code_sha"),
    }


def integrity(cfg, rows, manifest):
    """Provenance and tamper checks across the whole sweep."""
    mp = manifest["provenance"]
    present = [r for r in rows if r["record"]]

    mismatched = [r["tag"] for r in present
                  if any(r["record"]["provenance"].get(k) != mp.get(k)
                         for k in ("git_commit", "protocol_sha", "code_sha"))]
    rewritten = [r["tag"] for r in present if r["rewritten_after_write"]]

    # Worker count was not recorded by the driver, so measure it instead: the largest
    # number of runs whose [started, ended] intervals overlap at any instant.
    events = []
    for r in present:
        rec = r["record"]
        if rec.get("started") and rec.get("ended"):
            events += [(rec["started"], 1), (rec["ended"], -1)]
    cur = peak = 0
    for _, d in sorted(events):
        cur += d
        peak = max(peak, cur)

    used = [r for r in rows if r["used_in_analysis"]]
    digest = hashlib.sha256(
        "".join(f"{r['tag']}:{r['sha256']}" for r in sorted(used, key=lambda r: r["tag"]))
        .encode()).hexdigest()

    return {
        "training_provenance": mp,
        "code_vs_commit": verify_code_matched_commit(mp),
        "analysis_environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
        },
        "observed_max_concurrency": peak,
        "worker_count_recorded_by_driver": False,
        "n_planned": len(rows),
        "n_present": len(present),
        "n_missing": sum(1 for r in rows if not r["exists"]),
        "n_valid": sum(1 for r in rows if r["valid"]),
        "n_invalid": sum(1 for r in rows if r["exists"] and not r["valid"]),
        "n_failed": sum(1 for r in rows if r["status"] == "failed"),
        "n_used_in_analysis": len(used),
        "provenance_mismatches": mismatched,
        "records_rewritten_after_launch": rewritten,
        "no_record_rewritten": not rewritten,
        "analysis_input_digest": digest,
    }


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
    """§6.4. Reports whether the pre-registered condition is met. Never executes it.

    lam=None means no lambda was selectable (§6.1 found no valid aggregate). In that case
    the condition is evaluated over every (lambda, h) mean: the task is too easy only if
    nothing anywhere reaches the band floor.
    """
    used = abs(cfg.run_cfg.env_cfg.p_low - 0.25) > 1e-12
    lams = cfg.lambdas if lam is None else [lam]
    means = [grid[(l, h)][0] for l in lams for h in cfg.h_values if (l, h) in grid]
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
    recs, manifest, problems, rows = load(cfg)
    integ = integrity(cfg, rows, manifest)
    grid = regret_grid(cfg, recs)
    lam, lam_table, lam_note = select_lambda(cfg, grid)

    result = {
        "config": cfg.name,
        "written": time.time(),
        "provenance": stage1.provenance(),
        "integrity": integ,
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
        # §6.1: "If no lambda has a valid aggregate, the escape hatch (§6.4) applies."
        hatch = hatch_check(cfg, grid, None)
        result.update({"band": [], "hatch": hatch})
        if hatch["eligible"]:
            result.update({"decision": "escape_hatch",
                           "message": f"no valid lambda aggregate ({lam_note}); "
                                      f"{hatch['condition']}; pre-registered action: "
                                      f"{hatch['action']}"})
        else:
            result.update({"decision": "stop",
                           "message": "no valid lambda aggregate; " + lam_note})
        return result, grid, None, rows

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
        return result, grid, None, rows

    # §6.3 power gate, on the band, at the selected lambda.
    sigma = {h: grid[(lam, h)][1] for h in band}
    thin = [h for h in band if grid[(lam, h)][2] < len(cfg.seeds)]
    if any(not np.isfinite(s) for s in sigma.values()):
        result.update({"decision": "stop",
                       "message": "cannot estimate seed SD (a band h has fewer than 2 ok seeds)"})
        return result, grid, None, rows

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
    return result, grid, gate, rows


# ------------------------------------------------------------------ report

def report(cfg, result, grid, gate):
    L = []
    w = L.append
    i = result["integrity"]
    tp, ae = i["training_provenance"], i["analysis_environment"]

    w("# Stage 1 Report\n")
    w(f"- config: `{cfg.name}`")
    w(f"- runs planned {result['manifest_planned']}, used in analysis "
      f"{i['n_used_in_analysis']}, problems {len(result['problems'])}\n")

    w("## Provenance\n")
    cvc = i["code_vs_commit"]
    w(f"- **training commit:** `{tp['git_commit']}`")
    w(f"- **working tree at launch:** {'DIRTY' if tp['git_dirty'] else 'clean'}")
    if cvc.get("checked"):
        ok = cvc["all_result_determining_files_matched_commit"]
        w(f"- **code that ran vs. that commit:** "
          f"{'IDENTICAL — every result-determining file matched' if ok else '**MISMATCH**'} "
          f"(protocol_sha {'ok' if cvc['protocol_sha_matches'] else 'DIFFERS'}, "
          f"code_sha {'ok' if cvc['code_sha_matches'] else 'DIFFERS'})")
        if tp["git_dirty"] and ok:
            w("  - the dirty flag counts any modified or untracked file; none of them "
              "were inputs to training")
    else:
        w(f"- **code vs. commit:** not verified — {cvc.get('reason')}")
    w(f"- protocol sha `{tp['protocol_sha']}`, code sha `{tp['code_sha']}`")
    w(f"- training env: Python {tp['python']}, PyTorch {tp['torch']}, NumPy {tp['numpy']}")
    w(f"- analysis env: Python {ae['python']}, NumPy {ae['numpy']}, SciPy {ae['scipy']}")
    w(f"- platform: {tp['platform']} (host `{tp['hostname']}`)")
    w(f"- observed max concurrency: **{i['observed_max_concurrency']} workers** "
      f"(measured from run intervals; the driver does not record the worker count — "
      f"fix before Stage 2)")
    w(f"- runs: {i['n_present']}/{i['n_planned']} present, {i['n_valid']} valid, "
      f"**{i['n_invalid']} invalid**, **{i['n_failed']} failed**, {i['n_missing']} missing")
    w(f"- provenance mismatches across records: "
      f"{i['provenance_mismatches'] or 'none — all records share one commit and code sha'}")
    w(f"- **no completed record rewritten after launch:** {i['no_record_rewritten']}"
      + ("" if i["no_record_rewritten"] else f" — {i['records_rewritten_after_launch']}"))
    w(f"- analysis input digest (sha256 over the records used): `{i['analysis_input_digest']}`")
    w(f"- per-record checksums: `completion.json`\n")

    w("## Validation coverage\n")
    w("The driver was exercised end-to-end through config-only variants: resume after "
      "SIGKILL, atomic writes, corrupt-record halt, failure recording without retry, "
      "λ tie-break, empty-band stop, and power-gate generation.\n")
    w("**Not exercised end-to-end:** the *continue* branch (a band found with required "
      "n ≤ 30). No smoke configuration produced a band with low enough seed variance to "
      "reach it, and no synthetic success case was fabricated to cover it. The gate "
      "arithmetic underlying that branch is unit-tested across all three outcomes in "
      "`power.py`.\n")

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
    result, grid, gate, rows = analyse(cfg)
    out = os.path.join(_HERE, cfg.runs_dir)

    # Raw completion manifest: every planned run, its status, and its checksum.
    stage1.atomic_write_json(os.path.join(out, "completion.json"), {
        "config": cfg.name,
        "written": time.time(),
        "integrity": result["integrity"],
        "runs": [{k: v for k, v in r.items() if k != "record"} for r in rows],
    })
    stage1.atomic_write_json(os.path.join(out, "selection.json"), result)
    if gate:
        stage1.atomic_write_json(os.path.join(out, "power.json"), gate)
    text = report(cfg, result, grid, gate)
    stage1.atomic_write_text(os.path.join(out, "STAGE1_REPORT.md"), text)

    print(text)
    print(f"wrote {os.path.relpath(out, _HERE)}/: completion.json, selection.json"
          + (", power.json" if gate else "") + ", STAGE1_REPORT.md")
    return 0 if result["decision"] in ("continue", "stop", "escape_hatch") else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(stage1.CONFIGS))
    a = ap.parse_args()
    sys.exit(main(stage1.CONFIGS[a.config]))
