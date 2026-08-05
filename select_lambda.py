"""Select lambda per Protocol v2 §4, then hand off to Stage 2.

B2 only, at the primary horizon k=8, across the frozen band. Selection cannot favour the
treatment because no treatment condition is trained. Rule frozen here before any output:

  1. mean regret across the band h, per lambda
  2. lowest aggregate wins
  3. tie-break (within 0.005): prefer 0.1, then the smaller lambda
"""
import glob, json, os
import numpy as np
import stage1, stage2

OUT = os.path.join(stage2._HERE, "runs/lambda")

rows = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "*.json"))
        if os.path.basename(p) not in ("manifest.json", "lambda_selected.json")]
agg = {}
for lam in stage2.LAMBDA_SEL.lams:
    per_h = [np.mean([r["final_regret"] for r in rows
                      if r["lam"] == lam and r["h_dim"] == h])
             for h in stage2.LAMBDA_SEL.h_values]
    agg[lam] = float(np.mean(per_h))
    print(f"  lambda={lam:<6} mean regret over band = {agg[lam]:.4f}")

best = min(agg.values())
tied = [l for l, v in agg.items() if v - best < 0.005]
chosen = 0.1 if 0.1 in tied else min(tied)
print(f"\nselected lambda = {chosen}" + (" (tie-break)" if len(tied) > 1 else ""))

stage1.atomic_write_json(os.path.join(OUT, "lambda_selected.json"), {
    "selected_lambda": chosen, "aggregate_regret": agg, "tied": tied,
    "rule": "lowest mean regret over band; tie-break within 0.005 -> 0.1, then smaller",
    "n_runs": len(rows), "provenance": stage1.provenance()})
print("wrote runs/lambda/lambda_selected.json")
