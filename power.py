"""Stage 2 seed-count power gate. Implements PROTOCOL.md §6.3 exactly.

PLANNING APPROXIMATION, NOT EXACT POWER.
The final hypothesis test is Wilcoxon signed-rank. This module computes power for a
paired t-test assuming approximately normal paired differences. It is used to choose a
seed count, never to report a result.

Stage 1 trains B2 only and therefore never observes a paired A-B2 difference. The
planning SD is sqrt(2) * sigma_B2 -- the value that would hold if A and B2 were
independent with equal variance. Paired seeds should make them positively correlated,
which makes the true paired SD smaller, so this is conservative by construction.
"""

import json
import math
import sys

from scipy import stats

# Frozen constants (PROTOCOL.md §6.3)
TARGET_EFFECT = 0.05
TARGET_POWER = 0.80
ALPHA = 0.05
MIN_SEEDS = 10
MAX_SEEDS = 30
SAFETY_FACTOR = math.sqrt(2.0)

_SEARCH_CAP = 500


# scipy's nct.cdf underflows to NaN in the far lower tail for noncentrality above
# roughly 20, and does so patchily (NaN at nc=20,30,35,45 but finite at 40,70,100).
# Since `NaN >= target` is False, an unguarded call silently breaks the monotonicity
# that every search here assumes. Above this noncentrality power is 1.0 to double
# precision (at nc=10, df=9 it is already 1 - 8.3e-12), so clamping is exact.
_NC_SATURATED = 10.0


def _power_from_nc(df: int, nc: float, alpha: float = ALPHA) -> float:
    """Two-sided t-test power at a given noncentrality. Monotone increasing in nc."""
    if nc >= _NC_SATURATED:
        return 1.0
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    upper = float(stats.nct.cdf(tcrit, df, nc))
    lower = float(stats.nct.cdf(-tcrit, df, nc))
    # Defensive: any residual underflow is a vanishing tail, not a missing value.
    if not math.isfinite(upper):
        upper = 0.0
    if not math.isfinite(lower):
        lower = 0.0
    return 1.0 - upper + lower


def power_at(n: int, sigma_delta: float, effect: float = TARGET_EFFECT,
             alpha: float = ALPHA) -> float:
    """Two-sided paired t-test power via the noncentral t distribution."""
    if n < 2:
        return 0.0
    return _power_from_nc(n - 1, effect * math.sqrt(n) / sigma_delta, alpha)


def required_n(sigma_delta: float, effect: float = TARGET_EFFECT,
               target: float = TARGET_POWER, alpha: float = ALPHA) -> int:
    """Smallest n reaching target power. Returns _SEARCH_CAP if unreachable."""
    for n in range(2, _SEARCH_CAP + 1):
        if power_at(n, sigma_delta, effect, alpha) >= target:
            return n
    return _SEARCH_CAP


def next_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def mde_at(n: int, sigma_delta: float, target: float = TARGET_POWER,
           alpha: float = ALPHA) -> float:
    """Smallest effect detectable at n with the target power. Reported, not used to decide.

    Inverts on noncentrality rather than on effect: nc* depends only on (df, alpha,
    target) and lives in a small, well-conditioned range, so nct is never evaluated
    where it underflows. The conversion to an effect size is then exact algebra.
    """
    if n < 2:
        return float("inf")
    df = n - 1
    lo, hi = 0.0, _NC_SATURATED
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _power_from_nc(df, mid, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return hi * sigma_delta / math.sqrt(n)


def stop_threshold() -> float:
    """The sigma_B2 above which the gate stops. Diagnostic; derived from the frozen rule."""
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if next_even(required_n(SAFETY_FACTOR * mid)) > MAX_SEEDS:
            hi = mid
        else:
            lo = mid
    return lo


def gate(sigma_b2_by_h: dict) -> dict:
    """Apply §6.3. `sigma_b2_by_h` maps band h -> between-seed SD of final B2 regret.

    Returns the record for power.json.
    """
    assert sigma_b2_by_h, "power gate needs at least one band h"

    table = []
    for h in sorted(sigma_b2_by_h):
        s_b2 = float(sigma_b2_by_h[h])
        s_plan = SAFETY_FACTOR * s_b2
        n_req = required_n(s_plan)
        table.append({
            "h": h,
            "sigma_b2": s_b2,
            "sigma_delta_plan": s_plan,
            "required_n": n_req,
            "mde_at_min_seeds": mde_at(MIN_SEEDS, s_plan),
            "mde_at_max_seeds": mde_at(MAX_SEEDS, s_plan),
        })

    worst = max(r["required_n"] for r in table)  # §6.3 step 4: max across band h
    rounded = next_even(worst)  # step 5

    if rounded <= MIN_SEEDS:
        decision, seeds = "keep_minimum", MIN_SEEDS
    elif rounded <= MAX_SEEDS:
        decision, seeds = "raise_seeds", rounded
    else:
        decision, seeds = "stop", None

    return {
        "_note": "PLANNING APPROXIMATION (paired t-test). The reported test is Wilcoxon "
                 "signed-rank; this is not exact power for Wilcoxon.",
        "target_effect": TARGET_EFFECT,
        "target_power": TARGET_POWER,
        "alpha": ALPHA,
        "min_seeds": MIN_SEEDS,
        "max_seeds": MAX_SEEDS,
        "safety_factor": SAFETY_FACTOR,
        "safety_factor_rationale": "Stage 1 observes B2 only; sqrt(2)*sigma_B2 assumes A and "
                                   "B2 independent with equal variance. Positive pairing "
                                   "correlation would make the true paired SD smaller.",
        "per_h": table,
        "max_required_n": worst,
        "rounded_up_even": rounded,
        "decision": decision,
        "stage2_seeds": seeds,
        "message": (None if decision != "stop" else
                    "No feasible powered comparison under the pre-registered compute cap."),
    }


def _check_monotone():
    """Regression guard for the scipy nct NaN bug.

    That bug was silent: it produced plausible-looking numbers with no error, and was
    only visible because MDE came out non-monotonic in n. Assert the property directly.
    """
    grid = [1e-4 * 1.15 ** k for k in range(120)]  # spans nc from ~0.02 to ~1e5
    for n in (2, 3, 10, 30, 100):
        vals = [power_at(n, 0.05, e) for e in grid]
        assert all(math.isfinite(v) for v in vals), f"non-finite power at n={n}"
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), f"non-monotone at n={n}"

    # MDE must fall with more seeds and scale linearly with sigma.
    assert mde_at(10, 0.05) > mde_at(30, 0.05)
    assert abs(mde_at(10, 0.10) - 2 * mde_at(10, 0.05)) < 1e-9

    # ...and agree with the normal approximation, MDE ~ (z_{1-a/2} + z_{1-b}) * sigma / sqrt(n).
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    approx = z * 0.05 / math.sqrt(30)
    assert abs(mde_at(30, 0.05) - approx) / approx < 0.15, "MDE far from closed form"
    print("  monotonicity + closed-form cross-check OK")


def _self_test():
    """Exercises all three branches before any Stage 1 data exists."""
    print("power gate self-test (PROTOCOL.md §6.3)\n")
    _check_monotone()
    thr = stop_threshold()
    print(f"  sigma_B2 above {thr:.4f} triggers STOP (n>{MAX_SEEDS} required)\n")
    print(f"  {'sigma_B2':>9}  {'sigma_plan':>10}  {'req n':>6}  {'MDE@10':>7}  {'MDE@30':>7}  decision")
    for s in (0.010, 0.020, 0.040, 0.055, 0.069, 0.080, 0.120):
        g = gate({4: s})
        r = g["per_h"][0]
        print(f"  {s:9.3f}  {r['sigma_delta_plan']:10.4f}  {r['required_n']:6d}  "
              f"{r['mde_at_min_seeds']:7.4f}  {r['mde_at_max_seeds']:7.4f}  "
              f"{g['decision']}{'' if g['stage2_seeds'] is None else ' -> ' + str(g['stage2_seeds'])}")

    g = gate({2: 0.03, 3: 0.05, 4: 0.02})
    assert g["stage2_seeds"] == next_even(max(r["required_n"] for r in g["per_h"])), "max rule"
    print(f"\n  multi-h {{2:0.03, 3:0.05, 4:0.02}} -> worst h drives it, seeds={g['stage2_seeds']}")
    print("\n  self-test OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Later: read Stage-1 records, compute sigma per band h, write power.json.
        raise SystemExit("Stage-1 driver does not exist yet; run without arguments to self-test.")
    _self_test()
