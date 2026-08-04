"""Validation for env.py. Run before anything else exists.

If the environment or the reference filter is wrong, every later run looks
scientific and measures nothing. These checks are the only thing standing
between the protocol and that outcome.

    python3 test_env.py

Writes reference.json on success.
"""

import json
import sys

import numpy as np

import env
from env import EnvConfig

CFG = EnvConfig()
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def close(a, b, rel=0.10):
    return abs(a - b) <= rel * abs(b)


# ---------------------------------------------------------------- 1. generative

def test_frequencies():
    print("\n1. Empirical latent transition frequencies (256 seqs x 8192 steps)")
    d = env.generate(256, 8192, np.random.default_rng(0), CFG)
    x, z, v = d["x"], d["z"], d["v"]

    v_flip = float((v[:, 1:] != v[:, :-1]).mean())
    check("v flip rate", close(v_flip, CFG.p_v), f"{v_flip:.5f} vs {CFG.p_v}")

    # A z-flip at index t used eps(v[t]) -- the *new* v, per the fixed transition order.
    z_flip = z[:, 1:] != z[:, :-1]
    v_at = v[:, 1:]
    calm = float(z_flip[v_at == 0].mean())
    vol = float(z_flip[v_at == 1].mean())
    check("z flip rate | calm", close(calm, CFG.eps_calm), f"{calm:.5f} vs {CFG.eps_calm}")
    check("z flip rate | volatile", close(vol, CFG.eps_volatile), f"{vol:.5f} vs {CFG.eps_volatile}")

    p0 = float(x[z == 0].mean())
    p1 = float(x[z == 1].mean())
    check("P(x=1 | z=0)", close(p0, CFG.p_low, 0.01), f"{p0:.5f} vs {CFG.p_low}")
    check("P(x=1 | z=1)", close(p1, CFG.p_high, 0.01), f"{p1:.5f} vs {CFG.p_high}")

    mz, mv = float(z.mean()), float(v.mean())
    check("marginal P(z=1) ~ 0.5", abs(mz - 0.5) < 0.05, f"{mz:.4f}")
    check("marginal P(v=volatile) ~ 0.5", abs(mv - 0.5) < 0.05, f"{mv:.4f}")


# ---------------------------------------------------------------- 2. filter

def test_filter_against_brute_force():
    print("\n2. Bayes filter vs independent path enumeration (T=7)")
    rng = np.random.default_rng(1)
    cases = [rng.integers(0, 2, size=7).astype(np.int8) for _ in range(6)]
    cases += [np.zeros(7, np.int8), np.ones(7, np.int8)]  # edge cases

    worst = 0.0
    for xs in cases:
        got = env.bayes_predict(xs[None, :], CFG)[0]
        want = env.brute_force_predict(xs, CFG)
        worst = max(worst, float(np.abs(got - want).max()))
    check("filter matches enumeration", worst < 1e-12, f"max abs diff {worst:.2e}")

    # Transition matrix rows are already asserted inside env; confirm the chain is
    # symmetric under the z<->1-z relabelling, which is what makes the marginal 0.5.
    A = env.transition_matrix(CFG)
    perm = [1, 0, 3, 2]  # flip z, keep v
    check("transition symmetric under z-flip", np.allclose(A, A[np.ix_(perm, perm)]))


# ---------------------------------------------------------------- 3. references

def test_references():
    print("\n3. Reference losses on the held-out set (64 x 4096, burn-in 512)")
    d = env.heldout(CFG)
    x, z = d["x"], d["z"]
    n_scored = (x.shape[1] - 1 - env.BURN_IN) * x.shape[0]

    frozen = env.mean_loss(env.frozen_predict(x), x)
    oracle = env.mean_loss(env.oracle_predict(z, CFG), x)
    bayes = env.mean_loss(env.bayes_predict(x, CFG), x)

    h_oracle = -(CFG.p_high * np.log(CFG.p_high) + CFG.p_low * np.log(CFG.p_low))

    check("scored predictions == 3584/seq", n_scored == 64 * 3584, f"{n_scored} total")
    # The frozen predictor emits 0.5, so its loss is exactly ln 2 for either outcome.
    check("frozen == ln 2 exactly", abs(frozen - env.LN2) < 1e-12, f"{frozen:.6f}")
    check("oracle == H(p_high)", abs(oracle - h_oracle) < 5e-3, f"{oracle:.6f} vs {h_oracle:.6f}")
    check("oracle < bayes", oracle < bayes, f"{oracle:.6f} < {bayes:.6f}")
    check("bayes < frozen", bayes < frozen, f"{bayes:.6f} < {frozen:.6f}")

    print("\n4. Normalized regret")
    r_bayes = env.regret(bayes, bayes)
    r_frozen = env.regret(frozen, bayes)
    r_oracle = env.regret(oracle, bayes)
    check("regret(bayes) == 0", abs(r_bayes) < 1e-12, f"{r_bayes:.2e}")
    check("regret(frozen) == 1", abs(r_frozen - 1.0) < 1e-12, f"{r_frozen:.6f}")
    check("regret(oracle) < 0", r_oracle < 0, f"{r_oracle:.4f} (below Bayes, as expected)")

    return {"frozen": frozen, "oracle": oracle, "bayes": bayes, "n_scored": n_scored}


# ---------------------------------------------------------------- main

def main():
    print("=" * 68)
    print("Experiment 1 -- environment and reference filter validation")
    print("=" * 68)

    test_frequencies()
    test_filter_against_brute_force()
    refs = test_references()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        print("Do not proceed to the model.")
        return 1

    span = env.LN2 - refs["bayes"]
    print("Reference lines (nats)")
    print(f"  frozen (best constant)  {refs['frozen']:.6f}   regret 1.000")
    print(f"  Bayes-optimal           {refs['bayes']:.6f}   regret 0.000")
    print(f"  oracle (knows z)        {refs['oracle']:.6f}   regret {env.regret(refs['oracle'], refs['bayes']):.3f}")
    print(f"  regret denominator      {span:.6f}")
    print("\nALL CHECKS PASSED")

    with open("reference.json", "w") as f:
        json.dump(
            {
                "frozen_loss": refs["frozen"],
                "bayes_loss": refs["bayes"],
                "oracle_loss": refs["oracle"],
                "regret_denominator": span,
                "n_scored_predictions": refs["n_scored"],
                "heldout_seed": env.HELDOUT_SEED,
                "heldout_seqs": env.HELDOUT_SEQS,
                "heldout_pred": env.HELDOUT_PRED,
                "burn_in": env.BURN_IN,
                "env_config": vars(CFG) if hasattr(CFG, "__dict__") else CFG.__dict__,
            },
            f,
            indent=2,
        )
    print("wrote reference.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
