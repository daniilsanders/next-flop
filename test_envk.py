"""Validation for envk.py. Must pass before any calibration run.

The original environment's filter was correct and the experiment still failed, because
the *task* was wrong. So these tests check both: that the reference is right, and that
the environment has the property it was designed for -- dropping chain beliefs must cost
loss in proportion to how often those chains are queried.

    python3 test_envk.py
"""

import sys

import numpy as np

import envk
from envk import EnvKConfig

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def close(a, b, rel=0.05):
    return abs(a - b) <= rel * abs(b)


# ------------------------------------------------------- 1. generation

def test_generation():
    print("\n1. Generative process")
    cfg = EnvKConfig(K=8, eps=0.0625)
    d = envk.generate(128, 8192, np.random.default_rng(0), cfg)
    x, k, z = d["x"], d["k"], d["z"]

    flip = float((z[:, 1:] != z[:, :-1]).mean())
    check("per-chain flip rate", close(flip, cfg.eps), f"{flip:.5f} vs {cfg.eps}")

    kc = np.bincount(k.ravel(), minlength=cfg.K) / k.size
    check("query index uniform", np.all(np.abs(kc - 1 / cfg.K) < 0.1 / cfg.K),
          f"min {kc.min():.4f} max {kc.max():.4f} vs {1/cfg.K:.4f}")

    zq = np.take_along_axis(z, k[:, :, None], axis=2)[:, :, 0]
    p0, p1 = float(x[zq == 0].mean()), float(x[zq == 1].mean())
    check("P(x=1 | z=0)", close(p0, cfg.p_low, 0.02), f"{p0:.4f} vs {cfg.p_low}")
    check("P(x=1 | z=1)", close(p1, cfg.p_high, 0.02), f"{p1:.4f} vs {cfg.p_high}")
    check("marginal P(x=1) ~ 0.5", abs(x.mean() - 0.5) < 0.01, f"{x.mean():.4f}")
    check("chains independent", abs(float(np.corrcoef(z[:, :, 0].ravel(),
          z[:, :, 1].ravel())[0, 1])) < 0.02)


def test_determinism():
    print("\n2. Determinism under fixed seeds")
    cfg = EnvKConfig(K=8, eps=0.0625)
    a = envk.generate(8, 256, np.random.default_rng(42), cfg)
    b = envk.generate(8, 256, np.random.default_rng(42), cfg)
    c = envk.generate(8, 256, np.random.default_rng(43), cfg)
    check("same seed -> identical", all(np.array_equal(a[q], b[q]) for q in "xkz"))
    check("different seed -> different", not np.array_equal(a["x"], c["x"]))
    h1, h2 = envk.heldout(cfg), envk.heldout(cfg)
    check("heldout reproducible", np.array_equal(h1["x"], h2["x"]))


# ------------------------------------------------------- 3. filter correctness

def test_filter():
    print("\n3. Exact filter")
    rng = np.random.default_rng(1)

    # Path enumeration validates the joint filter, which in turn never assumes
    # factorisation -- so agreement of all three is a real check on the factorisation.
    cfg2 = EnvKConfig(K=2, eps=0.1)
    worst_bf = 0.0
    for _ in range(3):
        d = envk.generate(1, 5, rng, cfg2)
        bf = envk.brute_force_predict(d["x"][0], d["k"][0], cfg2, 4)
        jt = envk.joint_bayes_predict(d["x"], d["k"], cfg2)[0][:4]
        worst_bf = max(worst_bf, float(np.abs(bf - jt).max()))
    check("path enumeration == joint filter (K=2)", worst_bf < 1e-12, f"{worst_bf:.2e}")

    worst = 0.0
    for K in (2, 3, 4, 5):
        cfg = EnvKConfig(K=K, eps=0.08)
        d = envk.generate(4, 64, rng, cfg)
        fac = envk.bayes_predict(d["x"], d["k"], cfg)
        jt = envk.joint_bayes_predict(d["x"], d["k"], cfg)
        worst = max(worst, float(np.abs(fac - jt).max()))
    check("factorised == joint filter (K=2..5)", worst < 1e-10, f"max diff {worst:.2e}")


# ------------------------------------------------------- 4. reference lines

def test_references(cfg):
    print(f"\n4. Reference lines  (K={cfg.K}, eps={cfg.eps:g}, epsK={cfg.eps_K:g})")
    d = envk.heldout(cfg)
    x, k, z = d["x"], d["k"], d["z"]

    frozen = envk.mean_loss(envk.frozen_predict(x), x)
    oracle = envk.mean_loss(envk.oracle_predict(z, k, cfg), x)
    bayes = envk.mean_loss(envk.bayes_predict(x, k, cfg), x)
    h_or = -(cfg.p_high * np.log(cfg.p_high) + cfg.p_low * np.log(cfg.p_low))

    check("frozen == ln 2 exactly", abs(frozen - envk.LN2) < 1e-12, f"{frozen:.6f}")
    check("oracle == H(p_high)", abs(oracle - h_or) < 5e-3, f"{oracle:.6f} vs {h_or:.6f}")
    check("oracle < bayes < frozen", oracle < bayes < frozen,
          f"{oracle:.4f} < {bayes:.4f} < {frozen:.4f}")

    r_b, r_f = envk.regret(bayes, bayes), envk.regret(frozen, bayes)
    check("regret(bayes) == 0", abs(r_b) < 1e-12, f"{r_b:.2e}")
    check("regret(frozen) == 1", abs(r_f - 1.0) < 1e-12, f"{r_f:.6f}")
    print(f"       denominator (frozen - bayes) = {envk.LN2 - bayes:.4f} nats")
    return bayes


# ------------------------------------------------------- 5. marginal value

def test_marginal_value(cfg, bayes):
    """The property the redesign exists for. Dropping j of K chain beliefs must cost loss
    in proportion to j/K -- the fraction of queried steps left at chance."""
    print(f"\n5. Marginal value of a chain belief  (K={cfg.K})")
    d = envk.heldout(cfg)
    x, k = d["x"], d["k"]
    bi = envk.BURN_IN

    # Single chain: predicted cost is (share of steps querying it) x (ln2 - bayes there).
    kn = k[:, 1:][:, bi:]
    per = envk.bce_nats(envk.bayes_predict(x, k, cfg)[:, bi:], x[:, 1:][:, bi:])
    worst = 0.0
    for j in range(min(4, cfg.K)):
        m = kn == j
        pred = float(m.mean() * (envk.LN2 - per[m].mean()))
        got = envk.mean_loss(envk.bayes_predict(x, k, cfg, drop=[j]), x) - bayes
        worst = max(worst, abs(got - pred))
    check("single-chain drop matches its query share", worst < 1e-3, f"max err {worst:.2e}")

    # Proportionality across fractions dropped.
    fracs, regs = [], []
    for j in (0, cfg.K // 8, cfg.K // 4, cfg.K // 2, 3 * cfg.K // 4, cfg.K):
        j = max(0, min(cfg.K, j))
        r = envk.regret(envk.mean_loss(envk.bayes_predict(x, k, cfg, drop=range(j)), x), bayes)
        fracs.append(j / cfg.K)
        regs.append(r)
        print(f"       drop {j:>2}/{cfg.K} chains ({j/cfg.K:4.0%})  ->  regret {r:6.3f}")
    corr = float(np.corrcoef(fracs, regs)[0, 1])
    check("regret rises monotonically with fraction dropped",
          all(b >= a - 1e-9 for a, b in zip(regs, regs[1:])))
    check("regret ~ linear in fraction dropped", corr > 0.99, f"r={corr:.4f}")
    check("dropping all chains == frozen", abs(regs[-1] - 1.0) < 0.02, f"{regs[-1]:.4f}")
    check("half the state is worth a large share of the range", regs[3] > 0.35,
          f"drop 50% -> regret {regs[3]:.3f}")


def main():
    print("=" * 68)
    print("K-chain environment validation")
    print("=" * 68)
    cfg = EnvKConfig(K=16, eps=0.03125)  # epsK = 0.5

    test_generation()
    test_determinism()
    test_filter()
    bayes = test_references(cfg)
    test_marginal_value(cfg, bayes)

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
