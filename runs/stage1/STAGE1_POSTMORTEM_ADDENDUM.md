# Postmortem Addendum — coupling between δ_w and δ_s

**Diagnostic only. Sets no threshold for the new environment.** Its sole purpose is to establish how much confidence to place in structural arguments that reason from the generative process to the realized δ_s.

`δ_s = h_{t+1} − p_s(f(h_t))` depends on `g`, `f` and `p_s` as well as the environment. The environment constrains at most one of its determinants, so whether a design decouples the two errors is an empirical property of the trained system.

## Metric definitions

Canonical implementations in `coupling.py`; Protocol v2 §11 should cite them rather than restate them. All on the held-out set with the §8 split (train [0, 48], report [48, 64], burn-in 512).

| | quantity | definition |
|---|---|---|
| **M1** | magnitude coupling | `corr(|δ_w|, ‖δ_s‖₂)` |
| **M2** | linear predictability | held-out R² of a linear map scalar `δ_w` → vector `δ_s` |
| **M3** | signed directional | `‖Cov(δ_s, δ_w)‖₂ / sqrt(Var(δ_w)·tr Cov(δ_s))` |
| **M4** | shuffle baseline | M1–M3 after same-timestep batch permutation of `δ_w` (batch roll by 1, same timestep) |

M2 and M3 are algebraically linked — computed in-sample on the same data, `M3² == R²` (verified to 1.8e-7). M2 is reported held-out, so they differ only by generalisation. **They are not independent evidence.**

No expected direction is pre-registered. The unit tests in `coupling.py` confirm the metrics separate cases they should: `δ_s = c·|δ_w|` yields M1 = 1.000 with M2 ≈ 0 and M3 = 0.018 — magnitude coupling with no signed linear coupling.

## Results by hidden size

Mean over 9 runs (3 λ × 3 seeds). Shuffled baseline in parentheses.

| h | M1 | M2 | M3 | M2 shuffled | M3 shuffled | regret | R²→P(z) | R²→P(v) |
|---|---|---|---|---|---|---|---|---|
| 2 | +0.774 | **+0.8382** | 0.9143 | -0.0001 | 0.0023 | 0.0287 | 0.962 | 0.035 |
| 3 | +0.759 | **+0.8180** | 0.9047 | -0.0000 | 0.0025 | 0.0168 | 0.973 | 0.306 |
| 4 | +0.778 | **+0.7645** | 0.8736 | -0.0000 | 0.0026 | 0.0068 | 0.987 | 0.622 |
| 6 | +0.882 | **+0.8578** | 0.9266 | -0.0000 | 0.0019 | 0.0015 | 0.986 | 0.910 |
| 8 | +0.839 | **+0.8008** | 0.8956 | -0.0000 | 0.0023 | 0.0010 | 0.996 | 0.937 |
| 12 | +0.853 | **+0.7815** | 0.8849 | -0.0000 | 0.0025 | 0.0007 | 0.999 | 0.967 |
| 16 | +0.880 | **+0.7759** | 0.8819 | -0.0000 | 0.0024 | 0.0006 | 0.999 | 0.973 |
| 24 | +0.847 | **+0.7553** | 0.8700 | -0.0000 | 0.0026 | 0.0004 | 0.999 | 0.979 |
| 32 | +0.814 | **+0.7315** | 0.8568 | -0.0000 | 0.0025 | 0.0004 | 0.999 | 0.980 |
| 64 | +0.853 | **+0.7077** | 0.8426 | -0.0000 | 0.0024 | 0.0003 | 0.999 | 0.981 |

Seed spread of M2: h=2 sd 0.091, h=4 sd 0.095, h=8 sd 0.037, h=64 sd 0.015

## Verdict

**Coupling is high everywhere and only weakly capacity-dependent.** M2 runs 0.838 at h=2 to 0.708 at h=64 — between 71% and 84% of δ_s variance is linearly recoverable from the scalar δ_w alone, at every capacity level tested. The shuffle baseline is -0.0000, so this is real structure, not a finite-sample artifact.

Notably this holds even at h=64, where the model tracks both latents essentially perfectly (R²→P(z) = 0.999, R²→P(v) = 0.981) and is within 0.0003 of Bayes. Extra capacity does not buy a δ_s that represents internal dynamics irreducible to δ_w — it buys, at most, a slow decline from 0.84 to 0.71.

**The structural argument was not merely unsupported — it had the sign wrong.** The claim on record was that this environment makes δ_w and δ_s *anti-correlated by construction*, falling out of Bayes. The measurement shows strong positive linear coupling at every h. Reasoning from the generative process to the realized δ_s should carry no weight going forward, including the analogous claim made for the K-chain design.

### A confound that limits how far this generalises

With a binary observation, conditional on history there is exactly **one bit of news per step**. `h_{t+1}` depends on `x_{t+1}` only through that bit, so `δ_w` and `δ_s` each take exactly two values at any timestep, both determined by the same bit. Conditional on state they are in one-to-one correspondence *by construction*, and high unconditional coupling is close to forced.

So the honest reading is not "the GRU chose to re-encode δ_w". It is that a one-bit-per-step observation channel cannot support a δ_s that is very independent of δ_w, whatever the latent structure. M2 measures how consistently that two-valued split aligns linearly across histories, which is informative but is not evidence about learned representational choices.

**This applies to the K-chain environment too** — its observations are also one bit per step. The lever for an independent δ_s is therefore not latent dimensionality but state dynamics that change for reasons *other than the current observation*: richer per-step observations, or internal events such as eviction under capacity pressure. That is a hypothesis, and by the standard this addendum just established, it should be measured rather than argued.

## Which listed outcome obtained

| outcome | obtained |
|---|---|
| high coupling at h=2 | **yes** — M2 = 0.838 |
| coupling falls as h grows | **weakly** — 0.838 → 0.708, not to a low level |
| low coupling throughout | no |
| unstable / seed-dependent | **no** — sd falls from 0.091 at h=2 to 0.015 at h=64; the central tendency is stable |

## Scope

Measured on **B2** checkpoints, where δ_s is computed but never fed into `g`. In condition A the feedback changes the dynamics and therefore δ_s itself, so these values do not directly describe A. What they do bound is the premise: if δ_s is largely a re-encoding of δ_w, then A's extra channel is largely redundant with what B2 already receives, and the B3 shuffle control becomes more important, not less.

