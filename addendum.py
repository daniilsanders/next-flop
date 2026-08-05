"""Render STAGE1_POSTMORTEM_ADDENDUM.md from coupling.json."""

import json
import os

import numpy as np

import stage1

RUNS = os.path.join(stage1._HERE, "runs/stage1")


def main():
    with open(os.path.join(RUNS, "coupling.json")) as f:
        cj = json.load(f)
    rows = cj["rows"]
    hs = sorted({r["h"] for r in rows})
    A = lambda h, k: float(np.mean([r[k] for r in rows if r["h"] == h]))
    S = lambda h, k: float(np.std([r[k] for r in rows if r["h"] == h], ddof=1))
    allm = lambda k: float(np.mean([r[k] for r in rows]))

    L = []
    w = L.append
    w("# Postmortem Addendum — coupling between δ_w and δ_s\n")
    w("**Diagnostic only. Sets no threshold for the new environment.** Its sole purpose is "
      "to establish how much confidence to place in structural arguments that reason from "
      "the generative process to the realized δ_s.\n")
    w("`δ_s = h_{t+1} − p_s(f(h_t))` depends on `g`, `f` and `p_s` as well as the "
      "environment. The environment constrains at most one of its determinants, so whether "
      "a design decouples the two errors is an empirical property of the trained system.\n")

    w("## Metric definitions\n")
    w("Canonical implementations in `coupling.py`; Protocol v2 §11 should cite them rather "
      "than restate them. All on the held-out set with the §8 split "
      f"(train {cj['split']['train']}, report {cj['split']['test']}, burn-in "
      f"{cj['split']['burn_in']}).\n")
    w("| | quantity | definition |")
    w("|---|---|---|")
    w("| **M1** | magnitude coupling | `corr(|δ_w|, ‖δ_s‖₂)` |")
    w("| **M2** | linear predictability | held-out R² of a linear map scalar `δ_w` → vector `δ_s` |")
    w("| **M3** | signed directional | `‖Cov(δ_s, δ_w)‖₂ / sqrt(Var(δ_w)·tr Cov(δ_s))` |")
    w("| **M4** | shuffle baseline | M1–M3 after same-timestep batch permutation of `δ_w` "
      f"({cj['shuffle']}) |")
    w("")
    w("M2 and M3 are algebraically linked — computed in-sample on the same data, "
      "`M3² == R²` (verified to 1.8e-7). M2 is reported held-out, so they differ only by "
      "generalisation. **They are not independent evidence.**\n")
    w("No expected direction is pre-registered. The unit tests in `coupling.py` confirm the "
      "metrics separate cases they should: `δ_s = c·|δ_w|` yields M1 = 1.000 with M2 ≈ 0 "
      "and M3 = 0.018 — magnitude coupling with no signed linear coupling.\n")

    w("## Results by hidden size\n")
    w("Mean over 9 runs (3 λ × 3 seeds). Shuffled baseline in parentheses.\n")
    w("| h | M1 | M2 | M3 | M2 shuffled | M3 shuffled | regret | R²→P(z) | R²→P(v) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for h in hs:
        w(f"| {h} | {A(h,'m1_magnitude'):+.3f} | **{A(h,'m2_r2'):+.4f}** "
          f"| {A(h,'m3_directional'):.4f} | {A(h,'m2_r2_shuf'):+.4f} "
          f"| {A(h,'m3_directional_shuf'):.4f} | {A(h,'regret'):.4f} "
          f"| {A(h,'r2_pz'):.3f} | {A(h,'r2_pv'):.3f} |")
    w("")
    w("Seed spread of M2: " + ", ".join(f"h={h} sd {S(h,'m2_r2'):.3f}" for h in (2, 4, 8, 64)) + "\n")

    w("## Verdict\n")
    w(f"**Coupling is high everywhere and only weakly capacity-dependent.** M2 runs "
      f"{A(2,'m2_r2'):.3f} at h=2 to {A(64,'m2_r2'):.3f} at h=64 — between "
      f"{100*A(64,'m2_r2'):.0f}% and {100*A(2,'m2_r2'):.0f}% of δ_s variance is linearly "
      f"recoverable from the scalar δ_w alone, at every capacity level tested. The shuffle "
      f"baseline is {allm('m2_r2_shuf'):+.4f}, so this is real structure, not a "
      f"finite-sample artifact.\n")
    w(f"Notably this holds even at h=64, where the model tracks both latents essentially "
      f"perfectly (R²→P(z) = {A(64,'r2_pz'):.3f}, R²→P(v) = {A(64,'r2_pv'):.3f}) and is "
      f"within {A(64,'regret'):.4f} of Bayes. Extra capacity does not buy a δ_s that "
      f"represents internal dynamics irreducible to δ_w — it buys, at most, a slow decline "
      f"from {A(2,'m2_r2'):.2f} to {A(64,'m2_r2'):.2f}.\n")
    w("**The structural argument was not merely unsupported — it had the sign wrong.** "
      "The claim on record was that this environment makes δ_w and δ_s *anti-correlated by "
      "construction*, falling out of Bayes. The measurement shows strong positive linear "
      "coupling at every h. Reasoning from the generative process to the realized δ_s "
      "should carry no weight going forward, including the analogous claim made for the "
      "K-chain design.\n")

    w("### A confound that limits how far this generalises\n")
    w("With a binary observation, conditional on history there is exactly **one bit of news "
      "per step**. `h_{t+1}` depends on `x_{t+1}` only through that bit, so `δ_w` and `δ_s` "
      "each take exactly two values at any timestep, both determined by the same bit. "
      "Conditional on state they are in one-to-one correspondence *by construction*, and "
      "high unconditional coupling is close to forced.\n")
    w("So the honest reading is not \"the GRU chose to re-encode δ_w\". It is that a "
      "one-bit-per-step observation channel cannot support a δ_s that is very independent "
      "of δ_w, whatever the latent structure. M2 measures how consistently that two-valued "
      "split aligns linearly across histories, which is informative but is not evidence "
      "about learned representational choices.\n")
    w("**This applies to the K-chain environment too** — its observations are also one bit "
      "per step. The lever for an independent δ_s is therefore not latent dimensionality "
      "but state dynamics that change for reasons *other than the current observation*: "
      "richer per-step observations, or internal events such as eviction under capacity "
      "pressure. That is a hypothesis, and by the standard this addendum just established, "
      "it should be measured rather than argued.\n")

    w("## Which listed outcome obtained\n")
    w("| outcome | obtained |")
    w("|---|---|")
    w("| high coupling at h=2 | **yes** — M2 = {:.3f} |".format(A(2, "m2_r2")))
    w("| coupling falls as h grows | **weakly** — {:.3f} → {:.3f}, not to a low level |".format(
        A(2, "m2_r2"), A(64, "m2_r2")))
    w("| low coupling throughout | no |")
    w("| unstable / seed-dependent | **no** — sd falls from {:.3f} at h=2 to {:.3f} at h=64; "
      "the central tendency is stable |".format(S(2, "m2_r2"), S(64, "m2_r2")))
    w("")
    w("## Scope\n")
    w("Measured on **B2** checkpoints, where δ_s is computed but never fed into `g`. In "
      "condition A the feedback changes the dynamics and therefore δ_s itself, so these "
      "values do not directly describe A. What they do bound is the premise: if δ_s is "
      "largely a re-encoding of δ_w, then A's extra channel is largely redundant with what "
      "B2 already receives, and the B3 shuffle control becomes more important, not less.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    txt = main()
    stage1.atomic_write_text(os.path.join(RUNS, "STAGE1_POSTMORTEM_ADDENDUM.md"), txt)
    print(txt)
