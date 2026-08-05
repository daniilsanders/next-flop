"""Render STAGE1_POSTMORTEM.md from postmortem.json.

Separate from probe.py so the report can be regenerated without re-probing.

    python3 postmortem.py
"""

import json
import os

import numpy as np

import env
import probe as probe_mod
import stage1

RUNS = os.path.join(stage1._HERE, "runs/stage1")


def bayes_ceilings():
    """Best achievable linear-decode accuracy for z and v, from the exact posteriors.

    Without these, a raw accuracy is uninterpretable: z and v are latent and cannot be
    decoded perfectly even in principle. The ceiling is what the Bayes filter itself
    achieves on the same split.
    """
    d = env.heldout()
    bi = env.BURN_IN
    b = probe_mod.bayes_beliefs(d["x"])
    te = slice(*json.load(open(os.path.join(RUNS, "postmortem.json")))["probe"]["test_seqs"]) \
        if os.path.exists(os.path.join(RUNS, "postmortem.json")) else slice(48, 64)
    out = {}
    for name, p, t in (("z", b[..., 1] + b[..., 3], d["z"][:, 1:]),
                       ("v", b[..., 2] + b[..., 3], d["v"][:, 1:])):
        P, T = p[te, bi:].ravel(), t[te, bi:].ravel()
        out[name] = {"ceiling": float(((P > 0.5).astype(int) == T).mean()),
                     "chance": float(max(T.mean(), 1 - T.mean()))}
    return out


def captured(acc, c):
    """Fraction of the decodable signal the model actually carries. 1 = saturated."""
    return (acc - c["chance"]) / (c["ceiling"] - c["chance"])


def agg(rows, h, key):
    v = [r[key] for r in rows if r["h"] == h]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def main():
    with open(os.path.join(RUNS, "postmortem.json")) as f:
        pm = json.load(f)
    with open(os.path.join(RUNS, "manifest.json")) as f:
        tp = json.load(f)["manifest_provenance"] if False else json.load(f)["provenance"]

    rows = pm["rows"]
    hs = sorted({r["h"] for r in rows})
    lams = sorted({r["lambda"] for r in rows})
    bz, bv = pm["baseline_acc_z"], pm["baseline_acc_v"]

    L = []
    w = L.append
    w("# Stage 1 Postmortem — what does `h` actually encode?\n")
    w("**This is not part of Stage 1 selection.** It is a diagnostic run after the fact "
      "on the 90 Stage 1 checkpoints, to test whether the environment's predictive "
      "sufficient statistic is effectively one number. No checkpoint was retrained or "
      "altered; probe parameters are separate and no gradient reached a model.\n")

    w("## Setup\n")
    w(f"- checkpoints: {pm['n_checkpoints']} (Stage 1 B2, commit `{tp['git_commit'][:12]}`)")
    w(f"- probe: PROTOCOL.md §8 verbatim — single linear layer `h -> 2`, AdamW lr "
      f"{pm['probe']['lr']}, {pm['probe']['steps']} full-batch steps, frozen model")
    w(f"- data: held-out set, burn-in {pm['probe']['burn_in']} discarded, "
      f"fit on sequences {pm['probe']['train_seqs'][0]}–{pm['probe']['train_seqs'][1]-1}, "
      f"reported on {pm['probe']['test_seqs'][0]}–{pm['probe']['test_seqs'][1]-1}")
    w(f"- `h_t` is the state that produces the prediction for `x_{{t+1}}`, so it is aligned "
      f"with the Bayes *predictive* belief `b_{{t+1|t}}`; targets are `z_{{t+1}}`, `v_{{t+1}}`")
    cl = bayes_ceilings()
    w(f"- majority-class baselines: z **{bz:.3f}**, v **{bv:.3f}**")
    w(f"- **Bayes-optimal decode ceilings** (what the exact filter itself achieves on the "
      f"same split): z **{cl['z']['ceiling']:.4f}**, v **{cl['v']['ceiling']:.4f}**. "
      f"Both latents are substantially inferable; neither is unknowable. A raw accuracy "
      f"is meaningless without these.")
    w("- integrity: the replicated forward pass reproduced each run's recorded final eval "
      "loss to <1e-5, so the probed states are the states that produced the results\n")

    w("## Predictions under test\n")
    w("1. `z` highly linearly decodable even at `h=2`")
    w("2. `v` much weaker, possibly near chance")
    w("3. predictive performance near Bayes despite weak `v` decoding")
    w("4. (stronger) linear reconstruction of the exact Bayes posterior `P(z=1)` from `h` "
      "should be near-perfect at `h=2`\n")

    w("## By hidden size (mean ± sd over 9 runs: 3 λ × 3 seeds)\n")
    w("| h | params | acc(z) | acc(v) | R²→P(z=1) | R²→P(v) | regret | state velocity |")
    w("|---|---|---|---|---|---|---|---|")
    import model as _m
    for h in hs:
        c = {k: agg(rows, h, k) for k in
             ("acc_z", "acc_v", "r2_pz", "r2_pv", "regret", "state_velocity")}
        n = sum(p.numel() for p in _m.build(h, "B2", 0).parameters())
        w(f"| {h} | {n} | {c['acc_z'][0]:.3f} ± {c['acc_z'][1]:.3f} "
          f"| {c['acc_v'][0]:.3f} ± {c['acc_v'][1]:.3f} "
          f"| {c['r2_pz'][0]:.4f} ± {c['r2_pz'][1]:.4f} "
          f"| {c['r2_pv'][0]:.4f} ± {c['r2_pv'][1]:.4f} "
          f"| {c['regret'][0]:.4f} | {c['state_velocity'][0]:.3f} |")
    w("")

    w("## Full table, by h, λ and seed\n")
    w("| h | λ | seed | regret | velocity | acc(z) | acc(v) | R²→P(z=1) | R²→P(v) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["h"], r["lambda"], r["seed"])):
        w(f"| {r['h']} | {r['lambda']} | {r['seed']} | {r['regret']:.4f} "
          f"| {r['state_velocity']:.3f} | {r['acc_z']:.3f} | {r['acc_v']:.3f} "
          f"| {r['r2_pz']:.4f} | {r['r2_pv']:.4f} |")
    w("")

    # Mechanical verdicts. Thresholds are descriptive, chosen after seeing the data --
    # this is a postmortem, not a pre-registered test, and is labelled as such.
    h2 = [r for r in rows if r["h"] == 2]
    m = lambda k, rs=h2: float(np.mean([r[k] for r in rs]))
    w("## Verdict\n")
    w(f"At **h=2** (119 parameters, 2 state dimensions), averaged over 9 runs:\n")
    w(f"| quantity | value | chance | Bayes ceiling | signal captured |")
    w("|---|---|---|---|---|")
    w(f"| acc(z) | **{m('acc_z'):.3f}** | {cl['z']['chance']:.3f} | {cl['z']['ceiling']:.3f} "
      f"| **{100*captured(m('acc_z'), cl['z']):.0f}%** |")
    w(f"| acc(v) | **{m('acc_v'):.3f}** | {cl['v']['chance']:.3f} | {cl['v']['ceiling']:.3f} "
      f"| **{100*captured(m('acc_v'), cl['v']):.0f}%** |")
    w(f"| R² → exact Bayes P(z=1) | **{m('r2_pz'):.4f}** | 0 | 1.0 | — |")
    w(f"| R² → exact Bayes P(v) | **{m('r2_pv'):.4f}** | 0 | 1.0 | — |")
    w(f"| regret | **{m('regret'):.4f}** | 1 = frozen | 0 = Bayes | — |")
    w("")

    h6 = [r for r in rows if r["h"] == 6]
    m6 = lambda k: float(np.mean([r[k] for r in h6]))

    w("### Reading\n")
    w(f"**`z` is saturated from the smallest model.** At `h=2` the probe recovers "
      f"**{100*captured(m('acc_z'), cl['z']):.0f}%** of the decodable `z` signal and "
      f"reconstructs the exact Bayes posterior `P(z=1)` linearly at R² = {m('r2_pz'):.3f}, "
      f"from two state dimensions. It does not improve with more capacity — `acc(z)` is "
      f"flat at the ceiling across the entire sweep.\n")
    w(f"**`v` is acquired progressively, not ignored.** R² → `P(v)` runs "
      f"{m('r2_pv'):.2f} → " + " → ".join(
          f"{float(np.mean([r['r2_pv'] for r in rows if r['h']==h])):.2f}" for h in (3, 4, 6)) +
      f" across h = 2 → 3 → 4 → 6, and is essentially saturated by `h=6`. Volatility is "
      f"**not** unknowable — the exact filter decodes it at {cl['v']['ceiling']:.3f} — and "
      f"given room, the model does represent it.\n")
    w("**This corrects the working diagnosis.** The sufficient statistic is not one "
      "number; it is two, and the model acquires them in a strict order — `z` first, `v` "
      "only once `z` is free. The actual defect is the *marginal value* of the second "
      f"dimension: going from no `v` at all (`h=2`, R²={m('r2_pv'):.2f}) to `v` almost "
      f"fully represented (`h=6`, R²={m6('r2_pv'):.2f}) moves regret from "
      f"**{m('regret'):.4f} to {m6('regret'):.4f}** — the entire second dimension is worth "
      f"under {m('regret'):.1%} of the frozen-to-Bayes range.\n")
    w("So a maximally starved 119-parameter model is already ~97% of the way to optimal. "
      "There is no capacity deficit to find because there is nothing expensive to forget.\n")
    w("**Consequence for the redesign.** The target is not simply a higher-dimensional "
      "sufficient statistic — it is one where **each retained dimension carries large "
      "marginal value**, so that omitting a fraction of the state costs a comparable "
      "fraction of the achievable loss reduction. No emission setting can produce that: "
      "`p_low/p_high` changes how informative an observation is, not what dropping a "
      "dimension costs. K parallel chains have the required property directly — holding "
      "only j of K beliefs leaves the remaining (K−j)/K of steps predicted at chance, so "
      "regret should scale roughly as the fraction of state that does not fit, and the "
      "[0.20, 0.80] band should fall near h ∈ [0.2K, 0.8K].\n")
    return L, pm, rows, hs, lams


if __name__ == "__main__":
    L, *_ = main()
    print("\n".join(L))
