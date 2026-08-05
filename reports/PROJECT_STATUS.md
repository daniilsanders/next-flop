# Next-Flop — Project Status

Testing whether feeding **self-prediction error (δ_s) back into persistent state** produces
anything useful beyond an ordinary recurrent model with an auxiliary self-prediction loss.

Repository: <https://github.com/daniilsanders/next-flop>

---

## The question, unchanged

> Does feeding self-prediction error back into persistent state produce anything useful
> beyond an ordinary recurrent model with an auxiliary self-prediction loss?

Nothing below has answered it. Everything below is what had to be true before the question
could be asked at all.

---

## Findings so far

Four results, none of which were in the originating document. Each is measured, committed,
and reproducible.

### 1. The first environment had no capacity deficit to find

A single-chain HMM (regime `z`, volatility `v`) — 90 pre-registered runs, zero failures.

Its **predictive sufficient statistic is two numbers, and the second is nearly worthless**:
acquiring the volatility belief in full moves regret from 0.029 to 0.0015, so the entire
second dimension is worth **under 3% of the frozen-to-Bayes range**. A 119-parameter model
was already ~97% optimal. No capacity band existed at any emission setting.

The probe measured this rather than argued it: at h=2, `z` decodes at 99% of the Bayes
ceiling and reconstructs the exact posterior `P(z=1)` linearly at R² = 0.96 from two state
dimensions, while `v` — decodable at 0.871 by the exact filter — is captured at ~0% and
costs almost nothing to ignore.

**Consequence:** the design target is not "higher-dimensional sufficient statistic" but
"each dimension carries large marginal value". `→ STAGE1_POSTMORTEM.md`

### 2. Structural arguments about δ_s do not predict the realized δ_s

A claim was on record that this environment made δ_w and δ_s *anti-correlated by
construction*. Measured across all 90 checkpoints: **70–84% of δ_s is linearly recoverable
from the scalar δ_w alone**, at every capacity level, shuffle baseline −0.0000. The
argument had the sign wrong.

The reason is architectural, not environmental. Conditional on history, `h_{t+1}` depends
on the world only through `x_{t+1}` and `p_s` predicts one step, so δ_s is a *deterministic
function* of δ_w. Exactly redundant. No latent structure fixes it.

**Consequence:** reasoning from a generative process to δ_s carries no weight here; it gets
measured. `→ STAGE1_POSTMORTEM_ADDENDUM.md`

### 3. Multi-step self-prediction decouples them

Zero training, on the same checkpoints. Held-out R² of a linear map δ_w → δ_s^(k):

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| coupling | 0.702 | 0.386 | 0.142 | **0.049** | 0.014 | 0.004 |
| self-predictability | 0.912 | 0.832 | 0.715 | **0.566** | 0.423 | 0.305 |

Coupling falls **180×** (≈k^−1.50); self-predictability only **3×** (≈k^−0.32). Identical
at every h from 119 to 44k parameters — architectural, not capacity-dependent.

**Stated limit:** this does *not* make δ_s carry information absent from the world-error
stream. The sequence δ_w^t…δ_w^{t+k−1} still determines δ_s^(k) exactly. What the horizon
buys is that δ_s becomes an **accumulated summary `g` would otherwise have to build across
k steps** — a capacity-shortcut claim, not an independent-information one.
`→ HORIZON_DIAGNOSTIC.md`

### 4. A capacity experiment needs an architecture that can address its own state

The replacement environment (K parallel chains, revealed query index) first returned regret
**0.949 identically at h = 4, 8, 16 and 32** — flat from step 1000, unchanged across an 8×
capacity range. The read head was `Linear([h, m, onehot(k)] → 1)`: additive in the query
index, so `k` could only supply a per-chain bias and could never select which part of `h`
to read. The state was unreachable at any capacity.

Fixed with a content-addressed read (`⟨[h,m], read(k)⟩ + bias(k)`) and write. Caught by one
timing run before the 216-run grid launched; without it every cell would have returned
~0.949 and looked exactly like "the redesign failed".

---

## What is frozen

### Environment — K parallel chains

```
K chains z¹…z^K, each flips independently with probability ε each step
at step t an index k_t ~ Uniform{0..K−1} is drawn and REVEALED
x_t ~ Bernoulli(0.1 + 0.8 · z^{k_t}_t)
agent sees x_0..x_t and k_0..k_{t+1}, predicts x_{t+1}
```

Each observation touches one chain and chains evolve independently, so the posterior
**factorises exactly and forever** — exact filtering is O(K) per step, and normalized regret
keeps an exact ceiling. Scaling a single chain's latent space instead would have destroyed
that.

Verified three ways: explicit latent-path enumeration → full 2^K joint forward filter →
factorised filter (max diff 5.6e-16, K=2..5). The joint filter never assumes factorisation,
so their agreement tests it.

**The marginal-value property, measured directly:**

```
drop  2/16 chains (12%) → regret 0.121      drop  8/16 (50%) → regret 0.503
drop  4/16 chains (25%) → regret 0.245      drop 12/16 (75%) → regret 0.750
                                             r = 1.0000 vs fraction dropped
```

Half the state is worth half the range. The old environment's entire second dimension was
worth 3%.

### Calibrated parameters

```
K = 16,  εK = 0.25  (ε = 0.015625),  emissions 0.1/0.9
band  h ∈ {4, 6, 8, 12}  →  regret 0.763 / 0.646 / 0.530 / 0.294
smallest h (2) 0.881      largest h (64) 0.0047
mean between-seed SD over the band  0.0023
```

90 runs, 90 converged, all three εK eligible on all four conditions. Selection fell to
lowest seed SD; the frozen tie-break was never reached. Trained models hold **~0.94 chains
per state dimension**, constant until saturation.

Calibration v1 (216 runs, 8k steps, h≤32) is preserved as superseded — it validated the
mechanism but confounded convergence with capacity ceiling. v2 resolved that: the h=32
near-miss was **undertraining**, not a ceiling failure. No threshold was moved.

### Power outlook

σ_B2 = 0.0023 → planning σ = 0.0033 → required n = 2, so the gate returns the 10-seed
minimum, with **MDE 0.0032 — fifteen times smaller than the 0.05 minimum effect.** In the
first environment the gate nearly stopped the experiment on variance. Here it has
substantial headroom, which matters because the *interaction* is the binding constraint.

### Stage 2 architecture

Conditions **B1 / B2 / B3 / A** share one architecture, one parameter count, one
initialization (1324 params at h=8, K=16; byte-identical per seed). A condition changes only
what enters the δ_s slot.

Horizon `k` and feedback delay `d` are **separate parameters**. A prediction emitted at
iteration `j` targets `h_{j+k}`, which appears at iteration `j+k−1`, so maturity is `k−1`
steps after emission. Timing verified at six (k, d) combinations: first consumption lands
exactly at step `d`, contiguous after, target index always strictly less than consumption
index.

An off-by-one here — caught by writing the test, not by running anything — would have made
the k=8 arm consume at lag 8 instead of 9, leaving the two horizon arms differing by an
unknown amount of timing while both looked correct.

---

## Stage 2 design (agreed, not yet frozen)

A 2×4 ablation, because the redesign changed two independent axes:

| horizon | B1 | B2 | B3 | A |
|---|---|---|---|---|
| k=1 | ✓ | ✓ | ✓ | ✓ |
| k=8 | ✓ | ✓ | ✓ | ✓ |

Primary: **A vs B2 at k=8**, gated on **A vs B3** (matched-shuffle control — without it a
win means "a structured extra recurrent input helps", not "self-prediction error helps").

Two pre-registered interactions:

- `(A−B2)_{k=8} > (A−B2)_{k=1}` — does the advantage grow when δ_s becomes an accumulated summary?
- `(A−B3)_{k=8} > (A−B3)_{k=1}` — does the *semantic* advantage over shuffle grow with horizon?

Without the k=1 arm, a positive result at k=8 stays ambiguous between "the K-chain
environment finally exposed the original effect" and "multi-step accumulation created it".

**Projected cost** at 1103 s/run on 12 workers: 320 runs ≈ 8.2 h at 10 seeds, up to 960 runs
≈ 24.5 h at 30.

---

## Open decisions

Four modifications proposed and **not yet resolved**:

1. **Delay decoupling.** `d = k+1` gives lag 2 at k=1 and 9 at k=8, so the arms differ in
   timing *and* in what δ_s represents, and the interaction cannot separate them. Holding
   `d = 9` in both arms isolates the horizon at no extra cost. *(Code supports both.)*
2. **Aux-loss normalisation.** λ was selected at k=1; k=8 residuals are systematically
   larger, so the same λ silently reweights the auxiliary objective in the arm under test.
   Normalising by detached target variance makes λ comparable. *(Implemented, switchable.)*
3. **Separate effect size for the interaction.** A difference-of-differences carries more
   variance than a single paired difference and plausibly a smaller true effect; inheriting
   Δregret ≥ 0.05 tests it at a threshold designed for something else.
4. **Nonlinear readout for window-coupling.** The claim `δ_s^(k)` is recoverable from the
   full δ_w window is *informational*; a linear map from an 8-vector understates it.

---

## Current state

- **Running:** power pilot — h=8, conditions A/B2/B3 at k=8, seeds 301–308, 24 runs (~37 min).
  Measures the actual paired A−B2 σ instead of the √2·σ_B2 proxy, and smoke-tests the Stage 2
  architecture. At k=8 both delay designs coincide at d=9, so it prejudges nothing.
- **Not started:** Protocol v2, the doubled power calculation, Stage 2.
- **Uncommitted:** Stage 1 selection artifacts (`selection.json`, `completion.json`,
  `STAGE1_REPORT.md`, the `analyze1.py` routing fix) — pending review of the escape-hatch
  decision, now moot since the environment was retired instead.

---

## Reports

| file | what |
|---|---|
| `STAGE1_REPORT.md` | first environment, 90 runs, decision ESCAPE_HATCH |
| `STAGE1_POSTMORTEM.md` | what `h` actually encodes; the marginal-value diagnosis |
| `STAGE1_POSTMORTEM_ADDENDUM.md` | δ_w/δ_s coupling; structural arguments fail |
| `HORIZON_DIAGNOSTIC.md` | multi-step self-prediction decouples them |
| `CALIBRATION_V1_REPORT.md` | K-chain mechanism validated; superseded |
| `CALIBRATION_V2_REPORT.md` | (K, ε) frozen |

---

## A note on the method

The expensive finding cost 2.2 hours of compute. Everything since — the marginal-value
diagnosis, the coupling result, the horizon mechanism, the unreachable read head — was
caught by pre-flight checks on artifacts already in hand, at 15–20 minutes each.

Two bugs were caught by the smoke path that the protocol path would have hidden entirely:
`evaluate()` iterating the held-out set by the *training* sequence length (identical by
default, silently wrong otherwise), and `train.run()` checking only the training loss for
NaN, so a diverged model could be recorded as `status="ok"` with NaN finals. Both would have
produced clean-looking, invalid results.

The protocol has stopped the project three times — no capacity band, an unreachable
architecture, an eligibility condition that bit. Each stop was correct.
