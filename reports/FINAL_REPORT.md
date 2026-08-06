# Next-Flop — Final Report

**Does feeding self-prediction error back into persistent state produce anything useful
beyond an ordinary recurrent model with an auxiliary self-prediction loss?**

**No** — not as a compressed summary of recent internal trajectory, on a task calibrated so
that a positive effect would have been visible fifteen times over.

824 runs · 203 core-hours · every stage pre-registered before it ran ·
<https://github.com/daniilsanders/next-flop>

---

## 1. The result

Stage 2 ran a 2 × 4 design — horizons k ∈ {1, 8} × conditions {B1, B2, B3, A} — across the
calibrated capacity band h ∈ {4, 6, 8, 12}, 10 paired seeds, 320 runs. Zero failures, all
converged, zero FIFO timing mismatches.

The decision rule was frozen in `PROTOCOL_V2.md` §8 before any result existed: Wilcoxon
signed-rank paired by seed, Holm–Bonferroni across band-h, minimum effect Δregret ≥ 0.05
within horizon and ≥ 0.03 for the interactions.

| test | outcome |
|---|---|
| **A vs B2 at k=8** (primary) | **not declared.** A slightly *worse*, positive at 0/4 band-h |
| **A vs B3 at k=8** (gate) | **not declared.** A better at 4/4, but +0.001 to +0.007 |
| A vs B2 at k=1 (known-negative arm) | negative at 4/4, exactly as predicted |
| both interactions | null, Holm p = 1.0 at most h |

Mean regret across all cells, lower is better:

```
B2 0.5653   ≈   B1 0.5665   <   A 0.5706   <   B3 0.5739
```

**Adding a δ_s channel carries a small, consistent cost that is never repaid.** Adding a
*scrambled* δ_s costs more — so δ_s content is reliably better than noise of matched shape,
just not worth the channel it arrives on. That last point is an observation below the
pre-registered threshold, not a finding.

Absolute scale: the A-vs-B2 gap of ~0.006 regret is **0.0007 nats**.

### Why the null is informative rather than merely absent

The stand was built so failure would be loud:

- the task is **demonstrably capacity-limited** — trained models hold ~0.94 chain beliefs per
  state dimension, and regret tracks the fraction dropped at r = 1.0000
- the architecture **can address its own state** — the first version could not, and that was
  caught before it cost a grid
- λ was **selected, not assumed** — and it mattered: 0.03 beat the provisional 0.1
- **MDE 0.0082–0.0116** against a 0.05 threshold, computed from measured paired σ
- k=8 gave δ_s the horizon that provably decouples it from δ_w (coupling 0.70 → 0.049)

The effects are not too small to resolve. They are absent.

---

## 2. Five findings, in the order they were forced

### 2.1 The first environment had no capacity deficit to find

A single-chain HMM (regime `z`, volatility `v`), 90 pre-registered runs, zero failures — and
no eligible capacity band at any λ. A 119-parameter model was already ~97% of Bayes-optimal.

The probe measured why. At h=2, `z` decodes at **99% of the exact Bayes ceiling** and
reconstructs the posterior `P(z=1)` linearly at **R² = 0.96 from two state dimensions**.
`v` — decodable at 0.871 by the exact filter, so not unknowable — is captured at ~0%.
Acquiring it in full moves regret from 0.029 to 0.0015.

> **The sufficient statistic was two numbers whose second dimension was worth under 3% of
> the frozen-to-Bayes range.** Nothing was expensive to forget, so no capacity band could
> exist. The design target is not "higher-dimensional statistic" but "each dimension carries
> large marginal value."

### 2.2 Structural arguments about δ_s do not predict the realized δ_s

A claim was on record that the environment made δ_w and δ_s anti-correlated *by
construction*. Measured across all 90 checkpoints: **70–84% of δ_s is linearly recoverable
from the scalar δ_w alone**, at every capacity level, shuffle baseline −0.0000. The argument
had the sign wrong.

The reason is architectural, not environmental. Conditional on history, `h_{t+1}` depends on
the world only through `x_{t+1}` and `p_s` predicts one step, so δ_s is a **deterministic
function** of δ_w — exactly redundant, not merely correlated. No latent structure fixes it.

> From this point on, claims about δ_s were measured rather than argued. That rule is the
> single most useful thing the project produced.

### 2.3 Multi-step self-prediction decouples them — at a stated limit

Zero training, on the same checkpoints:

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| coupling to δ_w | 0.702 | 0.386 | 0.142 | **0.049** | 0.014 | 0.004 |
| self-predictability | 0.912 | 0.832 | 0.715 | **0.566** | 0.423 | 0.305 |

Coupling falls 180× (≈ k^−1.50); self-predictability only 3× (≈ k^−0.32). Identical at every
h from 119 to 44k parameters — architectural, not capacity-dependent.

**Stated at the time, and it turned out to matter:** this does *not* make δ_s carry
information absent from the world-error stream. The window δ_w^t…δ_w^{t+k−1} still determines
δ_s^(k) exactly. The horizon buys an **accumulated summary `g` would otherwise have to build
across k steps** — a capacity-shortcut claim, not an independent-information one.

Stage 2 tested exactly that claim, and it failed.

### 2.4 A capacity experiment needs an architecture that can address its own state

The replacement environment first returned regret **0.949 identically at h = 4, 8, 16 and
32** — flat from step 1000, unchanged across an 8× capacity range. The read head was
`Linear([h, m, onehot(k)] → 1)`: additive in the query index, so `k` could only supply a
per-chain bias and could never select which part of `h` to read. The state was unreachable at
any capacity.

Caught by one timing run before a 216-run grid launched. Without it, every cell would have
returned ~0.949 and looked exactly like *"the redesign failed."*

### 2.5 The null, and a mechanistic account of it

**Post-hoc reasoning, explicitly not a finding** — this project falsified two confident
structural arguments already, and this one has not been tested:

> δ_s enters through `g` and must be stored in `h` to persist. But `h` is the bound, and δ_s
> came *from* `h`. You cannot escape a capacity limit by routing more through the same
> bottleneck — and δ_s is h-dimensional, so the channel is as wide as the state it is meant
> to help.

If that is right, the capacity-shortcut argument was weaker than it looked from the start,
and the null is the expected outcome rather than a surprise. It also predicts where the idea
could still work: on mechanisms that operate **on** the state rather than feeding **into** it.

---

## 3. What was never tested

1. **§5 restructuring** — state dynamics driven by something other than the current
   observation: replay, consolidation, settling. The only route the coupling analysis left
   open, and the one not subject to the bottleneck argument above, because it changes `h`'s
   dynamics instead of adding to `h`'s input. This is the version the originating document
   proposed. Much larger build.
2. **Lossy δ_w** — observations richer than what `p_w` predicts, so δ_w is a projection and
   δ_s can carry components it never sees.
3. **Scale.** 1,324 parameters. Nothing here speaks to whether any of this changes at size.
4. **Anything about "I".** Out of scope from the first page, and still is.

---

## 4. The stand

**Environment.** K = 16 parallel binary chains, ε = 0.015625, emissions 0.1/0.9, one revealed
query index per step. Each observation touches one chain and chains evolve independently, so
the posterior **factorises exactly and forever** — exact filtering is O(K) per step and
normalized regret keeps an exact ceiling. Verified through three independent
implementations: latent-path enumeration → full 2^K joint forward filter → factorised filter
(max diff 5.6e-16).

Marginal value, measured directly:

```
drop  2/16 chains (12%) → regret 0.121      drop  8/16 (50%) → regret 0.503
drop  4/16 chains (25%) → regret 0.245      drop 12/16 (75%) → regret 0.750
                                             r = 1.0000 vs fraction dropped
```

**Conditions.** B1 (no error feedback), B2 (δ_w only — the honest control), B3 (δ_w plus
batch-rolled δ_s — the matched-shuffle control), A (δ_w plus real δ_s). All share one
architecture, one parameter count, one initialization; a condition changes only what enters
the δ_s slot.

**Timing.** Horizon k and delay d are separate parameters, with **d = 9 in both arms** so the
only difference between horizons is what δ_s summarises, not when it arrives. Verified at six
(k, d) combinations. An off-by-one here — maturity is `k−1` steps after emission, not `k` —
was caught by writing the test, and would have left the arms differing by an unknown amount
of timing while both looked correct.

**Statistics.** Wilcoxon signed-rank, Holm across band-h, two families (primary + gate;
interactions secondary), thresholds 0.05 and 0.03, power gate from measured paired σ.

---

## 5. Method notes

Total: **824 runs, 203 core-hours, ~17 h wall clock** on a 16-core Mac Studio.

| stage | runs | core-hours |
|---|---|---|
| Stage 1 (v1 environment) | 90 | 24.7 |
| Calibration v1 (superseded) | 216 | 17.0 |
| Calibration v2 | 90 | 28.0 |
| Pilot | 24 | 7.4 |
| λ selection | 36 | 12.2 |
| Stage 2 | 320 | 98.4 |
| discarded (wrong λ) | 48 | 15.0 |

**The protocol stopped the project four times, and each stop was correct:**

- no capacity band in the first environment → retired it rather than tuning it
- an architecture that could not address its own state → caught by one run before a grid
- an eligibility condition that bit by 0.0004 → re-measured with longer training rather than
  moving the threshold
- Stage 2 silently running at λ=0.1 when the selection chose 0.03 → caught by a routine
  health check, 48 runs quarantined rather than deleted, restarted from zero

**Bugs the smoke path caught that the protocol path would have hidden:** `evaluate()`
iterating the held-out set by the *training* sequence length (identical by default, silently
wrong otherwise); `train.run()` checking only the training loss for NaN, so a diverged model
could be recorded `status="ok"` with NaN finals; and `scipy.stats.nct.cdf` returning NaN
patchily above noncentrality ~20, which produced plausible but wrong MDE values with no
error. All three would have produced clean-looking, invalid results.

**One discipline was worth more than the rest:** after 2.2 hours of compute produced the
first finding, everything subsequent — the marginal-value diagnosis, the coupling result, the
horizon mechanism, the unreachable read head — was caught by pre-flight checks on artifacts
already in hand, at 15–20 minutes each.

---

## 6. Reports

| file | what |
|---|---|
| `PROJECT_STATUS.md` | running overview |
| `STAGE1_REPORT.md` | first environment, 90 runs, ESCAPE_HATCH |
| `STAGE1_POSTMORTEM.md` | what `h` encodes; the marginal-value diagnosis |
| `STAGE1_POSTMORTEM_ADDENDUM.md` | δ_w/δ_s coupling; structural arguments fail |
| `HORIZON_DIAGNOSTIC.md` | multi-step self-prediction decouples them |
| `CALIBRATION_V1_REPORT.md` | K-chain mechanism validated; superseded |
| `CALIBRATION_V2_REPORT.md` | (K, ε) frozen |
| `STAGE2_RESULTS.md` | the null, with the frozen rule applied |
| `FINAL_REPORT.md` | this document |

Protocols: `PROTOCOL.md` (v1, retired environment), `PROTOCOL_V2.md` (frozen, governs
Stage 2).

---

## 7. The claim, as it now stands

> Multi-step self-prediction error does **not** help a bounded recurrent state, even when
> supplied as a compressed summary of its recent trajectory on a task where the state is
> demonstrably too small to hold what it needs. δ_s never carried information absent from the
> world-error stream, and the cheaper encoding it offered was not worth its channel.
>
> This does not test whether internal dynamics *not driven by the current observation* —
> replay, consolidation, restructuring — would behave differently. That remains the open
> question, and it is a different mechanism, not a larger version of this one.
