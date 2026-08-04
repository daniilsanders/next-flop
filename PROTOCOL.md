# Experiment 1 — Pre-registered Protocol

**Question (the only one this experiment answers):**
Does feeding self-prediction error δ_s back into persistent state produce anything useful
beyond an ordinary recurrent model with an auxiliary self-prediction loss?

**Status:** FROZEN. Any change after Stage 1 begins must be recorded in `DEVIATIONS.md`
with a timestamp and a reason, and reported alongside results.

---

## 1. Environment

Three-level generative HMM. Binary observation, scalar prediction, no actions.

```
v_{t+1}  = flip(v_t) with prob p_v = 0.001        volatility ∈ {calm, volatile}
z_{t+1}  = flip(z_t) with prob ε(v_{t+1})         regime ∈ {0, 1}
             ε(calm) = 0.005      (mean dwell 200)
             ε(volatile) = 0.05   (mean dwell 20)
x_{t+1} ~ Bernoulli(0.25 + 0.5·z_{t+1})
```

**Transition order is fixed:** v transitions first; z then flips using the *new* v.

Initial state: `z_0 ~ Uniform{0,1}`, `v_0 ~ Uniform{calm, volatile}`.

**Agent interface.** At step t the agent has seen `x_1..x_t` and emits `p̂_t = P(x_{t+1}=1)`.
Loss is binary cross-entropy in **nats**. The agent never observes z or v.

**Latent state space is 4** — exact forward filtering is tractable and is used as the ceiling.

### Reference lines

| Line | Value | Source |
|---|---|---|
| Frozen (best constant) | **0.6931** nats | H(0.5), analytic |
| Oracle (knows z) | **0.5623** nats | H(0.75), analytic |
| **Bayes-optimal** | compute | exact forward filter on held-out set |

Bayes-optimal is computed numerically once, on the held-out set, before Stage 1, and
recorded in `reference.json`. It is not recomputed.

### Normalized regret (primary metric)

```
regret = (loss − bayes_loss) / (0.6931 − bayes_loss)
```

`0` = Bayes-optimal, `1` = frozen state. All comparisons use this scale.

---

## 2. Conditions

**All five conditions share one architecture, one parameter count, and one initialization.**
They differ *only* in what is written into the δ_w and δ_s input slots of `g`. This removes
the parameter-count confound entirely.

| | aux self-pred loss | δ_w slot | δ_s slot |
|---|---|---|---|
| **B0** floor | – | zeros | zeros |
| **B1** stated control | ✓ | zeros | zeros |
| **B2** honest control | ✓ | δ_w^{t−1} | zeros |
| **B3** matched-shuffle control | ✓ | δ_w^{t−1} | δ_s^{t−1} **batch-rolled** |
| **A** treatment | ✓ | δ_w^{t−1} | δ_s^{t−1} |

In B0/B1/B2 the δ_s slot receives an exact zero vector (bypassing LayerNorm, which is
degenerate on zeros). Those input weights exist but are inert.

### B3 — the confound control

B3 exists to separate *"self-prediction error helps"* from *"a structured extra recurrent
input helps."*

At step t, batch element i receives the δ_s computed by batch element `(i+1) mod B` **at the
same timestep t−1**, passed through the identical normalization path.

Matched by construction: dimensionality, temporal delay, normalization, marginal
distribution, marginal variance, and any training-time or within-sequence time trend
(same t, same optimizer step, same pool).

Destroyed: the relationship to *this* sequence's state. The swapped vector's *conditional*
variance is deliberately wrong — sequence j may be volatile while i is calm. That mismatch
is the semantics being removed, and is the point.

B3 still computes its own real δ_s for the aux loss and for logging. Compute and loss are
identical to A; only the routing differs. Single-variable change.

Optional secondary (`B3n`, run only if A > B2 and A > B3): δ_s slot receives Gaussian noise
with per-dimension mean/variance matched to a running estimate of δ_s. Decomposes
"structure" from "self-ness."

---

## 3. Model sizing

### Hidden-state sweep (Stage 1)

```
h ∈ {2, 3, 4, 6, 8, 12, 16, 24, 32, 64}
```

Ten points, dense at the bottom. Exact Bayes needs 3 free numbers, so the capacity-limited
band is expected low. `h = 64` is the ceiling anchor, not a candidate.

### Fixed dimensions

- observation `x`: 1
- self-model `m`: `max(4, h // 2)`
- δ_w slot: 1 (signed scalar residual `x_t − x̂_t`; no normalization — already bounded [−1,1])
- δ_s slot: `h + 1` — `LayerNorm(δ_s) ⊕ log(‖δ_s‖₂ + 1e−6)`

The split preserves both direction (normalized, stable input distribution across training)
and magnitude (explicit scalar — "how surprised am I" is not discarded). B3's swapped
vector goes through the identical path, so normalization is matched by construction.

---

## 4. Training parameters

| Parameter | Value |
|---|---|
| Sequence length | 4096 steps |
| Burn-in | 512 steps — excluded from **reported metrics only**, not from training loss |
| Truncated-BPTT window | 128 steps |
| `h` across windows | carried forward, **detached** at boundary (not zeroed) |
| Batch | 64 independent sequences |
| Total gradient steps | **30,000** (fixed; no early stopping) |
| Optimizer | AdamW, betas (0.9, 0.99), eps 1e−8 |
| Weight decay | **0.0** (regularization would confound the capacity sweep) |
| Peak LR | 3e−3 |
| Schedule | linear warmup 500 steps → cosine decay to 3e−4 at step 30,000 |
| Grad clip | global norm **1.0** |
| Aux-loss weight λ | selected in Stage 1 (§6.1), then **frozen and identical across all conditions** |
| Precision | fp32 |

**Hyperparameters are identical across all conditions.** No per-condition tuning, ever.

If Stage 1 shows a given `h` failing to converge at lr 3e−3, the LR may be retuned **once**
for that `h` and must then be applied to *all* conditions at that `h`, recorded in
`DEVIATIONS.md`.

Training data is generated on the fly from an infinite stream — there is no train set and
no overfitting. Only the fixed held-out set is used for comparison.

---

## 5. Evaluation

- **Interval:** every 1,000 gradient steps.
- **Held-out set:** 64 sequences × 4096 steps, generated once from RNG seed `9999`,
  **identical across every condition, every h, and every seed.**
- Burn-in 512 discarded → 3,584 scored predictions/sequence → 229,376 per eval.
- **Reported final value:** mean of the evals at steps 28k, 29k, 30k. Fixed in advance to
  remove eval noise without introducing a selection choice.

### Seeding

Seeds `1..10`. Seed `s` controls model init *and* the training data stream, and is **paired
across conditions**: A/seed-3 and B2/seed-3 see identical initialization and an identical
sequence of training batches. All comparisons are paired at the seed level.

| Stage | Runs |
|---|---|
| Stage 1 (λ + band) | B2 only, 10 h × 3 seeds × 3 λ = **90 runs** |
| Stage 2 (comparison) | band-h × {A, B1, B2, B3} × 10 seeds |
| Stage 2 floor | band-h × B0 × 5 seeds |

Stage 2 does not begin until Stage 1 is complete and both selections are written to
`selection.json`. That file is not edited afterward.

---

## 6. Stage 1 selections

Stage 1 runs **B2 only**. Both selections below are made from B2 runs, in the fixed order
given, before any Stage 2 condition is trained.

### 6.1 λ selection

```
λ ∈ {0.03, 0.1, 0.3}
```

1. Run the full B2 hidden-size sweep for all three λ values (10 h × 3 seeds × 3 λ).
2. For each λ, compute **mean regret across all h whose mean regret lies in [0.05, 0.95]** —
   this excludes both ceiling and completely failed models from dominating the aggregate.
3. Select the λ with the **lowest aggregate regret**.
4. **Tie-break** (differences < 0.01 aggregate regret): prefer `λ = 0.1`; if 0.1 is not among
   the tied values, prefer the smaller λ.
5. Write to `selection.json`. Never revisited.

If no h falls in [0.05, 0.95] for a given λ, that λ has no valid aggregate and is excluded.
If no λ has a valid aggregate, the escape hatch (§6.3) applies.

**Acknowledged selection dependency.** λ and the capacity band are chosen from the same
Stage-1 runs, over overlapping windows ([0.05, 0.95] and [0.20, 0.80]). These selections are
not statistically independent, and this is disclosed rather than claimed away. The bound on
the risk: **both selections use B2 only — the control.** Neither can favour the treatment.
If λ selection biases anything, it biases toward a *stronger* control, which is conservative
with respect to the primary hypothesis.

### 6.2 Capacity band

Using **only the selected λ**, an `h` enters the band iff **B2's mean regret across its 3
Stage-1 seeds ∈ [0.20, 0.80]**.

**If fewer than 2 values of h qualify, the experiment stops** and reports "no measurable
capacity band." That is the informative null. The task is not adjusted to manufacture a
band, and the criterion is not widened.

### 6.3 Escape hatch — specified in advance, usable exactly once

Decided before Stage 2, applied only to the whole of Stage 1 re-run:

- If **no** h reaches regret ≥ 0.20 (task too easy): weaken evidence per observation,
  emissions `0.25/0.75 → 0.35/0.65`. Rerun Stage 1.
- If **all** h exceed regret 0.80 (task too hard): strengthen, `0.25/0.75 → 0.15/0.85`.
  Rerun Stage 1.

One use total. If the rerun also fails to produce a band, stop. Reference lines and
Bayes-optimal are recomputed for whichever task version is used, and the version is
recorded in every result file.

---

## 7. Decision rule

**Primary endpoint:** paired difference in final regret, A − B2, across the 10 paired seeds
at each band-h.

- **Test:** Wilcoxon signed-rank, two-sided, paired by seed.
- **Multiple comparisons:** Holm–Bonferroni across the band-h values.
- **Minimum effect size:** mean paired Δregret ≥ **0.05** (absolute, normalized scale).

**Declare A > B2** iff *all three* hold:
1. Holm-corrected p < 0.05 at ≥ 1 band-h, **and**
2. mean paired Δregret ≥ 0.05 at that h, **and**
3. the effect has the same sign at ≥ half the band-h points (uncorrected consistency check).

**The same rule, unchanged, is applied to A vs B3.**

### Interpretation table — fixed before results exist

| Outcome | Conclusion |
|---|---|
| A ≈ B2 ≈ B3 | **Null.** δ_s feedback does nothing. |
| A > B2, A ≈ B3 | **Confound confirmed.** A structured extra recurrent input helps; self-ness does not. |
| A > B2, A > B3, B3 ≈ B2 | **Hypothesis supported.** Self-prediction error specifically helps. |
| A > B2, A > B3, B3 > B2 | Both effects real. Report the decomposition; run B3n. |
| A < B2 | δ_s feedback destabilizes. Report as such; inspect ‖δ_s‖ trajectory for divergence. |

**Secondary endpoint (pre-registered):** volatility-probe accuracy, A vs B2. If the
mechanism is what the hypothesis claims, A should encode `v` more accessibly. A win on
regret *without* a win on v-probe accuracy is reported as unexplained.

---

## 8. Probes

- **Architecture:** a single linear layer per target — `h → 2` logits for `z`, `h → 2` for
  `v`. **Linear only.** A nonlinear probe measures the probe's capacity, not h's.
- **When:** after training completes, on a fully frozen model. Never during training. No
  gradient reaches the model from any probe.
- **Data:** `h_t` trajectories collected on the held-out set, burn-in discarded. Fixed split:
  train on sequences 0–47, report on 48–63. Same split for every condition.
- **Probe training:** AdamW, lr 1e−2, 2,000 steps, full-batch.
- **Reported:** plain accuracy for `z` and for `v` (both latents are marginally balanced by
  symmetry).

---

## 9. Detached quantities

Gradient **does not** flow through:

1. **δ_w^{t−1} and δ_s^{t−1} as inputs to `g`.** Both fully detached. They are features, not
   gradient paths. Without this, `g` can be trained to make δ_s small — failure mode §3.2,
   directly.
2. **The self-prediction target.** Aux loss is `‖ sg(h_{t+1}) − p_s(m_t) ‖²`. Stop-grad on
   the target. Gradient reaches `p_s` and `f_φ` only. Without this the cheapest solution is
   for `g` to move `h` toward `ĥ` — §3.2 again, from the other side.
3. **`h` at truncated-BPTT window boundaries.** Standard.
4. **`h` entering any probe.**
5. **The batch-rolled δ_s in B3.** Same treatment as A.

Gradient **does** flow through:

- **`m_t` on its path into `g`.** Deliberate. The self-model should inform the state update,
  and task gradient should shape `f_φ` to make `m` useful. §3.1 is addressed by giving `f_φ`
  a self-referential *target*, not by severing the task gradient. This is a choice, recorded
  here so it is not mistaken for an oversight.
- `p_w` and the world-prediction loss: normal.

Note that δ_w appears in two roles — as a detached input feature to `g`, and inside the
world-prediction loss where gradient flows normally. These are separate paths.

---

## 10. Stopping and run validity

- **No early stopping on any metric.** Every run executes exactly 30,000 steps.
- **No seed replacement.** An outlier seed stays in the sample; Wilcoxon absorbs it.
- **Divergence abort:** a run is marked FAILED if loss is NaN, or if eval loss > 0.6931
  (worse than frozen) at step 10,000. If > 2 of 10 seeds fail in any condition, that
  condition is reported as **unstable** — failures are not replaced with fresh seeds.
- **Convergence gate (validity check, not a stopping rule):** for every condition, mean eval
  loss must change by < 1% between step 25k and step 30k. If any condition fails this, the
  comparison is invalid and the budget is extended to 60,000 steps **for all conditions**
  and everything is rerun. Never extended per-condition.

---

## 11. Logged every eval, all conditions

Non-negotiable — every failure mode in the design produces clean-looking numbers, and
without these you get a result you cannot interpret and will not know it.

- eval loss (nats), normalized regret
- mean ‖δ_w‖, mean ‖δ_s‖
- mean ‖h_{t+1} − h_t‖ — **freeze detector**
- per-dimension variance of `h` — detects collapse into a subspace
- z-probe and v-probe accuracy (final only)
- gradient norm, pre-clip

Three independent freeze instruments: regret pinned at 1.0, state velocity → 0, probe
accuracy at chance. A frozen state that scores well on any one of these alone is a bug.

---

## 12. Compute

~225 runs total (90 Stage 1, ~135 Stage 2). Model is ~2–20k parameters; sequential rollout
dominates, so CPU with parallel workers is expected to beat MPS at this tensor size.
Estimated 5–15 min per run, 8 workers → **5–7 hours wall clock** on the Mac Studio for the
full protocol.
