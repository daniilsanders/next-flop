# Protocol v2 — Stage 2 Pre-registration

**Question, unchanged from v1:**

> Does feeding self-prediction error δ_s back into persistent state produce anything useful
> beyond an ordinary recurrent model with an auxiliary self-prediction loss?

**Status:** FROZEN. Any change after Stage 2 begins goes in `DEVIATIONS.md` with a timestamp
and a reason, and is reported alongside results.

Protocol v1 governed a single-chain environment that was retired: its predictive sufficient
statistic was two numbers whose second dimension was worth under 3% of the frozen-to-Bayes
range, so no capacity band existed at any setting (`reports/STAGE1_POSTMORTEM.md`). v1's
conditions, stop-gradient discipline, power-gate machinery and decision rules carry over;
its environment does not.

---

## 1. Environment — frozen by Calibration v2

```
K chains z¹…z^K, each flips independently with probability ε each step
at step t an index k_t ~ Uniform{0..K−1} is drawn and REVEALED
x_t ~ Bernoulli(p_low + (p_high − p_low) · z^{k_t}_t)
agent has seen x_0..x_t and k_0..k_{t+1}, and predicts x_{t+1}
```

| parameter | value |
|---|---|
| K | **16** |
| ε | **0.015625** (εK = 0.25) |
| emissions | **0.1 / 0.9** |
| capacity band | **h ∈ {4, 6, 8, 12}** |

Each observation touches one chain and chains evolve independently, so the posterior
**factorises exactly and forever**. Exact filtering is O(K) per step, so normalized regret
retains an exact ceiling. Verified three ways: latent-path enumeration → full 2^K joint
forward filter → factorised filter, max diff 5.6e-16.

**Regret** is `(loss − bayes) / (ln 2 − bayes)`; 0 = Bayes-optimal, 1 = frozen state.
The frozen predictor emits 0.5 and scores exactly ln 2 because emissions are symmetric.

**Marginal value**, the property v1's environment lacked, measured directly: dropping j of K
chain beliefs costs regret ≈ j/K (r = 1.0000 against fraction dropped). Trained models hold
**~0.94 chains per state dimension**.

---

## 2. Design — 2 × 4

| horizon | B1 | B2 | B3 | A |
|---|---|---|---|---|
| **k = 1** | ✓ | ✓ | ✓ | ✓ |
| **k = 8** | ✓ | ✓ | ✓ | ✓ |

| | aux self-pred loss | δ_w slot | δ_s slot |
|---|---|---|---|
| **B1** | ✓ | zeros | zeros |
| **B2** honest control | ✓ | live | zeros |
| **B3** matched shuffle | ✓ | live | batch-rolled at consumption |
| **A** treatment | ✓ | live | live |

All conditions share **one architecture, one parameter count, one initialization**. A
condition changes only what enters the δ_s slot. B1/B2 receive exact zeros in that slot at
the identical timing.

**Why both horizons.** The redesign changed two independent axes — the environment now
creates a real capacity bottleneck, and the horizon changes what δ_s represents. Without the
k=1 arm, a positive result at k=8 remains ambiguous between *"the K-chain environment finally
exposed the original effect"* and *"multi-step accumulation created the effect."*

**k=1 is a known-negative arm.** Conditional on history, `h_{t+1}` depends on the world only
through `x_{t+1}` and `p_s` predicts one step, so δ_s is a *deterministic function* of δ_w —
exactly redundant, not merely correlated. A null at k=1 is predicted by construction.

---

## 3. Timing — horizon and delay are separate

```
horizon k   p_s(m_t) predicts h_{t+k}; the target matures k steps later
delay   d   steps from PREDICTION to CONSUMPTION.  FROZEN AT d = 9 IN BOTH ARMS.
```

At step t the predictor emits ĥ_{t+k}. At step t+k the realized h_{t+k} exists, so the
detached δ_s^(k) is computed then, held in an explicit FIFO, and routed into the recurrence
at step t+9.

**Holding d = 9 in both arms is the point.** With `d = k+1` the arms would differ in timing
(lag 2 vs 9) *and* in what δ_s represents, and the interaction tests below could not separate
them. The only intended difference between arms is what δ_s summarises.

A prediction emitted at iteration j targets `h_{j+k}`, which appears as `h_new` at iteration
`j+k−1` — maturity is `k−1` steps after emission, not k. Verified at six (k, d) combinations:
first consumption lands exactly at step d, contiguous thereafter, target index always
strictly less than consumption index. B3 swaps the **same matured vectors at the same
timestep** (batch roll by 1 at consumption).

---

## 4. Auxiliary loss

```
loss_s = ‖ sg(h_{t+k}) − p_s(m_t) ‖²  /  ( detach(Var[h_{t+k}]) + 1e−6 )
```

Normalised by the **detached** target variance, computed identically in both arms. λ was
selected at k=1; k=8 residuals are systematically larger (self-predictability falls
0.91 → 0.57 across k=1…8), so an unnormalised loss would silently reweight the auxiliary
objective in the arm under test.

**Both raw and normalised losses are logged every eval**, so normalisation cannot hide a
scale difference.

λ is re-selected under this environment by the §6.1 rule before Stage 2; calibration's
provisional λ = 0.1 is not carried forward.

---

## 5. Training

| parameter | value |
|---|---|
| sequence length | 4096 predictions |
| burn-in | 512 — excluded from reported metrics only |
| truncated-BPTT window | 128, state carried and **detached** at boundaries |
| batch | 64 independent sequences |
| total steps | **30,000**, fixed, no early stopping |
| optimizer | AdamW, betas (0.9, 0.99), eps 1e−8, **weight decay 0.0** |
| schedule | linear warmup 500 → cosine decay to 3e−4 at 30,000; peak 3e−3 |
| grad clip | global norm 1.0 |
| precision | fp32 |

Hyperparameters are **identical across all conditions and both horizons**. No per-condition
tuning. Training data is generated on the fly — there is no train set and no overfitting.

**Seeding.** Seeds `1..N`, N set by §7. Seed s fixes initialization *and* the training
stream, **paired across every condition and both horizons**: A/k=8/seed-3 and B2/k=1/seed-3
see identical initialization and an identical sequence of training batches.

**Evaluation.** Every 1,000 steps on a fixed held-out set (64 × 4096, RNG seed 9999),
identical for every cell. Reported value is the mean of the evals at 28k, 29k, 30k.

---

## 6. Detached quantities

Gradient **does not** flow through: δ_w and δ_s as inputs to `g` (features, not gradient
paths — without this `g` can be trained to shrink δ_s); the self-prediction **target**
`sg(h_{t+k})` (without this the cheapest fix is for `g` to move `h` toward `ĥ`); the
aux-loss variance normaliser; `h` at TBPTT boundaries; everything inside the FIFO; `h`
entering any probe; B3's batch-rolled vector.

Gradient **does** flow through `m_t` on its path into `g` — deliberate, recorded so it is
not mistaken for an oversight — and through `p_w` normally.

---

## 7. Power gate

Frozen constants: two-sided α = 0.05, 80% power, **minimum 10 seeds, maximum 30**.

| comparison | minimum effect |
|---|---|
| within-horizon A vs B2, A vs B3 | **Δregret ≥ 0.05** |
| interaction (difference-of-differences) | **Δregret ≥ 0.03** |

Required n is computed for **both**, from the paired SD measured in the pilot, and the
**larger** governs. The interaction is the binding constraint and carries roughly √2 the
variance of a single paired difference.

Measured pilot paired SDs (h=8, seeds 401–404): A−B2 0.0029 (k=1), 0.0060 (k=8);
A−B3 0.0082 (k=1), 0.0069 (k=8). Worst = **0.0082**.

Rounded up to the next even number; ≤10 keeps 10, 11–30 raises all cells, >30 stops and
reports *"no feasible powered comparison under the pre-registered compute cap."*

**Forbidden as responses to an unfavourable gate:** lowering either effect threshold,
narrowing the capacity band, selecting only low-variance h, or raising the cap.

Calculation is a **planning approximation** (paired t; the reported test is Wilcoxon) and is
labelled as such in `power_v2.json`, committed before any Stage 2 run.

---

## 8. Decision rule

**Primary:** paired difference in final regret, **A − B2 at k = 8**, across N paired seeds
at each band-h.

- Test: **Wilcoxon signed-rank**, two-sided, paired by seed
- Multiple comparisons: **Holm–Bonferroni across band-h**
- Minimum effect: mean paired Δregret ≥ 0.05

**Declare A > B2 at k=8** iff all three hold: Holm-corrected p < 0.05 at ≥1 band-h; mean
paired Δregret ≥ 0.05 there; same sign at ≥ half the band-h points.

**The same rule, unchanged, gates A vs B3 at k=8.** A win over B2 without a win over B3 is
reported as *"a structured extra recurrent input helps"*, not *"self-prediction error helps."*

**Interactions** (secondary family, Holm-corrected within itself, threshold 0.03):

```
(A − B2)_{k=8}  >  (A − B2)_{k=1}
(A − B3)_{k=8}  >  (A − B3)_{k=1}
```

A longer horizon could improve or hurt every condition equally, which would say nothing
about self-prediction feedback specifically. Only the interactions speak to it.

**Correction hierarchy, frozen:** family 1 = {A vs B2, A vs B3} at k=8, Holm across band-h.
Family 2 = the two interactions, Holm within family 2. Family 2 is secondary and is reported
as such regardless of family 1's outcome.

### Interpretation table — fixed before results exist

| outcome | conclusion |
|---|---|
| A ≈ B2 ≈ B3 at k=8 | **Null.** δ_s feedback does nothing on a calibrated task where it had room. |
| A > B2, A ≈ B3 | Confound: a structured extra recurrent input helps; self-ness does not. |
| A > B2, A > B3, interactions null | Self-prediction error helps, but not *because* of the horizon. |
| A > B2, A > B3, interactions positive | **Hypothesis supported.** The compressed-summary account survives. |
| A < B2 | δ_s feedback destabilises. Report; inspect ‖δ_s‖ and clip rate. |

---

## 9. Probes and coupling metrics

**Linear probes (§v1.8 verbatim).** Single linear layer, `h → 2` logits, one per chain
belief target. Trained after training completes on a frozen model; no gradient reaches the
model. AdamW lr 1e−2, 2,000 full-batch steps. Split: fit on held-out sequences 0–47, report
on 48–63, burn-in discarded. Accuracies are reported against the **exact Bayes decode
ceiling**, not against chance alone.

**Coupling metrics**, canonical implementations in `coupling.py`, same split:

| | quantity |
|---|---|
| M1 | `corr(|δ_w|, ‖δ_s‖₂)` |
| M2 | held-out R² of a linear map scalar `δ_w^t` → vector `δ_s^(k)` |
| M3 | `‖Cov(δ_s, δ_w)‖₂ / √(Var(δ_w)·tr Cov(δ_s))` |
| M4 | M1–M3 after same-timestep batch permutation of δ_w |

M2 and M3 are algebraically linked (in-sample `M3² == R²`) and are **not independent
evidence**.

**Window coupling** — recoverability of `δ_s^(k)` from the full window
`(δ_w^t … δ_w^{t+k−1})`. This is the check that the horizon buys *temporal compression*
rather than new information: conditional on history the window determines δ_s^(k) exactly,
so a high value is the expected and confirming result.

Because the claim is informational rather than about linear accessibility, the primary
readout is a **small MLP, frozen here before any Stage 2 result exists**:

```
Linear(k → 16) → Tanh → Linear(16 → h)
AdamW lr 1e−2, weight decay 1e−4, 2,000 full-batch steps
fit on held-out sequences 0–47, reported on 48–63, burn-in discarded
```

Deliberately small — at most ~256 parameters against ~172,000 training samples, so it
measures recoverability and cannot memorise the held-out trajectories. The linear map is
retained as a **secondary** diagnostic. At k=1 window coupling and current-δ_w coupling
coincide by definition.

---

## 10. Stopping and run validity

No early stopping; every run executes 30,000 steps. **No seed replacement** — an outlier
stays in; Wilcoxon absorbs it. A run is FAILED on non-finite training *or eval* loss, or eval
loss worse than ln 2 at step 10,000. If >2 of N seeds fail in any cell, that cell is reported
**unstable**; failures are not replaced. Convergence gate: mean eval loss must change < 1%
between 25k and 30k in every cell, or the comparison is invalid and the budget extends for
**all** cells.

---

## 11. Logged every eval, every cell

eval loss and regret · **raw and normalised** aux loss · ‖δ_s^(k)‖ · ‖h_{t+1} − h_t‖ (freeze
detector) · ‖h‖ (state-norm growth) · gradient norm pre-clip and clip rate · FIFO first
consumption step vs expected · number of δ_s consumptions.

Reported per horizon at the end: self-prediction R² · coupling to the current δ_w^t ·
coupling to the full δ_w window · treatment effect by hidden size.

Every failure mode in this design produces clean-looking numbers. Without these, a result is
uninterpretable and that fact is invisible.

---

## 12. Systems validity — passed before this protocol was frozen

The pilot ran B2/B3/A × k∈{1,8} × h=8 × 4 paired seeds through the **exact Stage 2 driver**,
full 30k schedule, and passed seven pre-defined conditions: FIFO timing exact in all 24
cells; no NaN or runaway ‖δ_s‖; A shows no clipping or state-norm growth absent in B2/B3
(clip 0.005 vs 0.008/0.005; h_norm 0.63 vs 0.61/0.62); B3's slot equals
`normalise(roll(matured, 1))` exactly; raw and normalised aux finite; evaluation
deterministic on replication; 24/24 complete.

Pilot seeds (401–404) and checkpoints are retired and are **not reused in Stage 2**. The
pilot selected nothing — architecture, λ, h, delay, normalisation and both thresholds were
frozen before it ran.

---

## 13. Compute

Band h ∈ {4, 6, 8, 12} × 2 horizons × 4 conditions × N seeds. At 10 seeds: **320 runs**, at
~1,140 s/run on 12 workers ≈ **8.4 hours**.
