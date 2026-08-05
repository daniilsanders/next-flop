# Stage 1 Postmortem — what does `h` actually encode?

**This is not part of Stage 1 selection.** It is a diagnostic run after the fact on the 90 Stage 1 checkpoints, to test whether the environment's predictive sufficient statistic is effectively one number. No checkpoint was retrained or altered; probe parameters are separate and no gradient reached a model.

## Setup

- checkpoints: 90 (Stage 1 B2, commit `fdbeda3fba19`)
- probe: PROTOCOL.md §8 verbatim — single linear layer `h -> 2`, AdamW lr 0.01, 2000 full-batch steps, frozen model
- data: held-out set, burn-in 512 discarded, fit on sequences 0–47, reported on 48–63
- `h_t` is the state that produces the prediction for `x_{t+1}`, so it is aligned with the Bayes *predictive* belief `b_{t+1|t}`; targets are `z_{t+1}`, `v_{t+1}`
- majority-class baselines: z **0.501**, v **0.574**
- **Bayes-optimal decode ceilings** (what the exact filter itself achieves on the same split): z **0.8611**, v **0.8706**. Both latents are substantially inferable; neither is unknowable. A raw accuracy is meaningless without these.
- integrity: the replicated forward pass reproduced each run's recorded final eval loss to <1e-5, so the probed states are the states that produced the results

## Predictions under test

1. `z` highly linearly decodable even at `h=2`
2. `v` much weaker, possibly near chance
3. predictive performance near Bayes despite weak `v` decoding
4. (stronger) linear reconstruction of the exact Bayes posterior `P(z=1)` from `h` should be near-perfect at `h=2`

## By hidden size (mean ± sd over 9 runs: 3 λ × 3 seeds)

| h | params | acc(z) | acc(v) | R²→P(z=1) | R²→P(v) | regret | state velocity |
|---|---|---|---|---|---|---|---|
| 2 | 119 | 0.858 ± 0.001 | 0.539 ± 0.039 | 0.9618 ± 0.0104 | 0.0345 ± 0.0742 | 0.0287 | 0.206 |
| 3 | 198 | 0.859 ± 0.001 | 0.689 ± 0.113 | 0.9731 ± 0.0142 | 0.3063 ± 0.2952 | 0.0168 | 0.218 |
| 4 | 293 | 0.860 ± 0.001 | 0.774 ± 0.129 | 0.9873 ± 0.0073 | 0.6220 ± 0.4283 | 0.0068 | 0.268 |
| 6 | 531 | 0.861 ± 0.000 | 0.857 ± 0.006 | 0.9861 ± 0.0072 | 0.9097 ± 0.0325 | 0.0015 | 0.389 |
| 8 | 833 | 0.861 ± 0.000 | 0.864 ± 0.002 | 0.9956 ± 0.0046 | 0.9373 ± 0.0171 | 0.0010 | 0.450 |
| 12 | 1753 | 0.861 ± 0.000 | 0.867 ± 0.002 | 0.9991 ± 0.0005 | 0.9666 ± 0.0119 | 0.0007 | 0.546 |
| 16 | 3009 | 0.861 ± 0.000 | 0.868 ± 0.002 | 0.9994 ± 0.0002 | 0.9735 ± 0.0041 | 0.0006 | 0.617 |
| 24 | 6529 | 0.861 ± 0.000 | 0.868 ± 0.001 | 0.9991 ± 0.0009 | 0.9795 ± 0.0081 | 0.0004 | 0.611 |
| 32 | 11393 | 0.861 ± 0.000 | 0.869 ± 0.001 | 0.9994 ± 0.0002 | 0.9802 ± 0.0040 | 0.0004 | 0.545 |
| 64 | 44289 | 0.861 ± 0.000 | 0.869 ± 0.001 | 0.9993 ± 0.0002 | 0.9808 ± 0.0028 | 0.0003 | 0.598 |

## Full table, by h, λ and seed

| h | λ | seed | regret | velocity | acc(z) | acc(v) | R²→P(z=1) | R²→P(v) |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.03 | 1 | 0.0253 | 0.200 | 0.858 | 0.534 | 0.9405 | 0.0684 |
| 2 | 0.03 | 2 | 0.0211 | 0.188 | 0.859 | 0.551 | 0.9687 | -0.0317 |
| 2 | 0.03 | 3 | 0.0210 | 0.191 | 0.858 | 0.491 | 0.9586 | -0.0341 |
| 2 | 0.1 | 1 | 0.0274 | 0.199 | 0.859 | 0.550 | 0.9684 | 0.1150 |
| 2 | 0.1 | 2 | 0.0318 | 0.191 | 0.856 | 0.586 | 0.9635 | 0.1566 |
| 2 | 0.1 | 3 | 0.0230 | 0.173 | 0.859 | 0.586 | 0.9748 | 0.0269 |
| 2 | 0.3 | 1 | 0.0303 | 0.190 | 0.857 | 0.482 | 0.9539 | -0.0362 |
| 2 | 0.3 | 2 | 0.0326 | 0.190 | 0.856 | 0.566 | 0.9696 | 0.0826 |
| 2 | 0.3 | 3 | 0.0455 | 0.332 | 0.855 | 0.509 | 0.9583 | -0.0365 |
| 3 | 0.03 | 1 | 0.0205 | 0.217 | 0.858 | 0.651 | 0.9752 | 0.1920 |
| 3 | 0.03 | 2 | 0.0190 | 0.258 | 0.859 | 0.525 | 0.9679 | -0.0069 |
| 3 | 0.03 | 3 | 0.0177 | 0.220 | 0.859 | 0.677 | 0.9854 | 0.2351 |
| 3 | 0.1 | 1 | 0.0018 | 0.184 | 0.861 | 0.861 | 0.9852 | 0.8993 |
| 3 | 0.1 | 2 | 0.0211 | 0.190 | 0.857 | 0.727 | 0.9578 | 0.3028 |
| 3 | 0.1 | 3 | 0.0173 | 0.281 | 0.860 | 0.634 | 0.9763 | 0.1502 |
| 3 | 0.3 | 1 | 0.0083 | 0.200 | 0.860 | 0.827 | 0.9862 | 0.6403 |
| 3 | 0.3 | 2 | 0.0257 | 0.212 | 0.857 | 0.556 | 0.9443 | 0.0029 |
| 3 | 0.3 | 3 | 0.0194 | 0.196 | 0.859 | 0.742 | 0.9792 | 0.3405 |
| 4 | 0.03 | 1 | 0.0166 | 0.257 | 0.858 | 0.679 | 0.9833 | 0.1715 |
| 4 | 0.03 | 2 | 0.0008 | 0.300 | 0.861 | 0.862 | 0.9902 | 0.9269 |
| 4 | 0.03 | 3 | 0.0020 | 0.256 | 0.861 | 0.857 | 0.9935 | 0.9192 |
| 4 | 0.1 | 1 | 0.0158 | 0.277 | 0.859 | 0.546 | 0.9820 | -0.0129 |
| 4 | 0.1 | 2 | 0.0025 | 0.302 | 0.861 | 0.856 | 0.9864 | 0.9058 |
| 4 | 0.1 | 3 | 0.0020 | 0.244 | 0.861 | 0.856 | 0.9954 | 0.9227 |
| 4 | 0.3 | 1 | 0.0168 | 0.264 | 0.859 | 0.601 | 0.9768 | 0.0077 |
| 4 | 0.3 | 2 | 0.0025 | 0.273 | 0.860 | 0.856 | 0.9804 | 0.8872 |
| 4 | 0.3 | 3 | 0.0025 | 0.239 | 0.860 | 0.857 | 0.9977 | 0.8703 |
| 6 | 0.03 | 1 | 0.0014 | 0.413 | 0.861 | 0.859 | 0.9933 | 0.9382 |
| 6 | 0.03 | 2 | 0.0021 | 0.454 | 0.860 | 0.855 | 0.9774 | 0.8566 |
| 6 | 0.03 | 3 | 0.0007 | 0.484 | 0.860 | 0.861 | 0.9886 | 0.9437 |
| 6 | 0.1 | 1 | 0.0017 | 0.359 | 0.860 | 0.858 | 0.9882 | 0.9093 |
| 6 | 0.1 | 2 | 0.0014 | 0.403 | 0.860 | 0.859 | 0.9912 | 0.9396 |
| 6 | 0.1 | 3 | 0.0003 | 0.344 | 0.861 | 0.861 | 0.9887 | 0.9258 |
| 6 | 0.3 | 1 | 0.0034 | 0.338 | 0.860 | 0.841 | 0.9924 | 0.8600 |
| 6 | 0.3 | 2 | 0.0024 | 0.394 | 0.861 | 0.857 | 0.9829 | 0.9032 |
| 6 | 0.3 | 3 | 0.0006 | 0.312 | 0.861 | 0.862 | 0.9721 | 0.9104 |
| 8 | 0.03 | 1 | 0.0010 | 0.400 | 0.861 | 0.865 | 0.9993 | 0.9400 |
| 8 | 0.03 | 2 | 0.0009 | 0.499 | 0.861 | 0.865 | 0.9973 | 0.9569 |
| 8 | 0.03 | 3 | 0.0004 | 0.486 | 0.861 | 0.865 | 0.9868 | 0.9200 |
| 8 | 0.1 | 1 | 0.0011 | 0.387 | 0.861 | 0.864 | 0.9993 | 0.9452 |
| 8 | 0.1 | 2 | 0.0007 | 0.485 | 0.861 | 0.866 | 0.9971 | 0.9573 |
| 8 | 0.1 | 3 | 0.0006 | 0.449 | 0.861 | 0.865 | 0.9896 | 0.9193 |
| 8 | 0.3 | 1 | 0.0016 | 0.362 | 0.861 | 0.864 | 0.9991 | 0.9478 |
| 8 | 0.3 | 2 | 0.0011 | 0.421 | 0.861 | 0.862 | 0.9985 | 0.9388 |
| 8 | 0.3 | 3 | 0.0019 | 0.560 | 0.861 | 0.860 | 0.9937 | 0.9099 |
| 12 | 0.03 | 1 | 0.0008 | 0.553 | 0.861 | 0.866 | 0.9991 | 0.9419 |
| 12 | 0.03 | 2 | 0.0004 | 0.497 | 0.862 | 0.866 | 0.9992 | 0.9724 |
| 12 | 0.03 | 3 | 0.0004 | 0.563 | 0.861 | 0.867 | 0.9995 | 0.9717 |
| 12 | 0.1 | 1 | 0.0007 | 0.549 | 0.861 | 0.866 | 0.9989 | 0.9561 |
| 12 | 0.1 | 2 | 0.0006 | 0.519 | 0.862 | 0.866 | 0.9992 | 0.9708 |
| 12 | 0.1 | 3 | 0.0005 | 0.591 | 0.861 | 0.865 | 0.9994 | 0.9695 |
| 12 | 0.3 | 1 | 0.0012 | 0.573 | 0.861 | 0.872 | 0.9978 | 0.9603 |
| 12 | 0.3 | 2 | 0.0011 | 0.500 | 0.861 | 0.865 | 0.9989 | 0.9780 |
| 12 | 0.3 | 3 | 0.0006 | 0.569 | 0.861 | 0.868 | 0.9995 | 0.9789 |
| 16 | 0.03 | 1 | 0.0009 | 0.583 | 0.861 | 0.868 | 0.9993 | 0.9708 |
| 16 | 0.03 | 2 | 0.0004 | 0.609 | 0.861 | 0.865 | 0.9996 | 0.9723 |
| 16 | 0.03 | 3 | 0.0003 | 0.735 | 0.861 | 0.866 | 0.9990 | 0.9695 |
| 16 | 0.1 | 1 | 0.0008 | 0.608 | 0.862 | 0.871 | 0.9996 | 0.9770 |
| 16 | 0.1 | 2 | 0.0007 | 0.644 | 0.861 | 0.868 | 0.9995 | 0.9765 |
| 16 | 0.1 | 3 | 0.0004 | 0.630 | 0.862 | 0.867 | 0.9993 | 0.9708 |
| 16 | 0.3 | 1 | 0.0008 | 0.594 | 0.861 | 0.869 | 0.9994 | 0.9764 |
| 16 | 0.3 | 2 | 0.0005 | 0.596 | 0.862 | 0.867 | 0.9993 | 0.9800 |
| 16 | 0.3 | 3 | 0.0003 | 0.554 | 0.861 | 0.869 | 0.9995 | 0.9679 |
| 24 | 0.03 | 1 | 0.0006 | 0.572 | 0.862 | 0.869 | 0.9997 | 0.9849 |
| 24 | 0.03 | 2 | 0.0002 | 0.594 | 0.861 | 0.868 | 0.9996 | 0.9822 |
| 24 | 0.03 | 3 | 0.0003 | 0.683 | 0.862 | 0.867 | 0.9982 | 0.9741 |
| 24 | 0.1 | 1 | 0.0007 | 0.570 | 0.861 | 0.869 | 0.9997 | 0.9758 |
| 24 | 0.1 | 2 | 0.0003 | 0.583 | 0.862 | 0.868 | 0.9995 | 0.9796 |
| 24 | 0.1 | 3 | 0.0002 | 0.626 | 0.861 | 0.866 | 0.9973 | 0.9620 |
| 24 | 0.3 | 1 | 0.0007 | 0.589 | 0.861 | 0.868 | 0.9997 | 0.9879 |
| 24 | 0.3 | 2 | 0.0003 | 0.604 | 0.862 | 0.870 | 0.9996 | 0.9872 |
| 24 | 0.3 | 3 | 0.0004 | 0.679 | 0.861 | 0.869 | 0.9984 | 0.9815 |
| 32 | 0.03 | 1 | 0.0005 | 0.552 | 0.862 | 0.868 | 0.9994 | 0.9827 |
| 32 | 0.03 | 2 | 0.0004 | 0.488 | 0.862 | 0.867 | 0.9996 | 0.9723 |
| 32 | 0.03 | 3 | 0.0004 | 0.657 | 0.861 | 0.870 | 0.9993 | 0.9814 |
| 32 | 0.1 | 1 | 0.0005 | 0.540 | 0.861 | 0.870 | 0.9992 | 0.9790 |
| 32 | 0.1 | 2 | 0.0004 | 0.469 | 0.861 | 0.867 | 0.9996 | 0.9795 |
| 32 | 0.1 | 3 | 0.0004 | 0.620 | 0.861 | 0.870 | 0.9992 | 0.9768 |
| 32 | 0.3 | 1 | 0.0007 | 0.532 | 0.862 | 0.868 | 0.9996 | 0.9864 |
| 32 | 0.3 | 2 | 0.0004 | 0.482 | 0.861 | 0.867 | 0.9994 | 0.9829 |
| 32 | 0.3 | 3 | 0.0002 | 0.564 | 0.861 | 0.869 | 0.9992 | 0.9811 |
| 64 | 0.03 | 1 | 0.0005 | 0.608 | 0.862 | 0.869 | 0.9991 | 0.9839 |
| 64 | 0.03 | 2 | 0.0004 | 0.645 | 0.862 | 0.868 | 0.9992 | 0.9803 |
| 64 | 0.03 | 3 | 0.0003 | 0.668 | 0.861 | 0.868 | 0.9995 | 0.9783 |
| 64 | 0.1 | 1 | 0.0006 | 0.581 | 0.861 | 0.868 | 0.9990 | 0.9822 |
| 64 | 0.1 | 2 | 0.0003 | 0.595 | 0.861 | 0.869 | 0.9992 | 0.9791 |
| 64 | 0.1 | 3 | 0.0002 | 0.629 | 0.861 | 0.870 | 0.9996 | 0.9794 |
| 64 | 0.3 | 1 | 0.0005 | 0.504 | 0.861 | 0.868 | 0.9993 | 0.9783 |
| 64 | 0.3 | 2 | 0.0002 | 0.568 | 0.861 | 0.871 | 0.9995 | 0.9865 |
| 64 | 0.3 | 3 | 0.0002 | 0.587 | 0.861 | 0.870 | 0.9995 | 0.9795 |

## Verdict

At **h=2** (119 parameters, 2 state dimensions), averaged over 9 runs:

| quantity | value | chance | Bayes ceiling | signal captured |
|---|---|---|---|---|
| acc(z) | **0.858** | 0.501 | 0.861 | **99%** |
| acc(v) | **0.539** | 0.574 | 0.871 | **-12%** |
| R² → exact Bayes P(z=1) | **0.9618** | 0 | 1.0 | — |
| R² → exact Bayes P(v) | **0.0345** | 0 | 1.0 | — |
| regret | **0.0287** | 1 = frozen | 0 = Bayes | — |

### Reading

**`z` is saturated from the smallest model.** At `h=2` the probe recovers **99%** of the decodable `z` signal and reconstructs the exact Bayes posterior `P(z=1)` linearly at R² = 0.962, from two state dimensions. It does not improve with more capacity — `acc(z)` is flat at the ceiling across the entire sweep.

**`v` is acquired progressively, not ignored.** R² → `P(v)` runs 0.03 → 0.31 → 0.62 → 0.91 across h = 2 → 3 → 4 → 6, and is essentially saturated by `h=6`. Volatility is **not** unknowable — the exact filter decodes it at 0.871 — and given room, the model does represent it.

**This corrects the working diagnosis.** The sufficient statistic is not one number; it is two, and the model acquires them in a strict order — `z` first, `v` only once `z` is free. The actual defect is the *marginal value* of the second dimension: going from no `v` at all (`h=2`, R²=0.03) to `v` almost fully represented (`h=6`, R²=0.91) moves regret from **0.0287 to 0.0015** — the entire second dimension is worth under 2.9% of the frozen-to-Bayes range.

So a maximally starved 119-parameter model is already ~97% of the way to optimal. There is no capacity deficit to find because there is nothing expensive to forget.

**Consequence for the redesign.** The target is not simply a higher-dimensional sufficient statistic — it is one where **each retained dimension carries large marginal value**, so that omitting a fraction of the state costs a comparable fraction of the achievable loss reduction. No emission setting can produce that: `p_low/p_high` changes how informative an observation is, not what dropping a dimension costs. K parallel chains have the required property directly — holding only j of K beliefs leaves the remaining (K−j)/K of steps predicted at chance, so regret should scale roughly as the fraction of state that does not fit, and the [0.20, 0.80] band should fall near h ∈ [0.2K, 0.8K].
