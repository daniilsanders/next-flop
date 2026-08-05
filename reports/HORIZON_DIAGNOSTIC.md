# Horizon Diagnostic — does multi-step self-prediction decouple δ_s from δ_w?

**Zero training.** Uses the 90 Stage 1 B2 checkpoints. Diagnostic; sets no threshold.

## Motivation

Conditional on history, one-step δ_s is a *deterministic function* of δ_w: `h_{t+1}` depends
on the world only through `x_{t+1}`, and `p_s` predicts exactly one step, so both errors are
invertible functions of the same bit. That is architectural — no latent structure fixes it.
At horizon k, `h_{t+k}` depends on `x_{t+1..t+k}` while δ_w^t still carries one bit.

## Method

For each k, an idealised k-step self-predictor is the best **linear** map `m_t -> h_{t+k}`,
fit on the §8 train split. Its held-out residual is the δ_s^(k) proxy. Reported:

- **coupling** — held-out R² of a linear map `δ_w^t -> δ_s^(k)`, with the same-timestep
  batch-permutation baseline used in `coupling.py`
- **self-predictability** — held-out R² of `m_t -> h_{t+k}`. If this collapses, δ_s^(k)
  degenerates into "the state" rather than "the surprise" and a large k buys nothing.

## Result

| k | coupling | shuffle | self-predictability | ratio |
|---|---|---|---|---|
| 1 | **0.7024** | -0.0001 | 0.912 | 1.3 |
| 2 | **0.3859** | -0.0001 | 0.832 | 2.2 |
| 4 | **0.1422** | -0.0001 | 0.715 | 5.0 |
| 8 | **0.0488** | -0.0001 | 0.566 | 11.6 |
| 16 | **0.0135** | -0.0001 | 0.423 | 31.4 |
| 32 | **0.0039** | -0.0001 | 0.305 | 79.1 |

Averaged over all 90 checkpoints. Coupling falls **180×** from k=1 to k=32; 
self-predictability falls only **3×**. Fitted decay: coupling ~ k^-1.50, self-predictability
~ k^-0.32.

## By hidden size

| h | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|
| 2 | 0.817 / 0.89 | 0.349 / 0.82 | 0.148 / 0.69 | 0.055 / 0.53 | 0.017 / 0.36 | 0.005 / 0.24 |
| 3 | 0.734 / 0.90 | 0.325 / 0.84 | 0.137 / 0.74 | 0.054 / 0.60 | 0.017 / 0.44 | 0.006 / 0.30 |
| 4 | 0.716 / 0.90 | 0.345 / 0.83 | 0.133 / 0.71 | 0.046 / 0.57 | 0.013 / 0.43 | 0.004 / 0.31 |
| 6 | 0.736 / 0.89 | 0.337 / 0.81 | 0.134 / 0.70 | 0.045 / 0.55 | 0.012 / 0.41 | 0.003 / 0.30 |
| 8 | 0.679 / 0.90 | 0.385 / 0.82 | 0.140 / 0.71 | 0.046 / 0.56 | 0.012 / 0.42 | 0.003 / 0.31 |
| 12 | 0.637 / 0.92 | 0.425 / 0.83 | 0.145 / 0.71 | 0.047 / 0.56 | 0.012 / 0.42 | 0.003 / 0.31 |
| 16 | 0.705 / 0.92 | 0.408 / 0.83 | 0.139 / 0.70 | 0.047 / 0.55 | 0.012 / 0.42 | 0.003 / 0.30 |
| 24 | 0.710 / 0.93 | 0.428 / 0.85 | 0.147 / 0.72 | 0.049 / 0.57 | 0.013 / 0.43 | 0.003 / 0.31 |
| 32 | 0.700 / 0.94 | 0.453 / 0.86 | 0.154 / 0.74 | 0.051 / 0.59 | 0.013 / 0.45 | 0.003 / 0.33 |
| 64 | 0.591 / 0.93 | 0.406 / 0.84 | 0.145 / 0.73 | 0.048 / 0.58 | 0.013 / 0.45 | 0.003 / 0.33 |

*coupling / self-predictability*

The curves are **essentially identical at every h**, from 119 parameters to 44k. This is an
architectural property, not a capacity-dependent one, so it holds wherever the capacity band
eventually lands.

## What this does and does not establish

**Does:** a k-step self-prediction target makes δ_s largely unrecoverable from any *single*
world-prediction error, at modest cost in how much of the state a self-model can still
predict. k=8 removes 93% of the coupling while retaining 62% of the self-predictability.

**Does not:** make δ_s carry information absent from the world-error *stream*. Conditional on
history, the sequence δ_w^t..δ_w^{t+k-1} determines every `x` in the window and therefore
determines δ_s^(k) exactly. The horizon makes δ_s an **accumulated summary that g would
otherwise have to construct across k steps from one-step errors** — which is the capacity-
shortcut hypothesis, not an independent-information claim. Stating it the stronger way would
repeat the error this project has already made once.

**Caveats.** (1) The self-predictor here is linear, so self-predictability is a lower bound —
a trained nonlinear `p_s` would do better. (2) These are models trained with a k=1
objective; a system trained at k=8 has different dynamics and is a different measurement.

## Suggested parameter

`k = 8` as the frozen horizon for Protocol v2: coupling 0.049 against a shuffle floor of
-0.0001, self-predictability 0.566. k=16 pushes coupling to 0.014 but leaves the self-model
explaining under half the variance.
