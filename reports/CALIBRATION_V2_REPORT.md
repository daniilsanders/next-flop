# Calibration v2 Report — K-chain environment

**Environment design, not evidence.** Disposable by construction: calibration-only seeds [201, 202, 203], never reused in v1, Stage 1 or Stage 2; no checkpoints kept. The only surviving output is a frozen (K, ε).

Calibration v1 is **not** loaded, reused, or averaged in. It is preserved separately as `runs/calib/CALIBRATION_V1_REPORT.md`, marked superseded.

- runs: 90 of 90 · converged: 90/90
- K = **16** · h ∈ [2, 4, 6, 8, 12, 16, 24, 32, 48, 64] · 30,000 steps · Protocol-v2 optimizer and eval schedule
- eligibility (unchanged from v1): smallest h ≥ 0.8, largest h ≤ 0.05, ≥3 adjacent in [0.2, 0.8], monotone
- selection: most adjacent → lowest mean seed SD → tie-break [0.125, 0.0625, 0.25]

## Regret by εK and h

| εK | h=2 | h=4 | h=6 | h=8 | h=12 | h=16 | h=24 | h=32 | h=48 | h=64 | band | adj | seed SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0625 | 0.880 | **0.757** | **0.650** | **0.533** | **0.299** | 0.063 | 0.036 | 0.009 | 0.006 | 0.004 | [4, 6, 8, 12] | 4 | 0.0060 |
| 0.125 | 0.882 | **0.758** | **0.645** | **0.528** | **0.297** | 0.060 | 0.040 | 0.010 | 0.007 | 0.004 | [4, 6, 8, 12] | 4 | 0.0041 |
| 0.25 | 0.881 | **0.763** | **0.646** | **0.530** | **0.294** | 0.059 | 0.036 | 0.019 | 0.007 | 0.005 | [4, 6, 8, 12] | 4 | 0.0023 |

*bold = inside the capacity band*

## Eligibility

| εK | monotone | smallest h ≥ 0.80 | largest h ≤ 0.05 | ≥3 adjacent | eligible |
|---|---|---|---|---|---|
| 0.0625 | ✓ | ✓ (0.8801) | ✓ (0.0039) | ✓ (4) | **yes** |
| 0.125 | ✓ | ✓ (0.8824) | ✓ (0.0044) | ✓ (4) | **yes** |
| 0.25 | ✓ | ✓ (0.8807) | ✓ (0.0047) | ✓ (4) | **yes** |

## What v1 could not separate

v1 left convergence and capacity ceiling confounded: K=16/h=32 finished at 0.0527 after 8k steps, still descending, missing the 0.05 ceiling by 0.0004. v2 answers it:

- εK=0.0625: h=32 → 0.0094, h=64 → 0.0039 · **undertraining** — h=32 crosses the ceiling once trained to 30k
- εK=0.125: h=32 → 0.0103, h=64 → 0.0044 · **undertraining** — h=32 crosses the ceiling once trained to 30k
- εK=0.25: h=32 → 0.0193, h=64 → 0.0047 · **undertraining** — h=32 crosses the ceiling once trained to 30k

## Chains held per state dimension

The marginal-value property, re-measured at full training. Chains held = (1 − regret)·K.

- **εK=0.0625** — h=2:0.96, h=4:0.97, h=6:0.93, h=8:0.93, h=12:0.93, h=16:0.94, h=24:0.64*, h=32:0.50*, h=48:0.33*, h=64:0.25*  → mean 0.95 before saturation
- **εK=0.125** — h=2:0.94, h=4:0.97, h=6:0.95, h=8:0.94, h=12:0.94, h=16:0.94, h=24:0.64*, h=32:0.49*, h=48:0.33*, h=64:0.25*  → mean 0.95 before saturation
- **εK=0.25** — h=2:0.95, h=4:0.95, h=6:0.94, h=8:0.94, h=12:0.94, h=16:0.94, h=24:0.64*, h=32:0.49*, h=48:0.33*, h=64:0.25*  → mean 0.94 before saturation

*\* saturated: holding >95% of chains*

## Selection

### K = 16, εK = 0.25 (ε = 0.015625), emissions 0.1/0.9

- band: h ∈ [4, 6, 8, 12] (4 adjacent)
- mean between-seed SD over the band: **0.0023**
- smallest h 0.8807 · largest h 0.0047
- regret denominator: see `envk`; seed SD is far below the v1 power-gate stop threshold of 0.0668

**Frozen for Protocol v2.** Stage 1 re-selects λ under this environment; the provisional λ used in calibration is not carried forward.

