# Calibration v1 Report — SUPERSEDED

> **Conclusion.** These 216 runs validated the *mechanism*: the K-chain environment creates
> the intended capacity bottleneck, regret decreases monotonically with h at all nine
> settings, and chains held per state dimension is ~0.89 and constant — the marginal-value
> property the old environment lacked.
>
> They were **not adequate to evaluate the convergence condition**, for two reasons fixed in
> Calibration v2: the 8,000-step training budget was deliberately short (K=16/h=32 finished
> at regret 0.0504, still descending, missing the 0.05 ceiling by 0.0004), and the h sweep
> was inherited from v1 of the protocol and truncated at 32, which is only two state
> dimensions per chain at K=16. Convergence and capacity ceiling are inseparable in this
> grid.
>
> **These runs are superseded and are not used in any selection.** They are not averaged or
> combined with Calibration v2. No threshold was changed in response to them.

---

# Calibration Report — K-chain environment

**Environment design, not evidence.** All runs here are disposable: calibration-only seeds [101, 102, 103] never reused in Stage 1 or 2, no checkpoints kept, 8000 training steps chosen to rank settings rather than establish performance. The only output that survives is a frozen (K, ε).

- runs: 216 of 216
- band: [0.2, 0.8]  ·  smallest-h floor: 0.8  ·  largest-h ceiling: **0.05**  *(as frozen)*
- rule: most adjacent h in band → lowest mean seed SD → smaller K

## Regret by setting

| K | εK | h=2 | h=4 | h=6 | h=8 | h=12 | h=16 | h=24 | h=32 | band | adj | seed SD | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.0625 | **0.755** | **0.521** | **0.292** | 0.068 | 0.015 | 0.008 | 0.006 | 0.005 | [2, 4, 6] | 3 | 0.0055 | — |
| 8 | 0.125 | **0.758** | **0.519** | **0.295** | 0.060 | 0.014 | 0.009 | 0.008 | 0.006 | [2, 4, 6] | 3 | 0.0067 | — |
| 8 | 0.25 | **0.768** | **0.520** | **0.281** | 0.055 | 0.012 | 0.008 | 0.007 | 0.006 | [2, 4, 6] | 3 | 0.0061 | — |
| 16 | 0.0625 | 0.891 | **0.775** | **0.675** | **0.560** | **0.333** | 0.106 | 0.092 | 0.053 | [4, 6, 8, 12] | 4 | 0.0033 | — |
| 16 | 0.125 | 0.893 | **0.777** | **0.672** | **0.553** | **0.324** | 0.097 | 0.083 | 0.050 | [4, 6, 8, 12] | 4 | 0.0037 | — |
| 16 | 0.25 | 0.895 | **0.773** | **0.673** | **0.554** | **0.315** | 0.101 | 0.075 | 0.053 | [4, 6, 8, 12] | 4 | 0.0063 | — |
| 32 | 0.0625 | 0.954 | 0.904 | 0.855 | 0.804 | **0.699** | **0.586** | **0.362** | 0.152 | [12, 16, 24] | 3 | 0.0053 | — |
| 32 | 0.125 | 0.948 | 0.903 | 0.851 | **0.798** | **0.699** | **0.582** | **0.352** | 0.143 | [8, 12, 16, 24] | 4 | 0.0039 | — |
| 32 | 0.25 | 0.953 | 0.902 | 0.847 | 0.801 | **0.683** | **0.579** | **0.343** | 0.121 | [12, 16, 24] | 3 | 0.0034 | — |

*bold = inside the capacity band*

## Eligibility detail

| K | εK | monotone | smallest h ≥ 0.80 | largest h ≤ ceiling | ≥3 adjacent |
|---|---|---|---|---|---|
| 8 | 0.0625 | ✓ | ✗ (0.755) | ✓ (0.005) | ✓ (3) |
| 8 | 0.125 | ✓ | ✗ (0.758) | ✓ (0.006) | ✓ (3) |
| 8 | 0.25 | ✓ | ✗ (0.768) | ✓ (0.006) | ✓ (3) |
| 16 | 0.0625 | ✓ | ✓ (0.891) | ✗ (0.053) | ✓ (4) |
| 16 | 0.125 | ✓ | ✓ (0.893) | ✗ (0.050) | ✓ (4) |
| 16 | 0.25 | ✓ | ✓ (0.895) | ✗ (0.053) | ✓ (4) |
| 32 | 0.0625 | ✓ | ✓ (0.954) | ✗ (0.152) | ✓ (3) |
| 32 | 0.125 | ✓ | ✓ (0.948) | ✗ (0.143) | ✓ (4) |
| 32 | 0.25 | ✓ | ✓ (0.953) | ✗ (0.121) | ✓ (3) |

## Does regret track the fraction of beliefs that do not fit?

The postmortem's prediction, and the reason this environment was built. Chains effectively held = (1 − regret)·K.

- **K=8, εK=0.0625** — chains per state dimension: h=2:0.98, h=4:0.96, h=6:0.94, h=8:0.93, h=12:0.66*, h=16:0.50*, h=24:0.33*, h=32:0.25*  → mean 0.95 before saturation
- **K=8, εK=0.125** — chains per state dimension: h=2:0.97, h=4:0.96, h=6:0.94, h=8:0.94, h=12:0.66*, h=16:0.50*, h=24:0.33*, h=32:0.25*  → mean 0.95 before saturation
- **K=8, εK=0.25** — chains per state dimension: h=2:0.93, h=4:0.96, h=6:0.96, h=8:0.94, h=12:0.66*, h=16:0.50*, h=24:0.33*, h=32:0.25*  → mean 0.95 before saturation
- **K=16, εK=0.0625** — chains per state dimension: h=2:0.87, h=4:0.90, h=6:0.87, h=8:0.88, h=12:0.89, h=16:0.89, h=24:0.61, h=32:0.47  → mean 0.80 before saturation
- **K=16, εK=0.125** — chains per state dimension: h=2:0.86, h=4:0.89, h=6:0.88, h=8:0.89, h=12:0.90, h=16:0.90, h=24:0.61, h=32:0.47  → mean 0.80 before saturation
- **K=16, εK=0.25** — chains per state dimension: h=2:0.84, h=4:0.91, h=6:0.87, h=8:0.89, h=12:0.91, h=16:0.90, h=24:0.62, h=32:0.47  → mean 0.80 before saturation
- **K=32, εK=0.0625** — chains per state dimension: h=2:0.73, h=4:0.77, h=6:0.77, h=8:0.79, h=12:0.80, h=16:0.83, h=24:0.85, h=32:0.85  → mean 0.80 before saturation
- **K=32, εK=0.125** — chains per state dimension: h=2:0.83, h=4:0.78, h=6:0.80, h=8:0.81, h=12:0.80, h=16:0.84, h=24:0.86, h=32:0.86  → mean 0.82 before saturation
- **K=32, εK=0.25** — chains per state dimension: h=2:0.75, h=4:0.78, h=6:0.82, h=8:0.80, h=12:0.85, h=16:0.84, h=24:0.88, h=32:0.88  → mean 0.82 before saturation

*\* saturated: holding >95% of chains, so per-dimension figure is floored by K*

## Selection

**No eligible setting.** No (K, εK) satisfied all three conditions with ≥3 adjacent h in the band. The environment is not ready to freeze; report and stop rather than widening the band.

