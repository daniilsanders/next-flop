# Stage 1 Report

- config: `protocol`
- runs planned 90, used in analysis 90, problems 0

## Provenance

- **training commit:** `fdbeda3fba19fe2c1effa2d518b057d3960d85e1`
- **working tree at launch:** DIRTY
- **code that ran vs. that commit:** IDENTICAL — every result-determining file matched (protocol_sha ok, code_sha ok)
  - the dirty flag counts any modified or untracked file; none of them were inputs to training
- protocol sha `f832b6545ab9b4f1`, code sha `03fa315e936c5698`
- training env: Python 3.13.2, PyTorch 2.11.0, NumPy 2.4.3
- analysis env: Python 3.13.2, NumPy 2.4.3, SciPy 1.18.0
- platform: macOS-26.5.2-arm64-arm-64bit-Mach-O (host `Mac`)
- observed max concurrency: **12 workers** (measured from run intervals; the driver does not record the worker count — fix before Stage 2)
- runs: 90/90 present, 90 valid, **0 invalid**, **0 failed**, 0 missing
- provenance mismatches across records: none — all records share one commit and code sha
- **no completed record rewritten after launch:** True
- analysis input digest (sha256 over the records used): `20b93b1f3dc55f77eb462a94e909fc22a3369a3847a4f299ed507cb8471043e0`
- per-record checksums: `completion.json`

## Validation coverage

The driver was exercised end-to-end through config-only variants: resume after SIGKILL, atomic writes, corrupt-record halt, failure recording without retry, λ tie-break, empty-band stop, and power-gate generation.

**Not exercised end-to-end:** the *continue* branch (a band found with required n ≤ 30). No smoke configuration produced a band with low enough seed variance to reach it, and no synthetic success case was fabricated to cover it. The gate arithmetic underlying that branch is unit-tested across all three outcomes in `power.py`.

## §6.1 lambda selection

Aggregate = mean regret over h with mean regret in [0.05, 0.95].

| lambda | h included | n | aggregate regret |
|---|---|---|---|
| 0.03 | — | 0 | — |
| 0.1 | — | 0 | — |
| 0.3 | — | 0 | — |

no lambda has any h inside the [0.05, 0.95] window. Selected λ = **None**

## Regret by hidden size (mean ± sd over seeds)

| h | λ=0.03 | λ=0.1 | λ=0.3 |
|---|---|---|---|
| 2 | 0.022 ± 0.002 | 0.027 ± 0.004 | 0.036 ± 0.008 |
| 3 | 0.019 ± 0.001 | 0.013 ± 0.010 | 0.018 ± 0.009 |
| 4 | 0.006 ± 0.009 | 0.007 ± 0.008 | 0.007 ± 0.008 |
| 6 | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.002 ± 0.001 |
| 8 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.002 ± 0.000 |
| 12 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 |
| 16 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.001 ± 0.000 |
| 24 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 32 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 64 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## §6.2 capacity band

Window [0.2, 0.8] at the selected λ. Band = **empty**

Escape hatch (§6.4): condition *task too easy (no h reaches regret 0.20)*, eligible=True, already used=False
→ weaken emissions to 0.35/0.65

## Decision

**ESCAPE_HATCH** — no valid lambda aggregate (no lambda has any h inside the [0.05, 0.95] window); task too easy (no h reaches regret 0.20); pre-registered action: weaken emissions to 0.35/0.65


Stage 1 ends here. Stage 2 is not started automatically.
