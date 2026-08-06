# Stage 2 Results

Decision rule applied exactly as frozen in `PROTOCOL_V2.md` §8. Wilcoxon signed-rank, two-sided, paired by seed; Holm–Bonferroni across band-h. Improvement is `regret_control − regret_A`, so **positive means A is better**.

- runs: 320/320 ok, 0 failed · all converged · λ = 0.03 · FIFO mismatches 0
- commit `e8edca24761b` · band h ∈ [4, 6, 8, 12] · 10 paired seeds

## Cell means (regret, lower is better)

| h | B1 k=1 | B1 k=8 | B2 k=1 | B2 k=8 | B3 k=1 | B3 k=8 | A k=1 | A k=8 |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.7700 | 0.7639 | 0.7663 | 0.7644 | 0.7762 | 0.7752 | 0.7712 | 0.7703 |
| 6 | 0.6546 | 0.6523 | 0.6550 | 0.6502 | 0.6605 | 0.6620 | 0.6608 | 0.6555 |
| 8 | 0.5402 | 0.5368 | 0.5376 | 0.5362 | 0.5467 | 0.5445 | 0.5448 | 0.5434 |
| 12 | 0.3082 | 0.3058 | 0.3077 | 0.3046 | 0.3152 | 0.3113 | 0.3127 | 0.3060 |

## PRIMARY — A vs B2 at k=8

| h | mean improvement | sd | p (raw) | p (Holm) | ≥ threshold |
|---|---|---|---|---|---|
| 4 | -0.0059 | 0.0084 | 0.0488 | 0.0977 | no |
| 6 | -0.0053 | 0.0064 | 0.0273 | 0.0820 | no |
| 8 | -0.0072 | 0.0058 | 0.0059 | 0.0234 | no |
| 12 | -0.0014 | 0.0082 | 0.6250 | 0.6250 | no |

**Declared: False** (needs Holm p < 0.05 AND improvement ≥ 0.05 at ≥1 h, AND same sign at ≥ half the band; positive at 0/4)

## GATE — A vs B3 at k=8

| h | mean improvement | sd | p (raw) | p (Holm) | ≥ threshold |
|---|---|---|---|---|---|
| 4 | +0.0049 | 0.0088 | 0.0371 | 0.0742 | no |
| 6 | +0.0066 | 0.0050 | 0.0059 | 0.0234 | no |
| 8 | +0.0010 | 0.0073 | 0.6250 | 0.6250 | no |
| 12 | +0.0052 | 0.0056 | 0.0195 | 0.0586 | no |

**Declared: False** (needs Holm p < 0.05 AND improvement ≥ 0.05 at ≥1 h, AND same sign at ≥ half the band; positive at 4/4)

## reference — A vs B2 at k=1 (known-negative arm)

| h | mean improvement | sd | p (raw) | p (Holm) | ≥ threshold |
|---|---|---|---|---|---|
| 4 | -0.0048 | 0.0065 | 0.0371 | 0.0371 | no |
| 6 | -0.0058 | 0.0060 | 0.0137 | 0.0273 | no |
| 8 | -0.0071 | 0.0048 | 0.0020 | 0.0078 | no |
| 12 | -0.0050 | 0.0022 | 0.0020 | 0.0078 | no |

**Declared: False** (needs Holm p < 0.05 AND improvement ≥ 0.05 at ≥1 h, AND same sign at ≥ half the band; positive at 0/4)

## Interactions (secondary family, threshold 0.03)

### (A − B2)_k=8  >  (A − B2)_k=1

| h | improvement k=8 | improvement k=1 | difference | p (Holm) |
|---|---|---|---|---|
| 4 | -0.0059 | -0.0048 | -0.0010 | 1.0000 |
| 6 | -0.0053 | -0.0058 | +0.0005 | 1.0000 |
| 8 | -0.0072 | -0.0071 | -0.0001 | 1.0000 |
| 12 | -0.0014 | -0.0050 | +0.0036 | 1.0000 |

**Declared: False**

### (A − B3)_k=8  >  (A − B3)_k=1

| h | improvement k=8 | improvement k=1 | difference | p (Holm) |
|---|---|---|---|---|
| 4 | +0.0049 | +0.0051 | -0.0001 | 1.0000 |
| 6 | +0.0066 | -0.0003 | +0.0069 | 0.2578 |
| 8 | +0.0010 | +0.0019 | -0.0009 | 1.0000 |
| 12 | +0.0052 | +0.0025 | +0.0028 | 0.9668 |

**Declared: False**

## Verdict

### NULL

delta_s feedback does nothing on a calibrated task where it had room to succeed.

