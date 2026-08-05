# Discarded — wrong lambda

These 48 Stage 2 records ran at lambda=0.1 while Protocol v2 §4's selection chose 0.03.

Cause: `STAGE2` set `lam=_selected_lambda()` but not `lams`, and `jobs()` builds the
per-run lambda from `lams`, which defaulted to `(0.1,)`. The selected value sat unused.

They are quarantined rather than deleted so the deviation is visible, and are NOT used in
any analysis. Stage 2 was restarted from zero with lambda=0.03 on every run.
