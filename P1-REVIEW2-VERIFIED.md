# Verification of review #2 (executed). Repro: bench/p1_verify_review2.py

| claim | verdict | measured |
|---|---|---|
| **V1**: df64 low word flushed at 1e-12, so df64 degrades to float32 there | **CONFIRMED — my earlier "refutation" was WRONG** | det low word: scale 1e-12 -> **0**; scale 1e12 -> **-5.9e29** (alive) |
| **V2**: near-singular resid==0 is impossible / harness bug | **EXPLAINED, not a bug — but the case is degenerate** | inverse is exact dyadic ([[2097153,-1048576,...]]); inv@A is *exactly* I. float32 rounding made 1+1e-6 dyadic, so the case carries no information |
| **V4**: fast-math may fold the low word away | **PASS (not happening here)** | (1+1e-10)-1 = 1.000000e-10 |
| **R3**: NaN ordering diverges across backends | **REFUTED** | `wp.max/min` with NaN: cpu and metal both return 1.0 for all four orders — identical |
| **CLIFF**: det overflow yields a NaN matrix, not zeros | **CONFIRMED** | inv(1e13*M): host det 1.8e40 > FLT_MAX; df64 inverse diag = [nan,nan,nan] |

## Corrections to my own earlier claims

1. **"R1 not reproduced" was wrong.** I measured relative error on float32 outputs
   and concluded the degradation wasn't real. Reading back the determinant's low
   word shows it IS flushed at 1e-12. The 1.99x (1e-12) vs 3.6x (1e12) asymmetry in
   my own table was the predicted signal; I misread it as noise.
2. **The near-singular row of my table is uninformative**, not evidence. Exact-dyadic
   inverse => resid identically 0 for any precision. Must be replaced with a
   non-degenerate ill-conditioned case.
3. **My 1e12 "win" is one decade below an overflow cliff** where the same code path
   returns a NaN matrix. Reporting the win without the cliff was misleading.

## What this means for the plan

- The df64 precision story is scale-dependent in a way I had not characterized:
  below |det| ~ 2e-31 the low word flushes and df64 == float32. Power-of-two
  pre-scaling fixes this by keeping intermediates near unity — so scaling is
  motivated by PRECISION, not only underflow, as the reviewer argued.
- Overflow->NaN must be fixed BEFORE any scale-related claim is reported, because
  NaN det defeats the `det == 0` guard and silently yields a NaN inverse matrix.
- R3 is not a backend divergence; drop it from the work list (the df64 `_wp_native_fmax`
  comment is still imprecise, but behavior matches the reference through wp.max).

## Still outstanding from the review (not yet done)

- V3: ~1000 random matrices per (scale, cond) bucket, median + p99 ratio, instead of
  one sample and a hard 2x cutoff.
- Hi/lo readback metric: emit the inverse as two float32 arrays, rebuild in host
  float64, compare to a host float64 inverse — removes the output-rounding ceiling.
- The reviewer's closing question: Newton stores inverse inertia as float32 mat33,
  so df64 gains may be discarded at the store. Justify df64 on this path with the
  P0b per-step divergence probe over N steps vs a CPU float64 reference, not
  single-inverse residuals.
