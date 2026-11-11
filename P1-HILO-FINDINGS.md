# Hi/lo readback metric — the measurement I should have had from the start

Repro: bench/p1_hilo_metric.py. 200 random SPD matrices per (scale, cond) bucket.
The kernel emits the inverse as TWO float32 arrays (hi, lo); the host rebuilds
hi+lo in float64 and compares against a float64 reference. This removes the
float32 output ceiling that made every earlier measurement uninformative.

| scale | cond | df64 median rel err | f32 median rel err | ratio median | ratio p99 |
|---|---|---|---|---|---|
| 1e0  | 1e2 | **9.24e-15** | 3.26e-07 | **2.5e7** | 8.9e8 |
| 1e0  | 1e6 | 1.01e-10 | 3.31e-03 | 4.0e7 | 1.2e9 |
| 1e-12 | 1e2 | **1.00e+00** | **1.00e+00** | 1.0 | 5.1 |
| 1e-12 | 1e6 | **1.00e+00** | **1.00e+00** | 1.0 | 1.0 |
| 1e12 | 1e2 | 3.10e-08 | 4.50e-07 | 15.6 | 325 |
| 1e12 | 1e6 | 2.09e-10 | 3.69e-03 | 1.4e7 | 2.7e9 |

## This overturns my previous claim in BOTH directions

1. **"df64 gains only 2-3.6x" was wrong and badly understated.** At unit scale df64
   is ~25,000,000x more accurate than float32 intermediates (9.2e-15 vs 3.3e-7) —
   genuine double-precision behaviour. The 2-3.6x figure was purely the float32
   output boundary, exactly as the reviewer suspected. My "honest, modest claim"
   was itself an artifact.

2. **At 1e-12 scale both paths are COMPLETELY WRONG, not merely degraded.**
   Relative error 1.0 = 100% wrong. I had described this as "df64 degrades to
   float32 precision"; in fact the inverse is garbage for both. This is the
   determinant-underflow / low-word-flush regime and it is far more severe than
   anything I reported.

3. **At 1e12 the df64 error (3.1e-8) is ~6 orders worse than at unit scale
   (9.2e-15)**, consistent with the nearby overflow cliff (det ~1e36-1e40).

## Consequences

- The case for pre-scaling is now much stronger: it is not a marginal precision
  tweak, it is the difference between a correct inverse and a 100%-wrong one for
  small-magnitude inertias.
- Any statement about df64 accuracy must specify the scale regime. A single
  number is meaningless.
- The overflow->NaN fix and pre-scaling are both prerequisites for reporting
  anything about small or large inertias.
- Newton's real robot inertias (~1e-3) sit in a regime I have NOT yet measured
  with this metric. That is the next measurement that actually matters, together
  with the reviewer's question of whether the float32 store discards the gain.
