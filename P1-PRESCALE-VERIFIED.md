# Verification of the pre-scaling review (executed, not accepted on faith)

Repro: bench/p1_verify_prescale.py, bench/p1_discriminating.py

| claim | verdict | measured |
|---|---|---|
| R2: max-abs scaling regresses diag(1e-20,1,1e20) | **CONFIRMED** | unscaled (today) = diag(1e20, 1, 1e-20) correct; max-abs scaled = **zero matrix** |
| R3: `_wp_native_fmax` drops NaN | **REFUTED** | `max(NaN,1)` = NaN (propagates). `max(1,NaN)` = 1, so the *asymmetry* is real but the "drops NaN" claim is wrong |
| R1: 1e-12 non-diagonal degrades to ~1e-7..1e-8 from low-word flushing | **NOT REPRODUCED** | unscaled rel_err 4.55e-8 at 1e-12, 2.17e-8 at scale 1 — i.e. float32 *output* rounding, and scaling changed nothing |

R2 is decisive: it rules out putting max-abs scaling in Warp's generic
`inverse_impl`, exactly as the reviewer argued.

## The methodological point stands, and I fixed it

The reviewer's real catch was that my validation could not discriminate: outputs
are float32, so every relative error bottoms out at ~6e-8 regardless of the
intermediate precision. My identity-matrix test had all-zero low words, making it
doubly blind.

New test (bench/p1_discriminating.py) compares df64 intermediates against a
**float32-intermediate control** running the same algorithm, and scores the
residual ||inv(A)@A - I|| in host float64:

| case | cond(A) | resid df64 | resid f32 | df64 better (>2x) |
|---|---|---|---|---|
| illcond 1e6 | 1.0e6 | 5.13e-3 | 7.62e-3 | no |
| near-singular | 9.4e6 | 0 | 0 | no |
| 1e-12 non-diag | 3.7 | 5.74e-8 | 1.15e-7 | no (1.99x, just under) |
| **1e12 non-diag** | 3.7 | **2.64e-8** | **9.56e-8** | **yes (3.6x)** |
| mixed SPD | 6.0 | 1.57e-8 | 1.57e-8 | no |

So df64 does measurably beat float32 intermediates, but only modestly (2-3.6x) and
only on some inputs — NOT the orders of magnitude implied by "44-48 mantissa bits".
The float32 output boundary caps the observable gain. This is the honest claim.

## Revised plan (pending review)

Per the review + verification:
1. DROP eig3 pre-scaling (Jacobi stopping tests are dimensionless; Newton already
   normalizes; no benefit).
2. DO NOT touch Warp's generic `inverse_impl` (R2 regression + hot-path cost +
   silent semantics change on CUDA/CPU).
3. Fix Newton's `_inverse_inertia_wp` with diagonal-max scaling guarded for
   zero/non-finite. Small, upstreamable on its own merits, helps float32 too.
4. Separately fix df64's overflow->NaN (confirmed earlier: 3e38*3e38, inf+1,
   1/0 all give NaN instead of inf).
5. Keep the zero-on-singular contract (point masses / kinematic bodies rely on it).
