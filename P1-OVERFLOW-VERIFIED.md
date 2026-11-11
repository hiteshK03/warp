# Verification of the overflow review (executed). Repro: bench/p1_verify_overflow.py

| claim | verdict | measured |
|---|---|---|
| **M1**: inv(1e13*M) NaN is correct IEEE (inf + -inf in det), so float32 gives NaN too and the guard cannot fix it | **PARTLY REFUTED** | float32 det = **-inf** (not NaN); df64 det = **NaN**. So it IS df64-specific. The reviewer's predicted mixed-sign cancellation does not occur at float32 evaluation order. (float32's -inf is also wrong: true det = +4e39.) |
| **M2**: diag(1e20) cofactors overflow -> NaN even though the true inverse 1e-20 is representable | **CONFIRMED, and not df64-specific** | true inverse diag 1e-20; **both** df64 and float32 give NaN |
| **M3**: signed zero lost | **CONFIRMED** | `f64(-1)*f64(0)` = **+0** (IEEE -0); `1/that` = NaN (IEEE -inf); `copysign(1,-0)` = **+1** (IEEE -1) |
| **M4**: 1/inf gives NaN | **CONFIRMED** | `1/(3e38*3e38)` = NaN (IEEE 0) |

## Net effect on the plan

- The overflow guard IS worth doing and DOES fix df64-specific NaN (M1, M3, M4).
  My "prerequisite" framing was partly justified, contrary to the review.
- But the review's deeper point stands: **pre-scaling is the only thing that fixes
  the cofactor/determinant overflow class** (M2), which afflicts float32 equally.
  The two fixes are complementary, not strictly ordered.
- Adopt the reviewer's `df_guard` post-check with a hi-word fallback rather than my
  naive output-only guard: the review showed concrete cases where mine fails
  (tail-normalize overflow -> {inf,NaN}; 1/inf -> NaN; inf+1 -> NaN).
- Use an **integer bit test** for finiteness, not `metal::isfinite`, since fast-math
  can fold FP predicates. My earlier canary only tested reassociation, which is a
  different flag — the review is right that it does not cover no-NaN/no-inf.

## Implementation decision

Implement `df_guard(r, naive)` per the review: if both words finite, return r;
else return {r.hi if r.hi is inf else naive, 0}, where `naive` is the same op on
hi words only (correct IEEE inf/NaN semantics). Apply to +, -, *, /, sqrt.
Verify with runtime-loaded values (not constants, which fold regardless).
