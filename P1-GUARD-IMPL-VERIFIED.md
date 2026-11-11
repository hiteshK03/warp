# Verification of the df_guard IMPLEMENTATION review

Repro: bench/p1_verify_guard_impl.py. Inputs loaded from buffers at runtime;
nonzero lo words built in-kernel; signed zero compared by BIT PATTERN.

| claim | verdict | measured |
|---|---|---|
| **I1 (HIGH)**: finite/finite division near FLT_MAX returns −inf | **CONFIRMED — REGRESSION I INTRODUCED** | `FLT_MAX / 1.9999998` = **−inf**, expected +1.7014120e38. Before the guard this was NaN; the guard turned it into a confidently wrong, wrong-sign answer |
| **I2**: sqrt(FLT_MAX)^2 spuriously overflows | **CONFIRMED** | got **inf**, expected FLT_MAX |
| **I2 add**: three-term sum with nonzero lo spuriously overflows | **CONFIRMED** | got **inf**, expected FLT_MAX |
| **I5**: signed zero lost | **CONFIRMED (all four)** | float32(-0.0)=0x0; (-1)*0=0x0; 1/((-1)*0)=**0x7f800000 (+inf)** want 0xff800000; sqrt(-0)=0x0 |
| **I6**: sqrt(subnormal) gives +inf | **REFUTED** | 1e-40f -> 0.0 |
| floor/int collapse to hi+lo | **REFUTED** | floor(2^30−0.25) and int() both correct (1073741823) |
| mod is floored-modulo, not fmod | **CONFIRMED** | mod(−1,3)=2 (fmod: −1); mod(1,inf)=**NaN** (fmod: 1) |

## My claim "IEEE inf/NaN semantics restored" was overstated

The guard fixes inf/NaN *inputs* but mishandles *intermediate* overflow with finite
inputs. Its stated premise — "an infinite hi only arises when finite inputs
overflowed in the tail normalize" — is FALSE: the infinity can come from a residual
contaminated by an inner op's overflow (I1), or from the leading EFT overflowing
while the true value is still in range (I2).

Worse, I1 is a **regression**: NaN -> wrong-sign infinity is a downgrade in
usefulness, because NaN at least propagates visibly.

## My test harness was blind in exactly the ways the review said

- every input had lo == 0, so cross-term and tail paths never ran
- no compound expressions ((x/y)*y, sqrt(x)*sqrt(x), three-term sums)
- `g == r` treats −0 == +0, hiding all signed-zero defects
- no values within an ulp of FLT_MAX, so "no regressions" said nothing about the
  region where the guard actually misbehaves

## Required fixes (from the review, now justified by measurement)

1. `df_guard`: keep an infinite `r.hi` ONLY if `naive` is also infinite (or
   |naive| == FLT_MAX with matching sign). Otherwise return `naive`. This alone
   fixes I1's wrong sign.
2. `operator/`: when |a.hi| >= 2^126, compute `2*((0.5*a)/b)` so the residual
   cannot overflow.
3. I2-class spurious overflow needs power-of-two pre-scaling; until then, document
   "values within ~1 ulp of FLT_MAX may spuriously overflow" and fix the comment.
4. Signed zero: preserve the sign in normalize/float conversion, or document.
5. `mod`: match Warp's float semantics (fmod), and handle inf.
6. Harness: nonzero lo operands, compound expressions, bit-pattern comparison,
   near-FLT_MAX sweep, hi/lo invariant check (non-finite => lo == 0).
