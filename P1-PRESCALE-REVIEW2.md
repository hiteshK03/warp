**Verdict:** items 1, 2 and 5 are right. Item 3 is right, but for a different reason than you give. Item 4 is incomplete. The verification still has two blind spots, and both cut against your conclusions.

## Problems with the verification

**V1. Your R1 result is not a refutation. The test is blind again, and your own data points toward R1.**
A df64 low word is about `hi·2^-24`. It stays normal only when `|hi| ≳ 2^-102 ≈ 2e-31`. Below that, the low word is subnormal and Metal flushes it, so df64 drops to float32 precision.
- At the 1e-12 scale, `det ≈ 1e-36`, which is below the cutoff. df64 gains 1.99x there.
- At the 1e12 scale, `det ≈ 1e36`, with a full low word. df64 gains 3.6x there.

That asymmetry is what R1 predicts. Output rounding hid the rest.
- **Test:** read back the `det` low word for the 1e-12 non-diagonal case. Expected: nonzero. Predicted actual: 0.
- **Better metric:** write the inverse out as two float32 arrays (hi and lo), rebuild it in host f64, and compare against a host f64 inverse. Until you do, "df64 gains only 2-3.6x" is a property of the harness, not of df64. The claim could be wrong in either direction.

**V2. "Near-singular, resid = 0 for both" is a harness bug or a degenerate case.**
With cond 9.4e6 and float32 outputs, `||inv(A)@A − I||` cannot be exactly 0. There are three possibilities:
- the matrix is diagonal with power-of-two entries,
- the residual code is wrong,
- or the singular guard returned something your metric scores as 0.

Note that a zero matrix would score `||−I|| = √3`, not 0. Print `inv(A)` for this case. Until it's explained, treat the whole table as suspect.

**V3. One matrix per case and a hard 2x cutoff.**
1.99x vs 3.6x on a single sample means nothing. Run around 1000 random matrices per (scale, cond) bucket and report the median and p99 ratio.

**V4. Check that df64 is not being simplified away by the compiler.**
This is cheap, so do it before anything else. Metal's default `fastMathEnabled=YES` allows reassociation. That can fold two-sum's `(s − a) − b` error term to 0. FP contraction can also break a Dekker split if two-prod doesn't use an explicit `fma`.
- **Test:** df64 `(1.0 + 1e-10) − 1.0`. Expected: `1e-10`. If you get `0`, the compiler is eating the low word.
- Your modest gains could partly come from this.

## The plan

**Item 3 (Newton `_inverse_inertia_wp` scaling): do it, but the real motivation is a Metal/CUDA divergence that is already live.**
Without scaling, small bodies get zero inverse inertia on Metal:
- Take a 0.1 mg grain, radius 1 mm: `I ≈ 4e-14·eye(3)`.
- `det = 6.4e-41`, which is subnormal.
- CUDA (no ftz): inverse ≈ `2.5e13·eye(3)`, correct.
- Metal (FTZ): `det` flushes to 0, the zero-on-singular path fires, and the body gets infinite rotational inertia without any warning.

Add this case as a test. Scaling fixes it, and it also keeps df64 intermediates above the 2e-31 cutoff from V1.

Three constraints on how you do it:
- **Use a power-of-two scale, not `max(diag)`** (for example `2^floor(log2(m))`). Dividing by a non-power-of-two rounds the input. With a power of two, `inv(A/2^k)·2^-k` is bit-identical to `inv(A)` whenever nothing under- or overflows. That means no behavior change on CUDA or CPU for normal inputs, which is what makes it upstreamable. Counter-example for plain max: `diag(3,5,7)` scaled by 7 gives `3/7`, which rounds, so the result differs from today's in the last ulp.
- **State the precondition.** Diagonal-max bounds every entry only for SPD matrices: `|I_ij| ≤ √(I_ii·I_jj)`. R2 still applies to diagonal input. `diag(1e-20, 1, 1e20)` scaled becomes `diag(1e-40, 1e-20, 1)`, the first entry flushes, and you get a zero matrix. That input breaks the triangle inequality (`I1 ≤ I2 + I3`), so it is not physical. Document that, and don't claim the fix works for arbitrary 3x3 input.
- **The NaN guard has a bug, given the asymmetry you measured.** `max(max(I00, I11), I22)`:
  - `diag(NaN, 1, 1)`: the first `max` returns NaN, the guard fires, and the result is a zero inverse. Bad data silently becomes a kinematic body.
  - `diag(1, 1, NaN)`: `max(1, NaN) = 1`, the guard passes, and the result is a NaN inverse.

  Same bad input, two different outcomes depending on position. Check `isfinite` on every element explicitly, and pick one policy. Propagating NaN is the better choice.

**R3 follow-up: this is a separate backend bug, not a footnote.**
The measured `max(NaN,1) = NaN`, `max(1,NaN) = 1` must match Warp's CPU and CUDA builtin bit for bit. Run `wp.max(nan, 1.0)` and `wp.max(1.0, nan)` on cpu, cuda and metal. If Metal's order is the reverse of the reference (for example, the reference is `a > b ? a : b`), every NaN-producing kernel diverges across backends.

**Item 4 (df64 overflow giving NaN):** do it, with two cautions.
- **Root cause:** error-free transforms on `inf` produce `inf − inf`. In `1/0`, the Newton-step residual computes `inf·0`. The fix is: if `!isfinite(hi)`, set `lo = 0`, and do this in add, mul and div.
- **Under fast-math, `isfinite` and `isinf` may be folded to `true`.** Test the guard with fast-math on.
- **Test:** `3e38 * 3e38` should give `+inf`. Also `inv(1e13 * [[2,1,0],[1,2,1],[0,1,2]])`: `det ≈ 4e39` overflows. The current df64 gives NaN, and a NaN `det` fails a `det == 0` guard, so the output is a NaN matrix rather than zeros. Your 1e12 "win" sits about one decade below this cliff. Fix overflow before reporting any 1e12-scale result.

**Items 1, 2 and 5:** agreed. Dropping the eig3 scaling and leaving `inverse_impl` alone are correct. On item 5, the zero-on-singular contract is correct, but on Metal the FTZ case above silently triggers it for non-singular matrices. Item 3 with a power-of-two scale is what covers that.

## The question you haven't asked

Does Newton store the inverse inertia as float32 `mat33`? If it does, df64 inside `_inverse_inertia_wp` buys at most a few ulps that are thrown away at the store, and the 2-3.6x figure has no downstream effect. Decide whether df64 is justified on this path using your P0b per-step divergence probe against a CPU float64 reference over N steps, not with single-inverse residuals.