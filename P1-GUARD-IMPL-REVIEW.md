# Review: df_guard post-check

The post-check works for its stated job. It handles inf and NaN *inputs*, which is what the test covers. It does not handle an *intermediate* that overflows while the inputs are finite. In that case the rule "keep an infinite hi" returns a wrong infinity, and in division the sign can flip. Every test input has `lo == 0`, so whole categories are never exercised. All bugs below are derived by hand with IEEE round-to-nearest assumed; each one needs a run to confirm. Findings are ordered by severity.

## 1. Division of finite numbers returns −inf (high)

**Input:** `a = FLT_MAX`, `b = 1.99999976f` (`0x3FFFFFFE`, i.e. 2 − 2⁻²²). Both are loaded from a buffer, so `lo = 0`.

**Expected:** `+1.7014120e38` (`0x7F000001`).
**Got (derived):** `-inf`.

Trace:
- `q1 = a.hi/b.hi` rounds up to 2¹²⁷ + 2¹⁰⁴.
- `q1·b.hi` is exactly (2²⁴ − 2⁻²²)·2¹⁰⁴. That is above the overflow midpoint, so in `two_prod` `p = inf` and `e = -inf`.
- The inner `operator*` guard returns `(inf, 0)`.
- `r = a − inf` gives `(-inf, 0)`, so `q2 = -inf`.
- `normalize(q1, -inf)` gives `(-inf, NaN)`.
- `df_guard` sees an infinite hi, keeps it, and returns `-inf`.

With `a = -FLT_MAX` you get `+inf`. Before this diff the same input returned NaN, so the guard turned a NaN into a confident wrong answer. The trigger is any `|a|` within about 1 ulp of FLT_MAX where `q1` rounds up. To find more triggers, sweep `b` over [1.5, 2).

The comment's reasoning ("an infinite hi means finite inputs overflowed in the tail") is false here. The infinity comes from a residual that was contaminated before the last step.

**Fix:**
- In `df_guard`, keep an infinite `r.hi` only if `naive` is also infinite, or if `|naive| == FLT_MAX` with the same sign. Otherwise return `naive`.
- In `operator/`, when `|a.hi| ≥ 2¹²⁶`, compute `2·((0.5·a)/b)` so the residual cannot overflow.

## 2. Spurious overflow in the leading EFT when the true value is finite (medium)

The guard assumes overflow only happens in the tail normalize. But `two_sum` / `two_prod` on the hi words can overflow first, even when the low words bring the true value back into range.

**Multiply case, a natural one:**
`x = FLT_MAX` (loaded), `y = sqrt(x)`, output `float32(y*y)`.
- Your 1-ulp `sqrt(FLT_MAX)` result means `y = (2⁶⁴, -2³⁹)`.
- `two_prod(2⁶⁴, 2⁶⁴)` gives `p = inf`, and the guard falls back to `naive = inf`.
- **Expected:** `3.4028235e38`. **Got:** `inf`.

**Add case:**
`x0 = FLT_MAX`, `x1 = -5.0706024e30` (−2¹⁰²), `x2 = 1.0141205e31` (2¹⁰³), output `float32((x0 + x1) + x2)`.
- `x0 + x1` becomes `(FLT_MAX, -2¹⁰²)`.
- Adding 2¹⁰³ makes the hi sum hit an exact tie, which rounds to even, i.e. `inf`.
- The true value is FLT_MAX + 2¹⁰², which is below the midpoint.
- **Expected:** `FLT_MAX`. **Got:** `inf`.

Fixing this needs power-of-two pre-scaling. Otherwise, document it as "values within about 1 ulp of FLT_MAX may spuriously overflow" and correct the guard comment.

## 3. Your sqrt(FLT_MAX) residual is a Newton tie, not noise

- `s = 2⁶⁴ − 2⁴⁰`.
- `r.hi = 2¹⁰⁴ − 2⁸⁰`, so the correction is exactly 2³⁹.
- `hi + lo` lands exactly on a float32 midpoint and ties to even, giving `2⁶⁴`. The true value is slightly below the midpoint because the −r²/8s³ term is missing.

The same thing happens for any input where the single correction lands exactly on a midpoint. Item 2 shows it then feeds into an overflow. It is not cosmetic.

## 4. The fast-math premise is either wrong or not applied consistently (conditional, verify)

**If fast-math is on:**
- `a.hi + b.hi` carries `ninf`/`nnan` flags, so an infinite result is LLVM poison. The bit test on poison proves nothing.
- Recent InstCombine also rewrites `(bitcast x & 0x7fffffff) < 0x7f800000` into `llvm.is.fpclass`, which those flags can fold. So the comment's "cannot be folded" claim is false.
- Reassociation would also break `df_two_sum` itself.

**If fast-math is off:** the integer trick is unnecessary.

**Either way,** the header is inconsistent with its own premise:
- `isnan`, `isinf` and `isfinite` still call `metal::` predicates.
- `_wp_native_fmin`/`fmax` use `b == b`.

Test: `wp.isnan(wp.float64(a[i]))` with `a = NaN` must return true. Also pin `fastMathEnabled` / `mathMode` explicitly and assert it in a test.

Separately, `_wp_native_fmin` never propagates NaN, despite its comment:
- `min(1, NaN)`: `1 <= NaN` is false, `NaN == NaN` is false, so it returns `1`.
- `min(NaN, 1)` returns `1`.

Compare against the float32 `wp.min` on the same device.

## 5. Signed zero is lost, and the test cannot see it (low, but contradicts "IEEE restored")

- `wp.float32(wp.float64(-0.0))`: `hi + lo` is `-0 + +0 = +0`. **Expected:** `0x80000000`. **Got:** `0x00000000`.
- `x = -1`, `y = 0`, `1.0/(x*y)` in-kernel: `two_prod` gives `(-0, +0)`, normalize gives `+0`, so the result is `+inf`. **Expected:** `-inf`.
- `sqrt(-0)` returns `+0`; IEEE says `-0`.

Your checker uses `g == r`, which treats −0 and +0 as equal. Compare bit patterns instead.

## 6. sqrt with flush-to-zero (conditional)

The early return tests `a.hi == 0` instead of `s == 0`. If `metal::sqrt` flushes a subnormal input but the comparison does not:
- `s = 0`, the correction is `r.hi/0 = inf`, `normalize` gives `(inf, NaN)`, and the guard keeps `inf`.
- **Input:** `a = 1e-40f`. **Expected:** ≈1e-20 (or 0 under FTZ). **Got:** `+inf`.

Fix: test `df_abits(s) == 0` after the sqrt.

## 7. Test-harness gaps

- Every input has `lo == 0`, so the `e += a.lo ± b.lo`, cross-term and tail-overflow paths are never run. Build operands with nonzero lo at runtime (e.g. `x0 + x1` as in item 2).
- No compound expressions, which is where items 1–3 show up: `(x/y)*y`, `sqrt(x)*sqrt(x)`, three-term sums.
- No hi/lo readback of guarded outputs. The invariant to check: for non-finite results `lo == 0`, otherwise both words are finite.
- Signed zero is invisible to the checker (item 5).
- The regression suites almost certainly have no values within an ulp of FLT_MAX. "No regressions" says nothing about this region.

## Outside the diff (cheap to confirm)

- **`floor` / `int()` collapse to `hi + lo` first:**
  - `floor(float64(2^30) - float64(0.25))` gives `1073741824`; expected `1073741823`.
  - `int()` on the same value has the same bug.
- **`mod`** is `a - b*floor(a/b)`, i.e. floored modulo. The float `mod` in `builtin.h` is `fmod`.
  - `mod(-1, 3)` gives `2`; fmod gives `-1`.
  - `mod(1, inf)` gives NaN; expected `1`.
- **Host packing:** if a host float64 → (hi, lo) packer exists and computes `lo = x − float32(x)`, then `x = 1e39` packs to `(inf, -inf)`, and `wp.float32(x)` returns NaN instead of `inf`. Check that the packer zeroes `lo` for non-finite values.