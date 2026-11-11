**Verdict:** df64 is defensible for one thing, recovering precision lost to cancellation in internal intermediates. It does nothing for Newton's stated hazard, which is range. As written it also has several concrete bugs, and the ABI gate probably has a hole that produces silent garbage. The kernel you mainly built this for is a one-time setup kernel, where speed doesn't matter, so the trade (non-IEEE semantics in exchange for speed) is backwards for that case.

## 1. Footgun? Opt-in?

Running it automatically is acceptable only if every way it can go wrong is loud. Right now several are silent. The one I worry about most is **df64 quietly falling back to float32 precision without anything failing**. Your kernel outputs are float32, so tolerance-based tests keep passing. Three ways this happens:

- **Fast math.** Metal compiles with fast math by default unless your backend sets `mathMode = .safe`. Under reassociation, `(a - (s - bb)) + (b - bb)` is algebraically 0, and `fma(a,b,-(a*b))` can be folded the same way. Double-float libraries on Metal are known to collapse this way. Your `(1+1e-10)-1` test passing proves this compiler version left the code alone in this inlining context. It doesn't prove it always will. Fast math with `nnan`/`ninf` also lets the compiler fold your `isnan`/`isfinite` to constants. Compile any module containing df64 in safe mode, and add a canary test to CI.
- **Literals.** Codegen emits float literals, so `wp.float64(0.1)`, π, 1/3 and every epsilon are only 24-bit accurate. Any computation involving a constant is capped at float32 accuracy. Constants below about 1e-38 (e.g. `1e-300` sentinels or tolerances) become 0. Fix: on the host, split each literal into `hi = f32(x)`, `lo = f32(x - hi)` and emit `float64(hi, lo)`.
- **Integer conversion.** `float64(int)` gives `wp.float64(16777217) == 16777216`.

Your validation numbers can't detect this kind of collapse. A 1.2e-8 relative error on a float32 output is just float32 output rounding (eps/2 ≈ 6e-8). "Matches CPU exactly" on float32 outputs would also be true with float32 intermediates in a well-conditioned case. Each test needs a float32-intermediate control that visibly fails. Your eig3 `{1e-8,1,3}` test does discriminate; the inverse-inertia test does not.

**Recommendation:** keep it on by default, because Newton can't easily opt in. But add:
- a once-per-module warning naming the kernels that use df64,
- `wp.config.metal_float64 = "df64" | "error"`, with `"error"` in CI,
- the fixes above.

## 2. Range: does df64 fix underflow?

No. Its thresholds are identical to float32's. In fact precision degrades earlier, because the low word needs about 24 bits of exponent headroom below `hi`:

- `lo` is about 2⁻²⁴·|hi|. With flush-to-zero, `lo` flushes once |hi| ≲ 2²⁴·1.18e-38 ≈ **2e-31**. Below that, df64 is plain float32.
- `two_prod` is exact only while its residual is normal, i.e. |a·b| ≳ 2⁻¹⁰² ≈ 2e-31. Below that, the error-free property is gone.
- A determinant scales like I³ and cofactors like I². So:
  - **I ≈ 4e-11** (a 0.1 g, 1 mm bead): det ≈ 6e-32, so the determinant already has float32 precision only.
  - **I ≈ 1e-13** (a 1 mm water droplet, m = 1e-6 kg): det ≈ 1e-39, which flushes to 0. 1/det overflows at the same threshold.
  - Binary64 handles both down to about I ~ 1e-100.

If the 3×3 inverse is determinant-based and returns zero on det == 0 (check whether Warp's does), CPU returns a correct inverse while Metal returns a zero or inf/NaN matrix: an infinitely stiff body or a crash. Your 1e-7 test has det ≈ 1e-21, which is comfortably inside float32 range, so it never exercises the hazard.

**Real fix:** power-of-two pre-scaling. Let k = exponent of max|Aᵢⱼ| and s = 2⁻ᵏ (exact). Then `inv(A) = s·inv(sA)` and `eig(A) = eig(sA)/s`, both exact rescalings. Put this inside the float64 `inverse`/`eig3` specializations; it removes the range problem for those ops at negligible cost. Better still, upstream it to Newton, since the same fix works in float32.

## 3. Bugs in the df64 ops

All of these were worked by hand on the code you pasted. I didn't execute any of them; each is a ready-made test case.

**Comparisons**
- `==` rounds `hi+lo` to float, but `<` is lexicographic, so they disagree. With `a = (1, 2⁻³⁰)` and `b = (1, 0)`: `a == b` is true and `a > b` is true. `a <= b` is false even though `a == b`.
- Fix `==` to `a.hi == b.hi && a.lo == b.lo`. That's correct for canonical pairs, because the canonical representation is unique.
- Lexicographic `<` is fine for canonical pairs. Non-canonical pairs only come from the `(h, l)` constructor, so assert or normalize there.
- NaN: `<=` is written as `!(b < a)`, so `NaN <= 1` and `NaN >= 1` are both true. Write `<=` and `>=` directly.
- `fmin`/`fmax` are therefore asymmetric: `min(NaN, x) = NaN`, `min(x, NaN) = x`.

**`floor`, `ceil`, `rint`, `int()`** all round `hi+lo` to float first.
- Example: `a = (1, -2⁻³⁰) = 0.99999999907`. `floor` returns 1 (should be 0); `int(a)` returns 1.
- Values above 2²⁴: `(2²⁵, -0.25)` floors to 2²⁵ instead of 2²⁵-1.
- `mod` inherits this: for `x = (1, -2⁻³⁰)`, `mod(x, 1)` gives -9.3e-10 instead of 0.99999999907. That's a negative result from positive operands.
- Fix: `fh = floor(hi)`; if `fh != hi`, return `(fh, 0)`; otherwise return `normalize(hi, floor(lo))`. Do the same for the others.

**Inf/overflow**
- `inf + 1`: in `two_sum`, `bb = inf - inf = NaN`, so the result is NaN.
- `inf * 1`: `fma(inf, 1, -inf)` is NaN.
- `1/0`: `q1 = inf`, then `inf * 0` is NaN.
- `sqrt(inf)` is NaN.
- Any overflow therefore becomes NaN instead of inf, which breaks inf sentinels and guarded divisions. Fix: if `!isfinite(s)`, return `(s, 0)` in add, mul, div and sqrt.

**Addition is the "sloppy" Dekker add.** Its error is bounded relative to |a|+|b|, not |a+b| (Joldes–Muller–Popescu 2017). That is weakest in exactly the cancellation case you built this for.
- Example: `a = (1, 2⁻²⁴−2⁻⁴⁸)`, `b = (−1, −2⁻⁵⁰)`. `a.lo + b.lo` rounds, giving relative error 2⁻²⁶ ≈ 1.5e-8 in the result. Binary64 gives this sum exactly.
- Use the accurate two-`two_sum` add (about 20 flops).

**Division and sqrt:** one correction step is fine. The residual is computed through the exact product, so it absorbs Metal's inexact fast `/` and `sqrt`. Error is a few df64 ulp. No bug apart from the inf/NaN issue above.

**Adjoints:** the ones you wrote are correct. The generic `adj_div`/`div` under `FP_CHECK` (debug mode) may pass a float64 struct to printf-style varargs. That's probably a loud failure, but build once with `mode="debug"` to confirm.

**Minor:**
- `copysign(x, -0.0)` returns a positive result.
- `isinf((inf, NaN))` is true, but `isnan` is also true for the same value.

## 4. The regex gate

The gate is mostly bypassable, and one bypass produces silent garbage.

- **Scalar float64 kernel arguments aren't checked.** The host packs binary64 into 8 bytes, and the struct `{float hi, lo}` is also 8 bytes, so nothing fails. Example: `k(s: wp.float64)` launched with 1.0 reads `hi = 0.0f`, `lo = 1.875f` (the high word `0x3FF00000`). `float(s)` returns 1.875. Structs with float64 fields and by-value `vec3d` arguments behave the same way. This applies unless your argument packer already converts them.
- **The array pattern only matches `array_t<float64>`.** It misses arrays of `vec_t<3,float64>`, `mat_t<3,3,float64>`, `quat_t`/`transform_t`, and `fabricarray`/`indexedarray` variants.
- **The atomic pattern almost never fires.** Codegen emits `wp::atomic_add(var_a, var_i, var_v)`, where the type isn't in the text. Atomics on `vec3d` arrays get through both checks, including adjoint accumulation in the backward pass.
- Comment stripping also eats string literals that contain `//`.

**What I'd do instead:**
1. **Python, by type.** Walk `kernel.adj.args` recursively (array dtype → `type_scalar_type`, struct fields) and reject float64, or better, marshal scalars and structs as `(f32(x), f32(x - hi))`. Catch atomics in `Adjoint.add_call`, where the overload's argument types are resolved.
2. **C++ backstop.** Add a `static_assert(!contains_float64<T>)` trait inside `array_t<T>` and the atomic templates, plus `= delete` atomic overloads for float64.

## 5. Better alternative

Yes, and it fits the motivating case better:

- **Softfloat IEEE binary64 on 64-bit integers.**
  - It is bit-exact against the CPU path, with full range and correct inf/NaN handling.
  - It is layout-compatible, so float64 arrays and arguments work, and most of the ABI gate disappears. Atomics stay blocked.
  - Cost is maybe 30–100× a float op (estimate; measure it). That is irrelevant for one-time `finalize()`.
- **Or run the fp64 setup kernels on the CPU device.** Real binary64, and unified memory makes the copy nearly free. This doesn't help Steel's per-step fp64.

**Suggested tiering:**
- `float64` on Metal means softfloat, which is correct by default.
- df64 becomes an explicit per-module opt-in for hot paths like Steel, after a discriminating validation.
- Range-scaling fixes go upstream to Newton regardless.

If you keep df64 as the default, the minimum before merging is:
- safe math mode for df64 modules,
- split literals,
- the `==`/`<=`/`floor`/inf fixes,
- the accurate add,
- type-based gating including scalar arguments,
- range and fast-math canary tests with float32 controls.