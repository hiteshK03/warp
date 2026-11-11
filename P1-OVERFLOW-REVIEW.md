# df64 overflow plan review: the guard design has holes, and the motivating inverse case stays NaN

**Verdict:** The fix is worth doing for scalar IEEE parity. But the guard as written leaves holes, and the main reason given for it is wrong. Your motivating case, `inv(1e13*A)`, still returns a NaN matrix after the fix. The cause is in the determinant formula, not in the error-free transforms.

## Main problem: the motivating case is not fixed

Warp's 3x3 determinant is `dot(row0, cross(row1, row2))`, or the equivalent cofactor expansion. For `s=1e13`, `A=[[2,1,0],[1,2,1],[0,1,2]]`:
- `cross(r1, r2) = (3, -2, 1)e26`
- `dot` products are `6e39` (+inf) and `-2e39` (-inf)
- `inf + (-inf) = NaN`

So det is NaN with perfect IEEE inf handling. Plain float32 does the same thing.

**Check:** run the same kernel with `mat33f` on Metal. If det is NaN there too, the plan's stated reason fails.

Only pre-scaling fixes this case. The two changes are independent, so stop calling the inf fix a prerequisite for pre-scaling.

There is a second NaN path even when det is a clean single-sign inf:
- **`inv(diag(1e13,1e13,1e13))`:** det `1e39` gives +inf. Then `1/inf` gives NaN under your plan (see Q1). With `1/inf` fixed, `inv_det = 0` and cofactors `1e26*0 = 0`, so you get a **zero matrix**. CPU gives `diag(1e-13)`.
- **`inv(diag(1e20,1e20,1e20))`:** cofactors `1e40` are inf, so `inf * 0 = NaN`. You get a **NaN matrix with every fix applied**, while the true inverse `1e-20` fits easily in range.

## Q1: guard placement — neither position works alone

**Output-of-first-EFT guard alone** (checking `s`/`p`/`q1`) misses overflow in the final normalize:
- Add: `x = f64(FLT_MAX) + f64(9e30)` is `{FLT_MAX, 9e30}` (9e30 < half-ulp 1.01e31). Then `x + f64(9e30)`: `s = FLT_MAX` is finite and passes the guard, `e = 1.8e31`, normalize rounds to inf. Result `{inf, NaN}`, expected `{inf, 0}`.
- Mul: `f64(FLT_MAX) * (f64(1.0) + f64(5e-8))`. Here `b = {1, 5e-8}` and `p = FLT_MAX` is finite. `e ≈ 1.7e31` > half-ulp, so normalize goes to inf. Result `{inf, NaN}`.
- Div: `f64(1) / f64(inf)` gives `q1 = 0` (finite, passes). Residual `0*inf` is NaN. Result NaN, expected 0. `approx_rcp(inf)` and `adj_sqrt` with `ret=inf` inherit this.

**Final-hi guard alone** returns NaN instead of inf:
- `inf + 1`: normalize(inf, NaN) makes the final hi NaN, so you return `{NaN, 0}`. Expected inf.
- `inf * 1`: same.

Input checks are not needed separately if you use a post-check with a fallback (Q6).

## Q2: fast-math

`metal::isfinite`, `isnan`, and your `b == b` in `_wp_native_fmin`/`fmax` all depend on finite-math assumptions. Clang folds these under no-NaN/no-inf assumptions, and Metal's fast-math does not promise NaN/inf behaviour.

Your canary does not cover this. It tested reassociation, which is a separate flag from the no-NaN/no-inf assumptions.

- **`abs(hi) < FLT_MAX`:** foldable under the same assumptions, and wrong anyway (it rejects FLT_MAX itself; it needs `<=`).
- **Use an integer bit test:** `(as_type<uint>(x) & 0x7fffffff) < 0x7f800000`. Integer ops aren't subject to FP folding.
- **Verify** with values loaded from a buffer at runtime, not constants, which fold correctly regardless.
- **Better:** check whether the backend compiles with `fastMathEnabled`/`MathMode.fast`. If it does, compile float64 modules in safe mode. Two-sum correctness under fast-math is also only canary-deep: one inlining context proves nothing about others.

## Q3: signs

- `{-inf, 0}` is fine. Negating `{inf, 0}` gives `{-inf, -0}`, also fine.
- **Existing -0 bug, unrelated to the guard:** `f64(-1) * f64(0)` gives `{+0, +0}`. normalize does `-0 + +0 = +0`; IEEE says -0. Concrete: `1.0 / (f64(-1)*f64(0))` gives +inf on Metal and -inf on CPU. `copysign(1, -0)` gives +1 (because `y < 0` is false for -0) vs -1 on CPU.
- `abs(-0)` returns -0.
- `sqrt(-0)` returns +0.

These are minor, but document them.

## Q4: `1/0`

Return inf, not a flag. There's no flag mechanism on the GPU, and CUDA fp64 also returns inf. For `inverse_impl` after all fixes:
- **Single-sign overflow:** zero matrix. Better than NaN (it doesn't propagate, and an "immovable" body is close in absolute terms), but 100% relative error, it fails CPU parity, and it looks exactly like the singular path.
- **Mixed-sign overflow, or cofactor overflow:** still NaN (see above).

Make `inverse` treat a non-finite det as its own case, or pre-scale.

## Q5: does it change passing results?

For finite outputs, no, provided the guard only fires when the result is non-finite. Two things to verify:
- The P1 suite should be **bitwise** identical before and after, not just allclose, because an added branch can shift codegen.
- Comparisons on a former NaN now behave differently: `x > limit` was false for NaN and is true for inf, and `fmax(inf, 5)` is now inf instead of 5. This is more correct, but it is a behaviour change. Any special-values tests for float64 on Metal may flip from fail to pass.

## Q6: alternative

Normalizing at construction doesn't help: finite canonical inputs still overflow (`3e38*3e38`). Instead, use one post-check per op with a hi-word fallback:

```cpp
inline uint df_abits(float x) { return as_type<uint>(x) & 0x7fffffffu; }
inline bool df_fin(float x)   { return df_abits(x) <  0x7f800000u; }
inline bool df_inf(float x)   { return df_abits(x) == 0x7f800000u; }

// r: full df64 result; naive: same op on hi words only (IEEE inf/NaN semantics)
inline float64 df_guard(float64 r, float naive)
{
    if (df_fin(r.hi) && df_fin(r.lo)) return r;
    // r.hi == ±inf only arises when finite inputs overflowed in the tail normalize
    return float64(df_inf(r.hi) ? r.hi : naive, 0.0f);
}
// +: df_guard(df_normalize(s, e), a.hi + b.hi)
// *: df_guard(df_normalize(p, e), a.hi * b.hi)
// /: df_guard(df_normalize(q1, q2), a.hi / b.hi)
// sqrt: df_guard(..., metal::sqrt(a.hi))
```

This handles every case above: `inf+1`, `3e38²`, `1/0`, `1/inf` (gives 0), `sqrt(inf)`, `inf-inf` (gives NaN), and both tail overflows (give inf with the correct sign). The cost is one uniform, rarely-taken branch per op.