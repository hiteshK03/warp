# Plan: fix df64 overflow/inf -> NaN (before any scale-related claims)

## Confirmed problem (measured on metal:0)

All of these return NaN where IEEE gives inf:
- `3e38 * 3e38` (plain overflow)
- `inf + 1`
- `inf * 1`
- `sqrt(inf)`
- `1 / 0`

Downstream consequence (measured): `inv(1e13 * [[2,1,0],[1,2,1],[0,1,2]])` has
det = 1.8e40 > FLT_MAX. det becomes NaN, and `inverse_impl`'s `det != Type(0)`
guard passes for NaN, so the result is a **NaN matrix** rather than the zero
matrix the singular path would give. Silent corruption of body inverse inertia.

## Root cause

Error-free transforms are invalid on non-finite inputs:
- `df_two_sum(inf, b)`: `bb = s - a = inf - inf = NaN`, so the residual is NaN.
- `df_two_prod(inf, b)`: `fma(inf, b, -inf) = NaN`.
- division: `q1 = a.hi / b.hi = inf`, then the residual computes `inf * 0 = NaN`.
- sqrt: same, via `df_two_prod(s, s)` when s is inf.

## Proposed fix

Guard each op: if the computed high word is not finite, return `{hi, 0}` immediately
(no low-word arithmetic). Specifically in `operator+`, `operator-`, `operator*`,
`operator/`, and `sqrt`.

Reviewer's caution: under fast-math `isfinite`/`isinf` may be folded to `true`.
Must verify the guard actually survives compilation, not just that it looks right.

## Questions for review

1. Is "if !isfinite(hi) return {hi, 0}" the right guard placement, or must the
   INPUTS be checked too (e.g. a finite hi produced from inf inputs via NaN
   cancellation)? Give a concrete case if the output-only check is insufficient.
2. `metal::isfinite` under fast-math: is it reliable? If not, what is a
   fast-math-proof finite test (e.g. `abs(hi) < FLT_MAX` comparison, or bit test
   on the exponent)? Note the canary already showed reassociation is NOT eating
   my low words in this context, so fast-math may be less aggressive than feared.
3. Sign of zero and signed inf: does `{hi, 0}` preserve -inf and -0 correctly?
4. Should `1/0` give inf (IEEE) or should df64 raise/flag? Newton's inverse path
   relies on `det != 0` to decide singularity; with inf instead of NaN, what does
   `inverse_impl` now return for an overflowed determinant, and is that better or
   worse than the NaN matrix?
5. Does fixing this change any currently-passing result? The P1 acceptance
   (Newton finalize + Steel, inverses matching CPU) was all in-range, so I expect
   no change — confirm or identify a case.
6. Is there a cheaper/safer alternative to per-op guards (e.g. normalizing inputs
   once at construction)?
