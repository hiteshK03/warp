# Review: power-of-two pre-scaling plan for 3x3 `inverse` / `eig3`

**Verdict:** The plan works for `inverse` with a uniform scale on SPD inertia. As written, three parts are wrong:

- **`eig3` scaling does nothing.** Drop it.
- **The generic Warp scope is wrong.** Max-abs scaling makes some matrices that work today return a zero matrix.
- **The motivation understates the problem.** df64 loses precision well before the determinant underflows.

## 1. Algebra

Correct, for s > 0. inv(sA) = s⁻¹·inv(A), so inv(A) = s·inv(sA). For eigenvalues, (sA)v = sλv, so the eigenvectors are the same and λ = eig(sA)/s.

There is a stronger property you should state and test. Every float op here (+, −, ×, ÷, sqrt with exponent adjust, `fma`) gives bit-identical results under power-of-two scaling, as long as no intermediate goes subnormal or overflows. That means the scaled `inverse` is bit-identical to the old one wherever the old one stayed in normal range. That is your regression guarantee.

Test: random well-scaled A on CPU, Metal float32 and Metal df64. Assert scaled == unscaled bitwise.

## `eig3` scaling: drop it

Upstream `eig3` is quaternion Jacobi. Its stopping tests are all dimensionless:
- `thet` is a ratio of diagonal difference to off-diagonal;
- the `1e6` cutoff applies to `thet`;
- it stops when `c == 1`;
- it stops when `offdiag == 0`.

`D = Qᵀ A Q` is linear in A. So `eig3` already behaves the same at any scale. Inputs around 1e-15 are normal float32 and nothing squares them. Pre-scaling changes nothing apart from subnormal-floor cases. Newton already normalizes before calling it anyway.

Switching Newton's non-power-of-two scale to a power of two removes about 1 ulp of df64 noise (~1e-14 relative). That only matters if you compare results bitwise.

## 2. Where it should live

**Generic Warp `inverse_impl`: no.**
- Every float32 user pays about 9 abs, a max, an exponent extract and 18 ldexp on a hot builtin.
- Matrices whose determinant used to underflow now return a large inverse instead of zero. That changes semantics silently on CUDA and CPU.
- `wp.determinant(A)` still underflows. User code like `if determinant(A) != 0: inverse(A)` stops matching what `inverse` does.
- 2x2, 4x4 and NxN would not match 3x3.
- Max-abs regresses mixed-magnitude matrices (see §4). Upstream would reject it.

**Metal df64 specialization only: acceptable, second choice.**
- It brings Metal closer to CUDA fp64, whose range really is ~1e±308. That improves parity.
- Downside: one more Metal-only code path, and Metal float32 keeps the same hazard as CUDA float32. That is fine, since it matches.

**Newton inertia helpers: recommended.**
- Newton knows the matrix is SPD and physical, so a simpler, safe scale exists.
- No Warp API change.
- On CUDA fp64 it is harmless.
- It can go upstream on its own merits.

## 3. Determinant arithmetic for A = 1e-15·I

- **Before scaling:** det = 1e-45. That is below FLT_MIN (1.18e-38), and Metal flushes subnormals to zero, so det = 0 and `inverse` returns the zero matrix.
  - Cofactors are 1e-30 (normal). Only det underflows.
  - In float32 / df64-hi, det underflows once |A| < cbrt(1.18e-38) ≈ 2.3e-13.
- **After scaling:** frexp(1e-15) = 0.563·2⁻⁴⁹. Scaling by 2⁴⁹ gives sA = 0.563·I.
  - det(sA) = 0.178, in range.
  - inv(sA) = 1.776·I.
  - Unscaling by 2⁴⁹ gives 1e15·I. Correct and in range.
- **Can the unscale overflow?** Only when the true inverse cannot be represented: 1/σ_min(A) > 3.4e38.
  - Example: A = 1e38·I. The true inverse 1e-38 is subnormal, gets flushed, and you get a zero matrix. That limit is inherent.
  - **Bug: df64 turns overflow into NaN, not inf.** So an inherent overflow comes back as NaN, not inf.

### Missed issue: df64 precision fails before the determinant underflows

df64's low word becomes subnormal, and is flushed, whenever |hi| < 2⁻¹⁰² ≈ 2e-31. A 3x3 determinant drops below that once the entries are under about 6e-11. That covers exactly the 1e-11..1e-13 band Newton cares about.

In that band, the determinant is computed at **float32 precision**, even though it does not underflow.

The "1e-12 is exact" measurement used the identity. Its low words are zero everywhere, so it cannot show this.

Predicted test:
- **Input:** A = 1e-12·[[2,1,0],[1,3,1],[0,1,4]].
- **Expected (true df64):** ‖inv(A)·A − I‖ ≈ 1e-14.
- **Predicted actual:** about 1e-7 to 1e-8.

Scaling fixes this as well. The scaling is needed for precision, not only to prevent underflow.

## Implementation bug in "s = 2⁻ᵏ"

Building s as a float breaks at both ends of the range:

- **Large inputs:** max|A| ≥ 2¹²⁶ (8.5e37) gives s = 2⁻¹²⁷, which is subnormal. FTZ flushes s to 0, so sA = 0 and the result is a zero matrix.
  - Input: A = 1e38·I. Expected inverse: 1e-38·I, which is itself flushed, so zero is unavoidable here.
  - Input: A = 2e37·I. Expected 5e-38·I. With s = 2⁻¹²⁵ the scale step works, but check the boundary with an exact test.
- **Subnormal input bits:** If NumPy uploads subnormal values, the exponent bits read biased exponent 0, so k = −127. The arithmetic flushes those values to 0 anyway, so exponent and value disagree.

Fix:
- Apply `ldexp(hi, -e)` and `ldexp(lo, -e)` to each word directly. Never build s.
- Clamp e, or split it into two steps of at most 2⁶³ each.
- Check on the device that `metal::ldexp(1e38f, -127)` returns 0.588, not 0. Under fast-math it may be implemented as a multiply by `exp2`.
- Per-word ldexp also avoids a df64 multiply and keeps overflow as (inf, x) rather than NaN.

## 4. What breaks

**Mixed-magnitude matrix with max-abs scaling: regression (concrete).**
- Input: diag(1e-20, 1, 1e20).
- Today, unscaled: cofactors are 1e20, 1 and 1e-20, det = 1. Result diag(1e20, 1, 1e-20), **correct**.
- With max-abs scaling (2⁻⁶⁷): s·1e-20 = 6.8e-41, which is subnormal and flushed on Metal. det(sA) = 0, so the result is a **zero matrix**.
- On CPU without FTZ the result is non-zero but carries about 1e-5 relative error.
- This kills the generic-Warp placement.

**Physical inertia is safe.** For physical inertia, I_max ≤ I_mid + I_min, so I_mid ≥ I_max/2. With max-abs scaling, det(sA) ≥ λ_min/(2·λ_max²). That only underflows when the condition number is above about 1e37, which does not happen.

**Zero matrix.** max = 0, and frexp(0) gives e = 0 (or garbage from bit extraction). Guard explicitly: if max == 0, skip scaling. det = 0 then returns the zero matrix, as before.

**Singular matrices.** Exact det == 0 still returns zero, by the bit-equivariance in §1. Only matrices whose determinant underflowed change, and that change is the intended fix.

Keep the zero-on-singular contract. Newton point masses and kinematic bodies with zero inertia probably depend on zero inverse inertia.

Side effect to flag: tiny parts that used to get a zero inverse inertia will now get about 1e13. That can destabilize existing Newton scenes at the current dt. Release-note it.

**Inf / NaN inputs.**
- Guard with `!isfinite(max)` and skip scaling.
- Input diag(inf, 1, 1): bit extraction gives e = 128, the 1s scale to 0, and inf·0 is NaN.
- Two existing df64 bugs affect this:
  - **Overflow returns NaN.** `float64(INFINITY) * float64(2)` returns NaN (expected inf), because `fma(inf, 2, -inf)` is NaN. `float64(3e38f) + float64(3e38f)` returns NaN (expected inf), because inf − inf appears in two-sum. `float64(1) / float64(0)` returns NaN (expected inf). That last one is your measured "1/det → NaN".
  - **`_wp_native_fmax` drops NaN.** `max(float64(NaN), float64(1))` returns 1, which contradicts its "NaN-propagating" comment. If the scale is computed with it, NaN entries are skipped. NaN still reaches the output through det, so the outcome is fine here, but the comment is wrong.

## 5. Choice of scale

For SPD matrices, max|A_ij| = max diagonal entry, so reading 3 diagonal entries instead of 9 gives the same scale.

For `inverse` on SPD, prefer **symmetric diagonal power-of-two equilibration**:
- D_ii = 2^(−round(e_ii/2)), where e_ii is the exponent of A_ii.
- Compute inv(A) = D·inv(DAD)·D. It is exact, and handles diag(1e-20, 1, 1e20) because DAD ≈ I.
- van der Sluis: this is near-optimal among diagonal scalings for SPD matrices.

This is **not** valid for `eig3`. DAD is a congruence, not a similarity, so it changes the eigenpairs. `eig3` may only use uniform scaling, and it does not need any (see above).

## 6. Cheaper option

A two-line Newton change, mirroring its existing `eig3` code:

```python
s = max(inertia[0,0], inertia[1,1], inertia[2,2])  # SPD: equals max-abs
return wp.mat33(wp.inverse(wp.mat33d(inertia) / wp.float64(s)) / wp.float64(s))
```

- It needs a guard for s == 0 (keep returning zero).
- A non-power-of-two s adds about 2 roundings at ~1e-14 relative in df64, which does not matter.
- It needs no Warp change, no Metal-specific code, and no frexp.

Alternative: Newton already computes `eig3`, so it can build inv = Q·diag(1/λ)·Qᵀ with no determinant at all. That also gives a natural place to clamp tiny λ. It loses about 1 ulp of orthogonality, which is fine.

## Recommendation

1. Drop `eig3` from the plan.
2. Do not touch the generic `inverse_impl`.
3. Fix `_inverse_inertia_wp` in Newton with diagonal-max scaling, guarded for zero and non-finite values.
4. Separately, fix df64 inf/overflow propagating as NaN and the `fmax` comment.
5. Verify by running:
   - 1e-15·I
   - 1e-12·M, checking precision
   - diag(1e-20, 1, 1e20)
   - 1e38·I
   - zeros
   - diag(inf, 1, 1)
   - the bitwise-equivariance property test