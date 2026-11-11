# Plan under review: power-of-two pre-scaling for 3x3 inverse / eig3

## Motivation (measured, not assumed)

df64 on Metal keeps float32's exponent range. Measured on metal:0:
- det of a 1e-15-scale identity underflows to 0; 1/det returns NaN
- 1e-12 scale and above are exact
- Warp's `inverse_impl<Type,false>` (mat.h) returns a ZERO MATRIX when det == 0,
  so an underflowed determinant silently produces an infinitely stiff body.

Newton's inertia finalization calls `wp.inverse(wp.mat33d(inertia))` and
`wp.eig3(wp.mat33d(inertia)/scale)`. Real robot inertias are ~1e-3 (safe), but small
lightweight parts (gram-scale, mm-size) reach 1e-11..1e-13 where this bites.

Note Newton ALREADY does a max-abs normalization before eig3:
```python
scale = max(|inertia[i,j]|)
vectors, values = wp.eig3(wp.mat33d(inertia) / wp.float64(scale))
return wp.mat33(vectors), wp.vec3(values * wp.float64(scale))
```
but that scale is NOT a power of two, and `_inverse_inertia_wp` does NO scaling:
```python
return wp.mat33(wp.inverse(wp.mat33d(inertia)))
```

## Proposed change

Add power-of-two pre/post-scaling inside the 3x3 `inverse` and `eig3` paths:

1. k = exponent of max|A_ij| (via frexp-style bit extraction on the float32 hi word)
2. s = 2^-k, exactly representable
3. compute inv(s*A), then multiply the result by s   [since inv(sA) = inv(A)/s]
4. for eig3: eigenvectors unchanged, eigenvalues divided by s

Claimed properties:
- exact rescaling (power of two => no rounding in the scale/unscale)
- removes the underflow/overflow hazard for these ops
- works for float32 too, so it is upstreamable to Newton independent of Metal

## Questions

1. Is the algebra right? inv(sA) = (1/s) inv(A), so inv(A) = s * inv(sA). And
   eig(sA) has eigenvalues s*lambda with the SAME eigenvectors, so lambda = eig(sA)/s.
   Confirm or correct.
2. Where should this live: inside Warp's generic `inverse_impl`/`eig3` (affects all
   backends and all users), only in the Metal float64 specialization, or in Newton's
   inertia helpers? What are the risks of each?
3. Does pre-scaling actually rescue the determinant, given det(sA) = s^3 det(A)?
   Work the exponent arithmetic for A ~ 1e-15 * I and say whether the intermediate
   determinant is now in range, and whether the final unscale can still overflow.
4. What breaks: singular matrices, zero matrices, mixed-magnitude entries (e.g.
   diag(1e-20, 1, 1e20)), inf/NaN inputs, and the existing det==0 -> zero-matrix
   contract? Should the zero-matrix-on-singular behavior be preserved?
5. Is max-abs the right scale choice, or is there a better conditioned choice for
   a symmetric positive-definite inertia tensor?
6. Anything cheaper/simpler that solves the same problem for this specific case?
