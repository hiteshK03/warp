"""Probe the two correctness risks I suspect in my own df64 implementation:

1. Lexicographic (hi, lo) comparison on NON-CANONICAL pairs.
   My operator< compares hi then lo. That is only valid if both pairs are
   canonical (|lo| <= ulp(hi)/2). Warp code can construct float64(h, l) directly
   with arbitrary l, and two_sum residuals are not always renormalized.

2. RANGE: does df64 actually fix underflow, or only precision?
   Newton's stated motivation is avoiding float32 determinant underflow for
   small inertias. df64 keeps float32's exponent, so a determinant that
   underflows float32 should STILL underflow. Test a scale where the cube
   goes below float32's normal range.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def cmp_noncanonical(out: wp.array(dtype=wp.int32)):
    """Compare two df64 values that are mathematically equal but built by
    different arithmetic paths, so their (hi, lo) splits may differ.
    Lexicographic hi-first ordering can disagree with the true value."""
    # a = 1/3 computed directly; b = (1/3)*3/3 via a different path
    a = wp.float64(1.0) / wp.float64(3.0)
    b = (wp.float64(1.0) / wp.float64(6.0)) * wp.float64(2.0)
    out[0] = 1 if a < b else 0
    out[1] = 1 if a > b else 0
    out[2] = 1 if (a - b) == wp.float64(0.0) else 0
    # catastrophic-cancellation ordering: x vs x+tiny where tiny < ulp(hi)
    x = wp.float64(1.0e8)
    y = x + wp.float64(1.0e-4)
    out[3] = 1 if y > x else 0            # should be 1 (df64 resolves this)
    out[4] = 1 if (y - x) > wp.float64(0.0) else 0


@wp.kernel
def range_underflow(out: wp.array(dtype=wp.float32)):
    """Determinant of a tiny scaled identity, computed in df64.
    scale=1e-15 -> det = 1e-45, below float32 normal min (~1.18e-38)."""
    s = wp.float64(1.0e-15)
    det = s * s * s
    out[0] = wp.float32(det)
    # and the inverse that Newton actually needs: 1/s^3 = 1e45 (OVERFLOW float32)
    inv = wp.float64(1.0) / det
    out[1] = wp.float32(inv)
    # a scale that stays in range: 1e-7 -> det 1e-21 (representable)
    s2 = wp.float64(1.0e-7)
    out[2] = wp.float32(s2 * s2 * s2)


o = wp.zeros(5, dtype=wp.int32, device=DEV)
wp.launch(cmp_noncanonical, dim=1, inputs=[o], device=DEV)
wp.synchronize_device(DEV)
v = o.numpy()
print("=== comparison consistency (equal values, different arithmetic paths) ===")
print(f"  a<b: {bool(v[0])}  a>b: {bool(v[1])}  (a-b)==0: {bool(v[2])}")
inconsistent = (v[0] or v[1]) and not v[2]
print(f"  VERDICT: {'INCONSISTENT (ordering disagrees with subtraction)' if inconsistent else 'consistent'}")
print(f"  large-magnitude tiny-delta: y>x {bool(v[3])}, (y-x)>0 {bool(v[4])}")
print(f"  VERDICT: {'INCONSISTENT' if bool(v[3]) != bool(v[4]) else 'consistent'}")

r = wp.zeros(3, dtype=wp.float32, device=DEV)
wp.launch(range_underflow, dim=1, inputs=[r], device=DEV)
wp.synchronize_device(DEV)
rv = r.numpy()
print("\n=== range: does df64 fix underflow? ===")
print(f"  det(1e-15 * I) in df64 -> {rv[0]:.6e}   (true 1e-45; float32 min normal ~1.18e-38)")
print(f"  1/det in df64          -> {rv[1]:.6e}   (true 1e45; float32 max ~3.4e38)")
print(f"  det(1e-7 * I) in df64  -> {rv[2]:.6e}   (true 1e-21, in range)")
print(f"  VERDICT: range {'NOT widened (underflow/overflow persists)' if rv[0] == 0.0 or np.isinf(rv[1]) else 'appears widened'}")
