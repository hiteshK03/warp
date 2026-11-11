"""Verify overflow-review claims BEFORE implementing.

M1 (central): inv(1e13*M) NaN comes from inf + (-inf) in the determinant, i.e.
   correct IEEE behaviour, NOT from broken error-free transforms. Test: run the
   SAME case with float32 intermediates (mat33f). If det is NaN there too, the
   overflow guard cannot fix it and my "prerequisite" framing was wrong.
M2: inv(diag(1e20,1e20,1e20)) -> cofactors 1e40 overflow -> inf*0 = NaN, even
   though the true inverse 1e-20 is perfectly representable.
M3: signed zero: f64(-1)*f64(0) gives +0 instead of -0; 1/that gives +inf vs CPU -inf.
M4: 1/inf currently NaN (should be 0).
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def det_f64(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float32(wp.determinant(wp.mat33d(inp[i])))


@wp.kernel
def det_f32(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.determinant(inp[i])


@wp.kernel
def inv_f64(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.mat33(wp.inverse(wp.mat33d(inp[i])))


@wp.kernel
def inv_f32(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.inverse(inp[i])


@wp.kernel
def signed_zero(out: wp.array(dtype=wp.float32)):
    z = wp.float64(-1.0) * wp.float64(0.0)
    out[0] = wp.float32(z)                       # IEEE: -0.0
    out[1] = wp.float32(wp.float64(1.0) / z)     # IEEE: -inf
    out[2] = wp.float32(wp.copysign(wp.float64(1.0), z))  # IEEE: -1
    one = wp.float64(1.0)
    big = wp.float64(3.0e38) * wp.float64(3.0e38)   # +inf
    out[3] = wp.float32(one / big)               # IEEE: 0


def scalar(kernel, mats, dtype=wp.float32):
    a = wp.array(np.array(mats, dtype=np.float32), dtype=wp.mat33, device=DEV)
    o = wp.zeros(len(mats), dtype=dtype, device=DEV)
    wp.launch(kernel, dim=len(mats), inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return o.numpy()


M = np.array([[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]])
print("=== M1: is the 1e13 determinant NaN in FLOAT32 too? ===")
A = (1e13 * M).astype(np.float32)
print(f"  host float64 det = {np.linalg.det(A.astype(np.float64)):.3e}")
d64 = scalar(det_f64, [A])[0]
d32 = scalar(det_f32, [A])[0]
print(f"  df64 det   = {d64!r}")
print(f"  float32 det= {d32!r}")
print(f"  VERDICT: {'float32 ALSO NaN -> overflow guard cannot fix this; pre-scaling required' if np.isnan(d32) else 'float32 finite -> df64-specific'}")
i64 = scalar(inv_f64, [A], wp.mat33)[0]
i32 = scalar(inv_f32, [A], wp.mat33)[0]
print(f"  df64 inv diag    = {np.diag(i64)}")
print(f"  float32 inv diag = {np.diag(i32)}")

print("\n=== M2: diag(1e20) — true inverse 1e-20 is representable ===")
B = np.diag([1e20, 1e20, 1e20]).astype(np.float32)
print(f"  true inverse diag = {np.diag(np.linalg.inv(B.astype(np.float64)))}")
print(f"  df64 inv diag     = {np.diag(scalar(inv_f64, [B], wp.mat33)[0])}")
print(f"  float32 inv diag  = {np.diag(scalar(inv_f32, [B], wp.mat33)[0])}")

print("\n=== M3/M4: signed zero and 1/inf ===")
o = wp.zeros(4, dtype=wp.float32, device=DEV)
wp.launch(signed_zero, dim=1, inputs=[o], device=DEV)
wp.synchronize_device(DEV)
v = o.numpy()
print(f"  (-1)*0      = {v[0]!r}  (IEEE -0.0; sign bit {'lost' if not np.signbit(v[0]) else 'kept'})")
print(f"  1/((-1)*0)  = {v[1]!r}  (IEEE -inf)")
print(f"  copysign(1,-0) = {v[2]!r}  (IEEE -1)")
print(f"  1/inf       = {v[3]!r}  (IEEE 0)")
