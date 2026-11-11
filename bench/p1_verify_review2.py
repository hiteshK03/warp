"""Verify review #2. V2 first: it invalidates my table if true.

V2: near-singular resid == 0 for BOTH is impossible with cond 9.4e6 and float32
    outputs. Print the actual inverse and residual components.
V1: read back the df64 determinant's LOW word at 1e-12 vs 1e12 scale.
    Predicted: low word flushed to 0 at 1e-12 (below the 2e-31 cutoff).
V4: fast-math canary — (1+1e-10)-1 must be 1e-10, not 0.
R3: NaN ordering in wp.max must match across backends (cpu vs metal here; no CUDA).
CLIFF: inv(1e13 * M) has det ~4e39 (overflow) -> predicted NaN matrix, not zeros.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"


@wp.kernel
def inv_f64(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.mat33(wp.inverse(wp.mat33d(inp[i])))


@wp.kernel
def det_lowword(inp: wp.array(dtype=wp.mat33), hi_out: wp.array(dtype=wp.float32),
                lo_out: wp.array(dtype=wp.float32)):
    """Recover the df64 low word: det - float32(det), evaluated in df64."""
    i = wp.tid()
    d = wp.determinant(wp.mat33d(inp[i]))
    hi = wp.float32(d)
    hi_out[i] = hi
    lo_out[i] = wp.float32(d - wp.float64(hi))


@wp.kernel
def canary(out: wp.array(dtype=wp.float32)):
    out[0] = wp.float32((wp.float64(1.0) + wp.float64(1.0e-10)) - wp.float64(1.0))


@wp.kernel
def nan_order(out: wp.array(dtype=wp.float32)):
    nan = wp.float32(0.0) / wp.float32(0.0)
    out[0] = wp.max(nan, wp.float32(1.0))
    out[1] = wp.max(wp.float32(1.0), nan)
    out[2] = wp.min(nan, wp.float32(1.0))
    out[3] = wp.min(wp.float32(1.0), nan)


def run(kernel, mats, dev=DEV):
    a = wp.array(np.array(mats, dtype=np.float32), dtype=wp.mat33, device=dev)
    o = wp.zeros(len(mats), dtype=wp.mat33, device=dev)
    wp.launch(kernel, dim=len(mats), inputs=[a, o], device=dev)
    wp.synchronize_device(dev)
    return o.numpy()


print("=== V2: explain near-singular resid == 0 ===")
A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0 + 1e-6, 1.0], [1.0, 1.0, 1.0 + 1e-6]], dtype=np.float32)
print(f"  A as stored (float32):\n{A}")
print(f"  A[1,1]-1 = {A[1,1]-1.0:.3e}   A[2,2]-1 = {A[2,2]-1.0:.3e}")
print(f"  host float64 det = {np.linalg.det(A.astype(np.float64)):.6e}")
inv = run(inv_f64, [A])[0]
print(f"  df64 inverse:\n{inv}")
prod = inv.astype(np.float64) @ A.astype(np.float64)
print(f"  inv@A:\n{prod}")
print(f"  ||inv@A - I||_max = {np.abs(prod - np.eye(3)).max():.6e}")

print("\n=== V1: df64 determinant low word by scale ===")
M = np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 4.]])
for scale in (1.0, 1e-12, 1e12):
    A2 = (scale * M).astype(np.float32)
    a = wp.array(A2.reshape(1, 3, 3), dtype=wp.mat33, device=DEV)
    hi = wp.zeros(1, dtype=wp.float32, device=DEV)
    lo = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(det_lowword, dim=1, inputs=[a, hi, lo], device=DEV)
    wp.synchronize_device(DEV)
    h, l = float(hi.numpy()[0]), float(lo.numpy()[0])
    print(f"  scale={scale:.0e}: det hi={h:.6e} lo={l:.6e}  low word {'FLUSHED/zero' if l == 0.0 else 'alive'}")

print("\n=== V4: fast-math canary ===")
c = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(canary, dim=1, inputs=[c], device=DEV)
wp.synchronize_device(DEV)
cv = float(c.numpy()[0])
print(f"  (1+1e-10)-1 = {cv:.6e}  -> {'PASS (low word survives)' if abs(cv-1e-10) < 1e-12 else 'FAIL: compiler ate the low word'}")

print("\n=== R3: NaN ordering, cpu vs metal (no CUDA on this host) ===")
for dev in ("cpu", DEV):
    o = wp.zeros(4, dtype=wp.float32, device=dev)
    wp.launch(nan_order, dim=1, inputs=[o], device=dev)
    wp.synchronize_device(dev)
    v = o.numpy()
    print(f"  {dev:8s} max(NaN,1)={v[0]!r} max(1,NaN)={v[1]!r} min(NaN,1)={v[2]!r} min(1,NaN)={v[3]!r}")

print("\n=== CLIFF: det overflow at 1e13 scale ===")
A3 = (1e13 * M).astype(np.float32)
print(f"  host float64 det = {np.linalg.det(A3.astype(np.float64)):.3e}  (float32 max 3.4e38)")
inv3 = run(inv_f64, [A3])[0]
print(f"  df64 inverse diag = {np.diag(inv3)}")
print(f"  contains NaN: {bool(np.isnan(inv3).any())}  all-zero: {bool(np.all(inv3 == 0))}")
