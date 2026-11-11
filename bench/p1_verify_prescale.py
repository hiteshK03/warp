"""Verify the pre-scaling review's claims. Most important: the DISCRIMINATING test
my earlier validation lacked.

R1 (methodology flaw): my "1e-12 is exact" used the IDENTITY, whose df64 low words
   are all zero — it cannot reveal precision loss. Predicted: for a non-diagonal
   1e-12-scale matrix, ||inv(A)@A - I|| ~ 1e-7..1e-8, not ~1e-14.
R2 (regression): max-abs scaling breaks diag(1e-20, 1, 1e20), which works TODAY.
R3: _wp_native_fmax drops NaN despite its "NaN-propagating" comment.
R4: overflow -> NaN (already confirmed, re-check via addition and division).
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def inv_kernel(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.mat33(wp.inverse(wp.mat33d(inp[i])))


@wp.kernel
def inv_scaled_kernel(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    """Reviewer's cheap fix: uniform diagonal-max scaling (SPD => max-abs)."""
    i = wp.tid()
    m = inp[i]
    s = wp.max(m[0, 0], wp.max(m[1, 1], m[2, 2]))
    if s > 0.0:
        md = wp.mat33d(m) / wp.float64(s)
        out[i] = wp.mat33(wp.inverse(md) / wp.float64(s))
    else:
        out[i] = wp.mat33(0.0)


@wp.kernel
def fmax_nan(out: wp.array(dtype=wp.float32)):
    nan = wp.float64(0.0) / wp.float64(0.0)
    one = wp.float64(1.0)
    out[0] = wp.float32(wp.max(nan, one))   # NaN-propagating would give NaN
    out[1] = wp.float32(wp.max(one, nan))


def run(kernel, mats):
    a = wp.array(np.array(mats, dtype=np.float32), dtype=wp.mat33, device=DEV)
    o = wp.zeros(len(mats), dtype=wp.mat33, device=DEV)
    wp.launch(kernel, dim=len(mats), inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return o.numpy()


print("=== R1: discriminating precision test (non-diagonal, 1e-12 scale) ===")
M = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
for scale in (1.0, 1e-7, 1e-12):
    A = (scale * M).astype(np.float32)
    ref = np.linalg.inv(A.astype(np.float64))
    got = run(inv_kernel, [A])[0]
    got_s = run(inv_scaled_kernel, [A])[0]
    # residual against the true inverse, normalized
    err = np.abs(got - ref).max() / np.abs(ref).max()
    err_s = np.abs(got_s - ref).max() / np.abs(ref).max()
    ident = np.abs(got.astype(np.float64) @ A.astype(np.float64) - np.eye(3)).max()
    print(f"  scale={scale:.0e}: unscaled rel_err={err:.2e}  scaled rel_err={err_s:.2e}  ||inv@A-I||={ident:.2e}")
print("  (reviewer predicted ~1e-7..1e-8 unscaled at 1e-12; identity tests cannot show this)")

print("\n=== R2: mixed-magnitude regression from max-abs scaling ===")
A = np.diag([1e-20, 1.0, 1e20]).astype(np.float32)
ref = np.linalg.inv(A.astype(np.float64))
got = run(inv_kernel, [A])[0]
got_s = run(inv_scaled_kernel, [A])[0]
print(f"  true inverse diag      = {np.diag(ref)}")
print(f"  unscaled (today) diag  = {np.diag(got)}")
print(f"  max-abs scaled diag    = {np.diag(got_s)}")
print(f"  VERDICT: scaling {'REGRESSES this case' if np.allclose(got_s, 0) else 'does not zero it'}")

print("\n=== R3: fmax NaN propagation ===")
o = wp.zeros(2, dtype=wp.float32, device=DEV)
wp.launch(fmax_nan, dim=1, inputs=[o], device=DEV)
wp.synchronize_device(DEV)
v = o.numpy()
print(f"  max(NaN, 1) = {v[0]!r}   max(1, NaN) = {v[1]!r}")
print(f"  VERDICT: {'NaN DROPPED (comment wrong)' if not np.isnan(v[0]) else 'propagates'}")
