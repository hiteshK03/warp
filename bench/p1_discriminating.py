"""The DISCRIMINATING test my validation was missing.

Problem: kernel outputs are float32, so any relative error bottoms out at
float32 output rounding (~6e-8). Both df64 and float32 intermediates look
identical. To prove df64 internals actually help, compare:
   A) inverse computed with df64 intermediates  (wp.mat33d)
   B) inverse computed with float32 intermediates (wp.mat33)  <- CONTROL
on inputs where float32 intermediates demonstrably lose accuracy, and measure
the residual ||inv(A) @ A - I|| computed in FLOAT64 ON THE HOST from the
float32 outputs. If df64 is doing real work, A must beat B on some input.

Also: cancellation-sensitive construction where the intermediate loses bits.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def inv_f64(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.mat33(wp.inverse(wp.mat33d(inp[i])))


@wp.kernel
def inv_f32(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    """CONTROL: same algorithm, float32 intermediates."""
    i = wp.tid()
    out[i] = wp.inverse(inp[i])


def run(kernel, mats):
    a = wp.array(np.array(mats, dtype=np.float32), dtype=wp.mat33, device=DEV)
    o = wp.zeros(len(mats), dtype=wp.mat33, device=DEV)
    wp.launch(kernel, dim=len(mats), inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return o.numpy()


def residual(inv_out, A):
    """||inv(A)@A - I||_max in host float64 — independent of device output rounding."""
    return np.abs(inv_out.astype(np.float64) @ A.astype(np.float64) - np.eye(3)).max()


cases = {}

# 1. ill-conditioned SPD: condition number ~1e6 (float32 cofactors lose bits)
Q = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
cases["illcond 1e6"] = (Q * np.array([1e-6, 1.0, 1.0])) @ Q.T

# 2. near-singular: two nearly-parallel rows -> catastrophic cancellation in det
cases["near-singular"] = np.array([[1.0, 1.0, 1.0],
                                   [1.0, 1.0 + 1e-6, 1.0],
                                   [1.0, 1.0, 1.0 + 1e-6]])

# 3. tiny scale, non-diagonal (the reviewer's predicted case)
cases["1e-12 non-diag"] = 1e-12 * np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 4.]])

# 4. large scale, non-diagonal
cases["1e12 non-diag"] = 1e12 * np.array([[2., 1., 0.], [1., 3., 1.], [0., 1., 4.]])

# 5. mixed magnitude but physically valid SPD (triangle inequality holds)
cases["mixed SPD"] = np.diag([1e-8, 5e-8, 6e-8])

print(f"{'case':18s} {'cond(A)':>10s} {'resid df64':>12s} {'resid f32':>12s} {'df64 better?':>13s}")
any_better = False
for name, A in cases.items():
    A = A.astype(np.float32)
    cond = np.linalg.cond(A.astype(np.float64))
    r64 = residual(run(inv_f64, [A])[0], A)
    r32 = residual(run(inv_f32, [A])[0], A)
    better = r64 < r32 * 0.5
    any_better = any_better or better
    print(f"{name:18s} {cond:10.2e} {r64:12.3e} {r32:12.3e} {str(better):>13s}")

print()
if any_better:
    print("DISCRIMINATING: df64 intermediates measurably beat float32 on at least one case.")
else:
    print("NOT DISCRIMINATING: no case shows df64 beating float32 by >2x.")
    print("=> Either df64 adds nothing here, or these inputs still don't stress it.")
    print("   Do NOT claim df64 improves accuracy without a case that shows it.")
