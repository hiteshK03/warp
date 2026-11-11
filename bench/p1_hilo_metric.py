"""Hi/lo readback metric: measure df64 accuracy WITHOUT the float32 output ceiling.

Problem with all earlier tests: the kernel writes a float32 mat33, so any relative
error floors at ~6e-8 regardless of intermediate precision. df64's real accuracy
is invisible.

Fix: have the kernel emit the inverse as TWO float32 arrays (hi and lo words),
rebuild the value in host float64 as hi+lo, and compare against a host float64
inverse. This exposes the full df64 mantissa.

Also implements the reviewer's V3: many random matrices per (scale, cond) bucket,
reporting median and p99 of the df64/float32 error ratio, not one sample.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)
N = 200  # matrices per bucket


@wp.kernel
def inv_df64_hilo(inp: wp.array(dtype=wp.mat33),
                  hi: wp.array(dtype=wp.mat33), lo: wp.array(dtype=wp.mat33)):
    """df64 inverse, emitted as hi/lo word pairs so no precision is lost at the store."""
    i = wp.tid()
    inv = wp.inverse(wp.mat33d(inp[i]))
    for r in range(3):
        for c in range(3):
            h = wp.float32(inv[r, c])
            hi[i][r, c] = h
            lo[i][r, c] = wp.float32(inv[r, c] - wp.float64(h))


@wp.kernel
def inv_f32_ctrl(inp: wp.array(dtype=wp.mat33),
                 hi: wp.array(dtype=wp.mat33), lo: wp.array(dtype=wp.mat33)):
    """CONTROL: float32 intermediates, same output plumbing (lo stays zero)."""
    i = wp.tid()
    inv = wp.inverse(inp[i])
    for r in range(3):
        for c in range(3):
            hi[i][r, c] = inv[r, c]
            lo[i][r, c] = wp.float32(0.0)


def measure(kernel, mats):
    n = len(mats)
    a = wp.array(np.array(mats, dtype=np.float32), dtype=wp.mat33, device=DEV)
    hi = wp.zeros(n, dtype=wp.mat33, device=DEV)
    lo = wp.zeros(n, dtype=wp.mat33, device=DEV)
    wp.launch(kernel, dim=n, inputs=[a, hi, lo], device=DEV)
    wp.synchronize_device(DEV)
    # rebuild in host float64: this is where the extra mantissa becomes visible
    return hi.numpy().astype(np.float64) + lo.numpy().astype(np.float64)


def rel_err(got, ref):
    num = np.abs(got - ref).max(axis=(1, 2))
    den = np.maximum(np.abs(ref).max(axis=(1, 2)), 1e-300)
    return num / den


def make_bucket(scale, cond, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        lam = np.array([1.0 / cond, 1.0, 1.0])
        out.append((scale * ((Q * lam) @ Q.T)).astype(np.float32))
    return np.array(out)


print(f"hi/lo readback metric, {N} random SPD matrices per bucket")
print(f"{'scale':>8s} {'cond':>8s} {'df64 med':>11s} {'f32 med':>11s} {'ratio med':>10s} {'ratio p99':>10s}")
for scale in (1.0, 1e-12, 1e12):
    for cond in (1e2, 1e6):
        mats = make_bucket(scale, cond, N, seed=hash((scale, cond)) % 2**31)
        ref = np.linalg.inv(mats.astype(np.float64))
        e64 = rel_err(measure(inv_df64_hilo, mats), ref)
        e32 = rel_err(measure(inv_f32_ctrl, mats), ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = e32 / np.maximum(e64, 1e-300)
        finite = np.isfinite(ratio)
        print(f"{scale:8.0e} {cond:8.0e} {np.median(e64):11.2e} {np.median(e32):11.2e} "
              f"{np.median(ratio[finite]):10.1f} {np.percentile(ratio[finite], 99):10.1f}")

print("\nratio > 1 means df64 intermediates are more accurate than float32.")
print("This metric has NO float32 output ceiling: hi+lo is rebuilt in host float64.")
