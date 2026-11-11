"""M-A / M-B: is df64 justified on Newton's REAL inertia path?

Newton stores body_inv_inertia as float32 mat33. If df64's extra mantissa is
discarded at that store, it buys nothing here.

Three paths, all taking the SAME stored float32 inertia as input:
  (a) df64 intermediates -> hi/lo readback, rebuilt in host float64  [theoretical gain]
  (b) df64 intermediates -> rounded to float32                      [what Newton stores]
  (c) float32 intermediates -> float32                              [today's behaviour]
Reference: host float64 inverse of the stored float32 inertia.

Decisive number: how many real bodies have (b) != (c) bitwise.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

import newton
import newton.examples


@wp.kernel
def inv_df64(inp: wp.array(dtype=wp.mat33),
             rounded: wp.array(dtype=wp.mat33),
             hi: wp.array(dtype=wp.mat33),
             lo: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    inv = wp.inverse(wp.mat33d(inp[i]))
    for r in range(3):
        for c in range(3):
            h = wp.float32(inv[r, c])
            rounded[i][r, c] = h
            hi[i][r, c] = h
            lo[i][r, c] = wp.float32(inv[r, c] - wp.float64(h))


@wp.kernel
def inv_f32(inp: wp.array(dtype=wp.mat33), rounded: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    inv = wp.inverse(inp[i])
    for r in range(3):
        for c in range(3):
            rounded[i][r, c] = inv[r, c]


def collect_inertias():
    """Real inertia tensors from Newton's shipped models."""
    out = {}
    specs = [
        ("quadruped.urdf", "add_urdf"),
        ("nv_ant.xml", "add_mjcf"),
        ("nv_humanoid.xml", "add_mjcf"),
        ("cartpole.urdf", "add_urdf"),
    ]
    for asset, method in specs:
        b = newton.ModelBuilder()
        s = newton.ModelBuilder()
        kwargs = {"joint_damping_mode": "drive"} if method == "add_urdf" else {}
        getattr(s, method)(open(newton.examples.get_asset(asset)).read(), **kwargs)
        s.add_ground_plane()
        b.add_world(s)
        m = b.finalize(skip_validation_inertia=True)
        out[asset] = m.body_inertia.numpy().copy()
    return out


models = collect_inertias()
all_mats, labels = [], []
for name, mats in models.items():
    for j, M in enumerate(mats):
        all_mats.append(M)
        labels.append(f"{name}#body{j}")
all_mats = np.array(all_mats, dtype=np.float32)
n = len(all_mats)

# magnitude census — confirm we are nowhere near the FLT_MAX guard regression
mags = np.abs(all_mats).max(axis=(1, 2))
nz = mags[mags > 0]
print(f"real bodies: {n} (from {len(models)} models)")
print(f"|inertia| max-abs: min={nz.min():.3e} median={np.median(nz):.3e} max={nz.max():.3e}")
print(f"zero-inertia bodies (point masses / kinematic): {int((mags == 0).sum())}")
print(f"FLT_MAX = {np.finfo(np.float32).max:.3e} -> guard-regression region is {'FAR away' if nz.max() < 1e30 else 'NEARBY'}")

a = wp.array(all_mats, dtype=wp.mat33, device=DEV)
rounded64 = wp.zeros(n, dtype=wp.mat33, device=DEV)
hi = wp.zeros(n, dtype=wp.mat33, device=DEV)
lo = wp.zeros(n, dtype=wp.mat33, device=DEV)
rounded32 = wp.zeros(n, dtype=wp.mat33, device=DEV)
wp.launch(inv_df64, dim=n, inputs=[a, rounded64, hi, lo], device=DEV)
wp.launch(inv_f32, dim=n, inputs=[a, rounded32], device=DEV)
wp.synchronize_device(DEV)

b_store = rounded64.numpy()                       # (b) df64 -> float32 store
c_store = rounded32.numpy()                       # (c) float32 -> float32 store
a_full = hi.numpy().astype(np.float64) + lo.numpy().astype(np.float64)   # (a) full df64
ref = np.linalg.inv(all_mats.astype(np.float64))  # reference from the SAME stored inputs

# skip singular bodies (inverse is zero by contract; not informative)
valid = mags > 0
det_ok = np.abs(np.linalg.det(all_mats.astype(np.float64))) > 0
use = valid & det_ok


def relerr(x):
    num = np.abs(x - ref).max(axis=(1, 2))
    den = np.maximum(np.abs(ref).max(axis=(1, 2)), 1e-300)
    return num / den


ea, eb, ec = relerr(a_full), relerr(b_store.astype(np.float64)), relerr(c_store.astype(np.float64))
print(f"\ninformative bodies (non-zero, non-singular): {int(use.sum())}")
print(f"  (a) df64 hi+lo rebuilt f64 : median rel err {np.median(ea[use]):.3e}")
print(f"  (b) df64 -> float32 store  : median rel err {np.median(eb[use]):.3e}")
print(f"  (c) float32 -> float32     : median rel err {np.median(ec[use]):.3e}")

same = np.array([np.array_equal(b_store[i], c_store[i]) for i in range(n)])
diff_idx = [i for i in range(n) if use[i] and not same[i]]
print(f"\nDECISIVE (M-B): bodies where df64-store != float32-store bitwise: "
      f"{len(diff_idx)} / {int(use.sum())}")
if diff_idx:
    d = np.abs(b_store[diff_idx].astype(np.float64) - c_store[diff_idx].astype(np.float64))
    rel = d / np.maximum(np.abs(ref[diff_idx]), 1e-300)
    print(f"  max relative difference: {rel.max():.3e}")
    for i in diff_idx[:5]:
        print(f"    {labels[i]}: |inertia|={mags[i]:.3e} relerr b={eb[i]:.2e} c={ec[i]:.2e}")
else:
    print("  => the float32 store discards df64's entire gain on every real body")
