"""Characterize the df64 range failure precisely, and find where it starts.

The self-check showed 1/det -> NaN for a 1e-15-scale inertia. Determine:
  - at what scale df64 division starts producing NaN / inf / garbage
  - whether the hi/lo split underflows independently (lo underflows long before hi)
  - what Newton-scale inertias actually look like (is this reachable in practice?)
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

SCALES = [1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 1e-15, 1e-20]


@wp.kernel
def probe(scales: wp.array(dtype=wp.float32), det_out: wp.array(dtype=wp.float32),
          inv_out: wp.array(dtype=wp.float32), lo_alive: wp.array(dtype=wp.int32)):
    i = wp.tid()
    s = wp.float64(scales[i])
    det = s * s * s
    det_out[i] = wp.float32(det)
    inv_out[i] = wp.float32(wp.float64(1.0) / det)
    # does the low word survive? add a delta below float32 resolution of s
    t = s + s * wp.float64(1.0e-9)
    lo_alive[i] = 1 if (t - s) > wp.float64(0.0) else 0


sc = wp.array(np.array(SCALES, dtype=np.float32), dtype=wp.float32, device=DEV)
det = wp.zeros(len(SCALES), dtype=wp.float32, device=DEV)
inv = wp.zeros(len(SCALES), dtype=wp.float32, device=DEV)
lo = wp.zeros(len(SCALES), dtype=wp.int32, device=DEV)
wp.launch(probe, dim=len(SCALES), inputs=[sc, det, inv, lo], device=DEV)
wp.synchronize_device(DEV)

d, iv, lv = det.numpy(), inv.numpy(), lo.numpy()
print(f"{'scale':>10s} {'true det':>12s} {'df64 det':>12s} {'true 1/det':>12s} {'df64 1/det':>12s} {'lo alive':>9s}  status")
for i, s in enumerate(SCALES):
    td = s**3
    ti = 1.0 / td
    bad = np.isnan(iv[i]) or np.isinf(iv[i]) or (ti < 3.4e38 and abs(iv[i] - ti) / ti > 1e-3)
    print(f"{s:10.0e} {td:12.3e} {d[i]:12.3e} {ti:12.3e} {iv[i]:12.3e} {str(bool(lv[i])):>9s}  {'BROKEN' if bad else 'ok'}")

print("\nfloat32 limits: max ~3.40e38, min normal ~1.18e-38, min subnormal ~1.40e-45")
print("=> df64 cannot represent 1/det once det < ~3e-39, and det itself loses all")
print("   precision below ~1e-38 (subnormal). NaN appears instead of a clean inf.")
