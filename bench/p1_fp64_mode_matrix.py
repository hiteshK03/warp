"""Capture fp64 behaviour so 'error' mode can be VERIFIED against pre-P1, not asserted.

Runs a set of fp64-using kernels and records, for each, whether it launched or
raised (and the reason). Run this on:
  - the pre-P1 overlay (git stash / checkout metal-1.17 headers+context)  -> baseline
  - the new opt-in build with metal_float64='error'                      -> must match
  - the new opt-in build with metal_float64='df64'                       -> must launch

Usage: python bench/p1_fp64_mode_matrix.py <label>
Writes bench/fp64_mode_<label>.json
"""
import json
import sys

import numpy as np
import warp as wp

label = sys.argv[1] if len(sys.argv) > 1 else "current"
wp.init()
DEV = "metal:0"
wp.set_device(DEV)

results = {}


def record(name, fn):
    try:
        fn()
        results[name] = {"launched": True}
    except Exception as e:
        results[name] = {"launched": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# 1. internal fp64 scalar math (the case P1 opened)
@wp.kernel
def internal_fp64(out: wp.array(dtype=wp.float32)):
    x = wp.float64(1.0) / wp.float64(3.0)
    out[0] = wp.float32(x)


def run_internal():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(internal_fp64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)


# 2. fp64 matrix inverse via mat33d (Newton's blocking kernel shape)
@wp.kernel
def mat_inverse(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    out[i] = wp.mat33(wp.inverse(wp.mat33d(inp[i])))


def run_inverse():
    a = wp.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3), dtype=wp.mat33, device=DEV)
    o = wp.zeros(1, dtype=wp.mat33, device=DEV)
    wp.launch(mat_inverse, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)


# 3. fp64 ARRAY in the kernel ABI — must refuse in BOTH modes
@wp.kernel
def fp64_array(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = a[i] * wp.float64(2.0)


def run_array():
    a = wp.array(np.array([1.0], dtype=np.float64), dtype=wp.float64, device=DEV)
    o = wp.zeros(1, dtype=wp.float64, device=DEV)
    wp.launch(fp64_array, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)


# 4. fp64 SCALAR argument — the confirmed silent-garbage hole; must refuse in both
@wp.kernel
def fp64_scalar(s: wp.float64, out: wp.array(dtype=wp.float32)):
    out[0] = wp.float32(s)


def run_scalar():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(fp64_scalar, dim=1, inputs=[1.0, o], device=DEV)
    wp.synchronize_device(DEV)
    results["fp64_scalar_value"] = float(o.numpy()[0])


# 5. plain float32 kernel — must ALWAYS work (no collateral damage)
@wp.kernel
def plain_f32(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] * 2.0


def run_plain():
    a = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(plain_f32, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    assert float(o.numpy()[0]) == 6.0


record("internal_fp64", run_internal)
record("mat33d_inverse", run_inverse)
record("fp64_array_abi", run_array)
record("fp64_scalar_arg", run_scalar)
record("plain_float32", run_plain)

mode = getattr(wp.config, "metal_float64", "<absent>")
print(f"label={label}  wp.config.metal_float64={mode!r}")
for k, v in results.items():
    print(f"  {k:20s} {v}")
with open(f"bench/fp64_mode_{label}.json", "w") as f:
    json.dump({"label": label, "mode": str(mode), "results": results}, f, indent=2)
print(f"written bench/fp64_mode_{label}.json")
