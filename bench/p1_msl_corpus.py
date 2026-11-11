"""Differential corpus: prove error mode == pre-P1 at the GENERATED-SOURCE level.

The review's requirement: behaviour matching on 5 cases is not enough. For each
case record (1) generated MSL text, (2) launch outcome, (3) exception type+message,
(4) module hash. Any MSL diff in error mode falsifies the claim.

Usage: python bench/p1_msl_corpus.py <label> <cache_dir>
Writes bench/corpus_<label>.json and <cache_dir>/*.metal
"""

import json
import os
import pathlib
import sys

from typing import Any

import numpy as np
import warp as wp

label = sys.argv[1]
cache_dir = sys.argv[2]
pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
wp.config.kernel_cache_dir = cache_dir
wp.config.cache_kernels = True

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


# --- 1. float32 only (must always work) ---
@wp.kernel(module="c_f32")
def k_f32(a: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = a[i] * 2.0


@case("f32_only")
def _():
    a = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_f32, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 2. float64 scalar internal ---
@wp.kernel(module="c_f64scalar")
def k_f64scalar(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


@case("f64_scalar_internal")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_f64scalar, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 3. vec3d ---
@wp.kernel(module="c_vec3d")
def k_vec3d(o: wp.array(dtype=wp.float32)):
    v = wp.vec3d(1.0, 2.0, 3.0)
    o[0] = wp.float32(wp.dot(v, v))


@case("vec3d")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_vec3d, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 4. mat33d inverse ---
@wp.kernel(module="c_mat33d")
def k_mat33d(a: wp.array(dtype=wp.mat33), o: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    o[i] = wp.mat33(wp.inverse(wp.mat33d(a[i])))


@case("mat33d_inverse")
def _():
    a = wp.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3), dtype=wp.mat33, device=DEV)
    o = wp.zeros(1, dtype=wp.mat33, device=DEV)
    wp.launch(k_mat33d, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0][0, 0])


# --- 5. quatd ---
@wp.kernel(module="c_quatd")
def k_quatd(o: wp.array(dtype=wp.float32)):
    q = wp.quatd(0.0, 0.0, 0.0, 1.0)
    v = wp.quat_rotate(q, wp.vec3d(1.0, 0.0, 0.0))
    o[0] = wp.float32(v[0])


@case("quatd")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_quatd, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 6. float64 struct field ---
@wp.struct
class S64:
    x: wp.float64
    y: float


@wp.kernel(module="c_struct64")
def k_struct64(o: wp.array(dtype=wp.float32)):
    s = S64()
    s.x = wp.float64(2.0)
    s.y = 3.0
    o[0] = wp.float32(s.x) + s.y


@case("struct_f64_field")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_struct64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 7. generic @wp.func instantiated at float64 ---
@wp.func
def double_it(v: Any):
    return v * 2.0


@wp.kernel(module="c_genericfunc")
def k_generic(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(double_it(wp.float64(3.0)))


@case("generic_func_f64")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_generic, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 8. wp.sin on float64 (no df64 overload defined) ---
@wp.kernel(module="c_sin64")
def k_sin64(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.sin(wp.float64(1.0)))


@case("sin_f64")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_sin64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 9. int64 -> float64 cast (claimed ambiguous ctor) ---
@wp.kernel(module="c_int64cast")
def k_int64cast(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(wp.int64(5)))


@case("int64_to_f64")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_int64cast, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 10. float64 array in ABI ---
@wp.kernel(module="c_arr64")
def k_arr64(a: wp.array(dtype=wp.float64), o: wp.array(dtype=wp.float64)):
    i = wp.tid()
    o[i] = a[i] * wp.float64(2.0)


@case("f64_array_abi")
def _():
    a = wp.array(np.array([1.0], dtype=np.float64), dtype=wp.float64, device=DEV)
    o = wp.zeros(1, dtype=wp.float64, device=DEV)
    wp.launch(k_arr64, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 11. float64 atomic ---
@wp.kernel(module="c_atomic64")
def k_atomic64(o: wp.array(dtype=wp.float64)):
    wp.atomic_add(o, 0, wp.float64(1.0))


@case("f64_atomic")
def _():
    o = wp.zeros(1, dtype=wp.float64, device=DEV)
    wp.launch(k_atomic64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 12. float64 scalar ARGUMENT ---
@wp.kernel(module="c_scalararg")
def k_scalararg(s: wp.float64, o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(s)


@case("f64_scalar_arg")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_scalararg, dim=1, inputs=[1.0, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 13. MIXED module: clean f32 kernel + fp64 kernel in the SAME module ---
@wp.kernel(module="c_mixed")
def k_mixed_clean(a: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = a[i] + 1.0


@wp.kernel(module="c_mixed")
def k_mixed_fp64(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(7.0))


@case("mixed_clean_kernel_launches")
def _():
    a = wp.array(np.array([1.0], dtype=np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_mixed_clean, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


@case("mixed_fp64_kernel")
def _():
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_mixed_fp64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


# --- 14. mixed module with sin(fp64): does the clean kernel still launch? ---
@wp.kernel(module="c_mixed_sin")
def k_ms_clean(a: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = a[i] * 3.0


@wp.kernel(module="c_mixed_sin")
def k_ms_sin(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.sin(wp.float64(0.5)))


@case("mixed_sin_clean_launches")
def _():
    a = wp.array(np.array([2.0], dtype=np.float32), dtype=wp.float32, device=DEV)
    o = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(k_ms_clean, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    return float(o.numpy()[0])


results = {}
for name, fn in CASES.items():
    try:
        val = fn()
        results[name] = {"launched": True, "value": val}
    except Exception as e:
        results[name] = {"launched": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

# module hashes
hashes = {}
for mname in ("c_f32", "c_f64scalar", "c_vec3d", "c_mat33d", "c_quatd", "c_struct64",
              "c_genericfunc", "c_sin64", "c_int64cast", "c_arr64", "c_atomic64",
              "c_scalararg", "c_mixed", "c_mixed_sin"):
    try:
        m = wp.get_module(mname)
        hashes[mname] = m.get_module_hash().hex()[:16]
    except Exception as e:
        hashes[mname] = f"<{type(e).__name__}>"

out = {"label": label, "mode": str(getattr(wp.config, "metal_float64", "<absent>")),
       "results": results, "hashes": hashes}
pathlib.Path(f"bench/corpus_{label}.json").write_text(json.dumps(out, indent=2))
print(f"label={label} mode={out['mode']}")
for k, v in results.items():
    print(f"  {k:30s} {'OK ' + str(v.get('value')) if v['launched'] else 'REFUSED ' + v['error'][:70]}")
print(f"written bench/corpus_{label}.json ; MSL in {cache_dir}")
