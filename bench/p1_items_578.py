"""Open verification items 5, 7, 8.
Item 5: does the metal_float64 option perturb module hashes on CPU/CUDA?
Item 7: in a df64-opted-in module, does a sin(float64) kernel get refused
        individually, or does it break the whole module (taking down a clean f32 kernel)?
Item 8: does module="unique" inherit the parent module's metal_float64 option?
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"

print("=== Item 5: cross-backend hash perturbation ===")
# Build the same float32-only kernel on CPU and Metal, with the option at default
# vs explicitly 'error'. If the option is hashed on all backends, CPU hashes differ
# from a build that never had the option. We compare CPU hash with option default
# vs a fresh module — and check whether df64 mode changes a CPU module's hash.
m_cpu_default = wp.Module("hash_cpu_default")
m_cpu_df64 = wp.Module("hash_cpu_df64")
wp.set_module_options({"metal_float64": "df64"}, module=m_cpu_df64)


@wp.kernel(module=m_cpu_default)
def k_cpu_a(o: wp.array(dtype=wp.float32)):
    o[0] = 1.0


@wp.kernel(module=m_cpu_df64)
def k_cpu_b(o: wp.array(dtype=wp.float32)):
    o[0] = 1.0


h_default = m_cpu_default.get_module_hash().hex()[:16]
h_df64 = m_cpu_df64.get_module_hash().hex()[:16]
print(f"  CPU module, option default: {h_default}")
print(f"  CPU module, option df64:    {h_df64}")
print(f"  -> option perturbs CPU hash: {h_default != h_df64}")
print("  (review item 5: ideally the option only affects Metal hashes)")

print("\n=== Item 7: per-kernel refusal in a df64 module with sin(float64) ===")
m7 = wp.Module("item7_mixed_sin")
wp.set_module_options({"metal_float64": "df64"}, module=m7)


@wp.kernel(module=m7)
def k7_clean(a: wp.array(dtype=wp.float32), o: wp.array(dtype=wp.float32)):
    i = wp.tid()
    o[i] = a[i] * 2.0


@wp.kernel(module=m7)
def k7_sin(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.sin(wp.float64(0.5)))


# Try the clean kernel FIRST (does the module compile at all?)
a = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEV)
o = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(k7_clean, dim=1, inputs=[a, o], device=DEV)
    wp.synchronize_device(DEV)
    print(f"  clean f32 kernel in df64 module: LAUNCHED value={float(o.numpy()[0])}")
    clean_ok = True
except Exception as e:
    print(f"  clean f32 kernel: FAILED -> {str(e)[:120]}")
    clean_ok = False
try:
    wp.launch(k7_sin, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    print(f"  sin(float64) kernel: LAUNCHED value={float(o.numpy()[0])} (no df64 sin overload?)")
except Exception as e:
    print(f"  sin(float64) kernel: refused/failed -> {str(e)[:120]}")
print(f"  -> item 7: clean kernel {'SURVIVES' if clean_ok else 'BROKEN by sibling'}")

print("\n=== Item 8: module='unique' inheritance ===")
m8 = wp.Module("item8_parent")
wp.set_module_options({"metal_float64": "df64"}, module=m8)


@wp.kernel(module="unique")
def k8_unique(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


try:
    wp.launch(k8_unique, dim=1, inputs=[wp.zeros(1, dtype=wp.float32, device=DEV)], device=DEV)
    wp.synchronize_device(DEV)
    print("  unique-module fp64 kernel: LAUNCHED (inherited df64)")
except Exception as e:
    print(f"  unique-module fp64 kernel: REFUSED -> {str(e)[:90]}")
    print("  (fails safe; unique modules may not inherit parent options)")
