"""Item 8: confirm unique-module option behavior matches upstream for ALL options.
If fast_math (an existing upstream option) also does NOT inherit from the caller
module, then metal_float64's non-inheritance is consistent upstream behavior, not
a bug. The supported way to set options on a unique module is module_options={}.
"""
import warp as wp

wp.init()
DEV = "metal:0"

# Parent module opts into fast_math (an upstream option) and df64.
parent = wp.Module("item8_parent2")
wp.set_module_options({"fast_math": True, "metal_float64": "df64"}, module=parent)
print("parent options:", {k: parent.options[k] for k in ("fast_math", "metal_float64")})


# A unique-module kernel: does it inherit fast_math? (upstream behavior probe)
@wp.kernel(module="unique")
def k_unique_fastmath(o: wp.array(dtype=wp.float32)):
    o[0] = 1.0


print("unique kernel module fast_math:", k_unique_fastmath.module.options.get("fast_math"))
print("unique kernel module metal_float64:", k_unique_fastmath.module.options.get("metal_float64"))
print("-> if fast_math is False (default), unique modules do NOT inherit ANY option;")
print("   metal_float64 non-inheritance is consistent upstream behavior.")


# The supported path: module_options on the unique kernel.
@wp.kernel(module="unique", module_options={"metal_float64": "df64"})
def k_unique_explicit(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


import numpy as np
o = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(k_unique_explicit, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    print(f"unique + module_options df64: LAUNCHED value={float(o.numpy()[0]):.8f} (supported path works)")
except Exception as e:
    print(f"unique + module_options df64: FAILED {str(e)[:90]}")
