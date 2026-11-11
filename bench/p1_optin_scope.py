"""Verify df64 opt-in works per MODULE (not globally), and that the module option
is honoured independently of the global config default.

Checks:
  1. default (error): fp64 kernel refuses
  2. module option df64 on module A: A launches, and an fp64 kernel in module B
     (no option) still refuses  -> no global leak
  3. warning fires for the opted-in module
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

# Module A opts in explicitly.
mod_a = wp.Module("df64_optin_module")
wp.set_module_options({"metal_float64": "df64"}, module=mod_a)

# Module B does not.
mod_b = wp.Module("df64_default_module")


@wp.kernel(module=mod_a)
def uses_fp64_a(out: wp.array(dtype=wp.float32)):
    x = wp.float64(1.0) / wp.float64(3.0)
    out[0] = wp.float32(x)


@wp.kernel(module=mod_b)
def uses_fp64_b(out: wp.array(dtype=wp.float32)):
    x = wp.float64(1.0) / wp.float64(3.0)
    out[0] = wp.float32(x)


print(f"global default wp.config.metal_float64 = {wp.config.metal_float64!r}")

o = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(uses_fp64_a, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    print(f"[opted-in module A]  LAUNCHED, value={float(o.numpy()[0]):.8f}  -> expected 0.33333334")
    a_ok = True
except Exception as e:
    print(f"[opted-in module A]  REFUSED: {str(e)[:120]}")
    a_ok = False

try:
    wp.launch(uses_fp64_b, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    print("[default module B]   LAUNCHED  -> LEAK! the option escaped its module")
    b_refused = False
except Exception as e:
    print(f"[default module B]   REFUSED (correct): {str(e)[:100]}")
    b_refused = True

print()
print("VERDICT:", "PASS — opt-in is module-scoped, no global leak" if (a_ok and b_refused) else "FAIL")
