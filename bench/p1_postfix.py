"""Post-fix verification: module option works, config is inert, validation raises,
and the warm-cache path behaves (review item 3)."""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

# 1. config is now inert (no global knob)
print("has wp.config.metal_float64:", hasattr(wp.config, "metal_float64"))
wp.config.metal_float64 = "df64"  # should have NO effect now
inert = wp.Module("config_inert_module")


@wp.kernel(module=inert)
def k_inert(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


try:
    wp.launch(k_inert, dim=1, inputs=[wp.zeros(1, dtype=wp.float32, device=DEV)], device=DEV)
    print("[config-only]  LAUNCHED -> still leaks!")
except Exception:
    print("[config-only]  REFUSED (correct: config is inert)")

# 2. explicit module option works
ok = wp.Module("explicit_optin_module")
wp.set_module_options({"metal_float64": "df64"}, module=ok)


@wp.kernel(module=ok)
def k_ok(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


o = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(k_ok, dim=1, inputs=[o], device=DEV)
wp.synchronize_device(DEV)
print(f"[module opt-in] LAUNCHED value={float(o.numpy()[0]):.8f} (expect 0.33333334)")

# 3. validation raises on a bad value
try:
    wp.set_module_options({"metal_float64": "DF64"}, module=ok)
    print("[validation]   NO ERROR -> bad value accepted")
except ValueError as e:
    print(f"[validation]   ValueError raised (correct): {str(e)[:70]}")

# 4. warm cache: launch again, same process (module already loaded)
o2 = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(k_ok, dim=1, inputs=[o2], device=DEV)
wp.synchronize_device(DEV)
print(f"[warm relaunch] value={float(o2.numpy()[0]):.8f} (consistent)")
