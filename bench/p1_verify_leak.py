"""Verify implementation-review items 2 (config leak/ordering) and 3 (warm cache).

Item 2a LEAK: library sets wp.config.metal_float64='df64' at import; a user module
  created AFTER inherits it -> user's fp64 kernel launches under emulation.
Item 2b ORDER: kernel defined FIRST, then config set to df64 -> should launch per
  'config supplies the default', but the default was captured at module construction.
Item 3 WARM: run an opted-in module twice without clearing cache; does the warning
  fire the second time?
"""
import subprocess
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

print("=== Item 2a: global config leak (library sets df64, user module inherits) ===")
# Simulate: config set to df64 BEFORE the user module is created.
wp.config.metal_float64 = "df64"
user_mod = wp.Module("user_module_after_import")  # created after the config change


@wp.kernel(module=user_mod)
def user_fp64(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


o = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(user_fp64, dim=1, inputs=[o], device=DEV)
    wp.synchronize_device(DEV)
    print(f"  user kernel LAUNCHED (value={float(o.numpy()[0]):.6f}) -> LEAK CONFIRMED")
    print("  (user never opted in; the library's global config leaked into their module)")
except Exception as e:
    print(f"  user kernel REFUSED -> no leak: {str(e)[:90]}")

print("\n=== Item 2b: ordering (kernel defined first, config set after) ===")
wp.config.metal_float64 = "error"
early_mod = wp.Module("early_module")


@wp.kernel(module=early_mod)
def early_fp64(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0) / wp.float64(3.0))


wp.config.metal_float64 = "df64"  # set AFTER the module/kernel exist
o2 = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(early_fp64, dim=1, inputs=[o2], device=DEV)
    wp.synchronize_device(DEV)
    print("  LAUNCHED -> config-as-default honoured lazily")
except Exception as e:
    print(f"  REFUSED -> default captured too early (ordering bug): {str(e)[:80]}")

print("\n=== Item 2c: invalid value silently means error mode ===")
wp.config.metal_float64 = "DF64"  # wrong case
inv_mod = wp.Module("invalid_value_module")


@wp.kernel(module=inv_mod)
def inv_fp64(o: wp.array(dtype=wp.float32)):
    o[0] = wp.float32(wp.float64(1.0))


try:
    wp.launch(inv_fp64, dim=1, inputs=[wp.zeros(1, dtype=wp.float32, device=DEV)], device=DEV)
    print("  'DF64' LAUNCHED -> treated as df64?!")
except Exception as e:
    print(f"  'DF64' refused (fails safe, but no validation error): {str(e)[:60]}")
