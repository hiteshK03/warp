"""Measure the cost breakdown of Metal fault isolation: codegen vs compile.
Determines whether source-caching (skips codegen) saves the dominant cost.
Times _metal_isolate_kernels and its phases on a module with 1 bad + N good kernels.
"""
import time
import numpy as np
import warp as wp
from warp._src import context as ctx

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

# Build a module with several good kernels + one bad (sin float64).
m = wp.Module("cache_bench")
wp.set_module_options({"metal_float64": "df64"}, module=m)

NGOOD = 6
good_kernels = []
for j in range(NGOOD):
    @wp.kernel(module=m)
    def good(a: wp.array[float], o: wp.array[float], c: float):
        i = wp.tid()
        o[i] = a[i] * c + float(j)
    good_kernels.append(good)


@wp.kernel(module=m)
def bad(o: wp.array[float]):
    o[0] = wp.float32(wp.sin(wp.float64(0.5)))


a = wp.array([1.0], dtype=float, device=DEV)
o = wp.zeros(1, dtype=float, device=DEV)

# Instrument: time codegen vs load_library inside isolation.
timings = {"codegen": 0.0, "compile": 0.0, "n": 0}
orig_builder_codegen = ctx.ModuleBuilder.codegen
orig_load = ctx.runtime.core.wp_metal_load_library


def timed_codegen(self, device):
    t = time.perf_counter()
    r = orig_builder_codegen(self, device)
    timings["codegen"] += time.perf_counter() - t
    return r


def timed_load(ordinal, source):
    t = time.perf_counter()
    r = orig_load(ordinal, source)
    timings["compile"] += time.perf_counter() - t
    timings["n"] += 1
    return r


ctx.ModuleBuilder.codegen = timed_codegen
ctx.runtime.core.wp_metal_load_library = timed_load
try:
    t0 = time.perf_counter()
    wp.launch(good_kernels[0], dim=1, inputs=[a, o, 2.0], device=DEV)
    wp.synchronize_device(DEV)
    total = time.perf_counter() - t0
finally:
    ctx.ModuleBuilder.codegen = orig_builder_codegen
    ctx.runtime.core.wp_metal_load_library = orig_load

print(f"first launch (triggers isolation): total={total*1000:.1f}ms")
print(f"  codegen sum={timings['codegen']*1000:.1f}ms  compile sum={timings['compile']*1000:.1f}ms  compiles={timings['n']}")
print(f"  -> compile is {timings['compile']/max(timings['codegen'],1e-9):.1f}x codegen")
print(f"  -> source-caching would save ~{timings['codegen']*1000:.1f}ms of {total*1000:.1f}ms ({timings['codegen']/total*100:.0f}%)")
