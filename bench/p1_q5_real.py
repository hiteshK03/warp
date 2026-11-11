"""Q5 gate (distinct kernels from a real module file). Warm-run bundle vs isolation."""
import time
import numpy as np
import warp as wp
from warp._src import context as ctx
import q5_kernels as K

wp.init()
DEV = "metal:0"
wp.set_device(DEV)
m = K.m
dev = wp.get_device(DEV)
options = m.resolve_options(wp.config) | {"output_arch": None, "block_dim": 256}
a = wp.array([1.0], dtype=float, device=DEV)
o = wp.zeros(1, dtype=float, device=DEV)

all_roots = list(ctx.ModuleHasher(list(m._get_live_kernels()), options).get_unique_kernels())
good_roots = [k for k in all_roots if "bad" not in k.key]
print(f"distinct roots: {len(all_roots)} total, {len(good_roots)} good")

# Warm Apple's shader cache.
wp.launch(K.goods[0], dim=1, inputs=[a, o, 2.0], device=DEV)
wp.synchronize_device(DEV)

t0 = time.perf_counter()
libs, refused, metas, bwd = m._metal_isolate_kernels(dev, options)
tA = time.perf_counter() - t0
for lib in libs.values():
    ctx.runtime.core.wp_metal_unload_library(lib)
print(f"A) per-kernel isolation: {tA*1000:.1f}ms ({len(libs)} libs, {len(refused)} refused)")

t0 = time.perf_counter()
hasher = ctx.ModuleHasher(good_roots, options)
builder = ctx.ModuleBuilder(m, options, hasher=hasher, device="metal")
src = builder.codegen("metal")
bundle = ctx.runtime.core.wp_metal_load_library(dev.metal_ordinal, src.encode("utf-8"))
tB = time.perf_counter() - t0
print(f"B) bundle (1 compile, {len(good_roots)} kernels): {tB*1000:.1f}ms compiled={bundle is not None}")
if bundle:
    ctx.runtime.core.wp_metal_unload_library(bundle)

print(f"WARM saving (A-B): {(tA-tB)*1000:.1f}ms/module")
print(f"Gate: {'WORTH BUILDING (>50ms)' if (tA-tB)*1000 > 50 else 'NOT WORTH IT (<50ms)'}")
