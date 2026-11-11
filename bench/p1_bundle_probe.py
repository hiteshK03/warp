"""Verify the bundle concept: ModuleHasher(subset_of_kernels) -> ModuleBuilder
emits ONE prelude + all subset kernels, and it compiles as a single library with
every kernel's symbols present. This is the foundation of bundle caching.
"""
import numpy as np
import warp as wp
from warp._src import context as ctx

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

m = wp.Module("bundle_probe")
wp.set_module_options({"metal_float64": "df64"}, module=m)


@wp.kernel(module=m)
def kA(a: wp.array[float], o: wp.array[float]):
    i = wp.tid()
    o[i] = a[i] * 2.0


@wp.kernel(module=m)
def kB(a: wp.array[float], o: wp.array[float]):
    i = wp.tid()
    o[i] = a[i] + 10.0


@wp.kernel(module=m)
def kbad(o: wp.array[float]):
    o[0] = wp.float32(wp.sin(wp.float64(0.5)))


# Resolve options as a load would.
options = m.resolve_options(wp.config)
options = options | {"output_arch": None, "block_dim": 256}

# Full hasher to get all roots (expands generics).
all_roots = list(ctx.ModuleHasher(list(m._get_live_kernels()), options).get_unique_kernels())
print("all roots:", [k.get_mangled_name() for k in all_roots])

# Subset: only the good kernels (kA, kB), excluding kbad.
good = [k for k in all_roots if "kbad" not in k.key]
print("good subset:", [k.get_mangled_name() for k in good])

# Build a bundle from the subset.
hasher = ctx.ModuleHasher(good, options)
builder = ctx.ModuleBuilder(m, options, hasher=hasher, device="metal")
src = builder.codegen("metal")
meta = builder.build_meta()

# Count preludes: the df64 header guard should appear once.
n_prelude = src.count("namespace wp {")
print(f"source length={len(src)}, 'namespace wp {{' count={n_prelude}")

# Compile the bundle as one library.
handle = ctx.runtime.core.wp_metal_load_library(wp.get_device(DEV).metal_ordinal, src.encode("utf-8"))
print("bundle compiled:", handle is not None)
if handle:
    for k in good:
        name = k.get_mangled_name()
        fwd = ctx.runtime.core.wp_metal_get_kernel(handle, (name + "_metal_forward").encode("utf-8"))
        print(f"  symbol {name}_metal_forward present: {fwd is not None}")
    # kbad must NOT be in the bundle
    badname = [k.get_mangled_name() for k in all_roots if "kbad" in k.key][0]
    badfwd = ctx.runtime.core.wp_metal_get_kernel(handle, (badname + "_metal_forward").encode("utf-8"))
    print(f"  kbad absent from bundle: {badfwd is None}")
    ctx.runtime.core.wp_metal_unload_library(handle)
