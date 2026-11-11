"""Does a forward-only codegen override change the kernel's mangled name?
The mangled name comes from kernel.hash, set by hash_kernel, which folds
kernel.options (per-kernel) but NOT module options like enable_backward.
Verify: build a kernel forward+backward vs forward-only (module option), compare
mangled names. If stable, the bundle can hold forward-only kernels under their
original names. If not, we need a different mechanism.
"""
import warp as wp
from warp._src import context as ctx

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

m = wp.Module("name_stability_probe")


@wp.kernel(module=m)
def k(a: wp.array[float], o: wp.array[float]):
    i = wp.tid()
    o[i] = a[i] * a[i]


opts_fb = m.resolve_options(wp.config) | {"output_arch": None, "block_dim": 256, "enable_backward": True}
opts_fo = m.resolve_options(wp.config) | {"output_arch": None, "block_dim": 256, "enable_backward": False, "metal_forward_only": True}

h_fb = ctx.ModuleHasher([k], opts_fb)
name_fb = k.get_mangled_name()
h_fo = ctx.ModuleHasher([k], opts_fo)
name_fo = k.get_mangled_name()

print(f"forward+backward mangled: {name_fb}")
print(f"forward-only     mangled: {name_fo}")
print(f"STABLE: {name_fb == name_fo}")
print(f"kernel.options keys: {sorted(k.options.keys())}")
print(f"enable_backward in kernel.options: {'enable_backward' in k.options}")
