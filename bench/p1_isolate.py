import numpy as np
import warp as wp

wp.init()
wp.set_device("metal:0")


@wp.kernel
def one(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x = wp.float64(a[i])
    y = wp.float64(b[i])
    out[0] = wp.float32(x + y)
    out[1] = wp.float32(x - y)
    out[2] = wp.float32(x * y)
    out[3] = wp.float32(x / y)
    out[4] = wp.float32(wp.sqrt(x))


fmax = np.float32(np.finfo(np.float32).max)
a = wp.array(np.array([fmax], dtype=np.float32), dtype=wp.float32, device="metal:0")
b = wp.array(np.array([np.float32(9e30)], dtype=np.float32), dtype=wp.float32, device="metal:0")
o = wp.zeros(5, dtype=wp.float32, device="metal:0")
wp.launch(one, dim=1, inputs=[a, b, o], device="metal:0")
wp.synchronize_device("metal:0")
g = o.numpy()
ad, bd = np.float64(fmax), np.float64(9e30)
with np.errstate(all="ignore"):
    r = np.array([ad + bd, ad - bd, ad * bd, ad / bd, np.sqrt(ad)], dtype=np.float32)
for n, gv, rv in zip(["+", "-", "*", "/", "sqrt"], g, r):
    same = (np.isnan(gv) and np.isnan(rv)) or gv == rv or (np.isinf(gv) and np.isinf(rv) and np.sign(gv) == np.sign(rv))
    print(f"{n:5s} got={gv!r:26s} ref={rv!r:26s} {'MISMATCH' if not same else 'ok'}")
print()
print("note: sqrt(FLT_MAX) true =", np.sqrt(ad), "-> f32", np.float32(np.sqrt(ad)))
