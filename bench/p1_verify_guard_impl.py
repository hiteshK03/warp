"""Verify the df_guard IMPLEMENTATION review. Item 1 first: it claims my fix
turned a NaN into a confidently WRONG -inf for finite inputs (a regression).

All inputs loaded from buffers at runtime. Operands with NONZERO lo words are
built in-kernel (x0+x1) since the previous test only ever had lo==0.
Signed zero compared by BIT PATTERN, not == (the review's harness critique).
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)
FLT_MAX = np.finfo(np.float32).max


@wp.kernel
def item1_div(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
              out: wp.array(dtype=wp.float32)):
    """I1: FLT_MAX / (2 - 2^-22) — finite/finite, claimed to give -inf."""
    i = wp.tid()
    out[i] = wp.float32(wp.float64(a[i]) / wp.float64(b[i]))


@wp.kernel
def item2_compound(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    """I2: sqrt(x)*sqrt(x) with x=FLT_MAX — claimed inf, true value FLT_MAX."""
    i = wp.tid()
    y = wp.sqrt(wp.float64(x[i]))
    out[i] = wp.float32(y * y)


@wp.kernel
def item2_add(x0: wp.array(dtype=wp.float32), x1: wp.array(dtype=wp.float32),
              x2: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    """I2 add: (FLT_MAX + -2^102) + 2^103 — nonzero lo words, claimed inf."""
    i = wp.tid()
    s = wp.float64(x0[i]) + wp.float64(x1[i])   # lo becomes nonzero here
    out[i] = wp.float32(s + wp.float64(x2[i]))


@wp.kernel
def item5_signedzero(out: wp.array(dtype=wp.float32)):
    """I5: signed zero; sign bits inspected host-side via numpy."""
    nz = wp.float64(-0.0)
    out[0] = wp.float32(nz)
    prod = wp.float64(-1.0) * wp.float64(0.0)
    out[1] = wp.float32(prod)
    out[2] = wp.float32(wp.float64(1.0) / prod)
    out[3] = wp.float32(wp.sqrt(nz))


@wp.kernel
def item6_sqrt_subnormal(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    """I6: sqrt of a subnormal input — claimed +inf."""
    i = wp.tid()
    out[i] = wp.float32(wp.sqrt(wp.float64(x[i])))


@wp.kernel
def outside_diff(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    """floor/int just below an integer; mod semantics vs fmod."""
    i = wp.tid()
    v = wp.float64(x[i]) - wp.float64(0.25)     # 2^30 - 0.25
    out[0] = wp.float32(wp.floor(v))            # want 1073741823
    out[1] = wp.float32(wp.int32(v))            # want 1073741823
    out[2] = wp.float32(wp.mod(wp.float64(-1.0), wp.float64(3.0)))  # fmod: -1
    out[3] = wp.float32(wp.mod(wp.float64(1.0), wp.float64(3.0e38) * wp.float64(3.0e38)))  # fmod(1,inf)=1


def bits(x):
    return hex(int(x))


print("=== I1 (HIGH): finite/finite division near FLT_MAX ===")
# b = 2 - 2^-22 = 0x3FFFFFFE
b_val = np.uint32(0x3FFFFFFE).view(np.float32) if hasattr(np.uint32(0), "view") else np.array([0x3FFFFFFE], dtype=np.uint32).view(np.float32)[0]
a = wp.array(np.array([FLT_MAX], dtype=np.float32), dtype=wp.float32, device=DEV)
b = wp.array(np.array([b_val], dtype=np.float32), dtype=wp.float32, device=DEV)
o = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(item1_div, dim=1, inputs=[a, b, o], device=DEV)
wp.synchronize_device(DEV)
got = float(o.numpy()[0])
ref = float(np.float32(np.float64(FLT_MAX) / np.float64(b_val)))
print(f"  b = {b_val!r} (0x3FFFFFFE)")
print(f"  got={got!r}  expected={ref!r}")
print(f"  VERDICT: {'BUG CONFIRMED — wrong sign infinity' if np.isinf(got) and got < 0 else 'not reproduced'}")

print("\n=== I2: compound sqrt(x)*sqrt(x), x=FLT_MAX ===")
x = wp.array(np.array([FLT_MAX], dtype=np.float32), dtype=wp.float32, device=DEV)
o2 = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(item2_compound, dim=1, inputs=[x, o2], device=DEV)
wp.synchronize_device(DEV)
g2 = float(o2.numpy()[0])
print(f"  got={g2!r}  expected={FLT_MAX!r}")
print(f"  VERDICT: {'BUG CONFIRMED (spurious inf)' if np.isinf(g2) else 'not reproduced'}")

print("\n=== I2 add: three-term sum with nonzero lo ===")
x0 = wp.array(np.array([FLT_MAX], dtype=np.float32), dtype=wp.float32, device=DEV)
x1 = wp.array(np.array([-5.0706024e30], dtype=np.float32), dtype=wp.float32, device=DEV)
x2 = wp.array(np.array([1.0141205e31], dtype=np.float32), dtype=wp.float32, device=DEV)
o3 = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(item2_add, dim=1, inputs=[x0, x1, x2, o3], device=DEV)
wp.synchronize_device(DEV)
g3 = float(o3.numpy()[0])
print(f"  got={g3!r}  expected={FLT_MAX!r}")
print(f"  VERDICT: {'BUG CONFIRMED (spurious inf)' if np.isinf(g3) else 'not reproduced'}")

print("\n=== I5: signed zero (bit patterns) ===")
o5 = wp.zeros(4, dtype=wp.float32, device=DEV)
wp.launch(item5_signedzero, dim=1, inputs=[o5], device=DEV)
wp.synchronize_device(DEV)
v5 = o5.numpy()
labels = ["float32(-0.0)", "(-1)*0", "1/((-1)*0)", "sqrt(-0)"]
want = [0x80000000, 0x80000000, 0xFF800000, 0x80000000]
for l, g, w in zip(labels, v5, want):
    gb = int(np.asarray(g, dtype=np.float32).view(np.uint32))
    print(f"  {l:16s} got={hex(gb):12s} want={hex(w):12s} {'MISMATCH' if gb != w else 'ok'}")

print("\n=== I6: sqrt of subnormal input ===")
xs = wp.array(np.array([1e-40], dtype=np.float32), dtype=wp.float32, device=DEV)
o6 = wp.zeros(1, dtype=wp.float32, device=DEV)
wp.launch(item6_sqrt_subnormal, dim=1, inputs=[xs, o6], device=DEV)
wp.synchronize_device(DEV)
g6 = float(o6.numpy()[0])
print(f"  input=1e-40f (stored as {np.float32(1e-40)!r})  got={g6!r}")
print(f"  VERDICT: {'BUG CONFIRMED (+inf)' if np.isinf(g6) else 'not reproduced'}")

print("\n=== Outside diff: floor/int/mod ===")
xf = wp.array(np.array([2.0**30], dtype=np.float32), dtype=wp.float32, device=DEV)
o7 = wp.zeros(4, dtype=wp.float32, device=DEV)
wp.launch(outside_diff, dim=1, inputs=[xf, o7], device=DEV)
wp.synchronize_device(DEV)
v7 = o7.numpy()
print(f"  floor(2^30-0.25) = {v7[0]!r}  want 1073741823.0  {'BUG' if v7[0] != 1073741823.0 else 'ok'}")
print(f"  int(2^30-0.25)   = {v7[1]!r}  want 1073741823.0  {'BUG' if v7[1] != 1073741823.0 else 'ok'}")
print(f"  mod(-1,3)        = {v7[2]!r}  fmod wants -1.0    {'DIFFERS from fmod' if v7[2] != -1.0 else 'ok'}")
print(f"  mod(1,inf)       = {v7[3]!r}  fmod wants 1.0     {'BUG' if v7[3] != 1.0 else 'ok'}")
