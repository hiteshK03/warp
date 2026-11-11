"""Verify Claude's review claims empirically. Highest severity first.

C1 (CRITICAL): scalar float64 kernel ARGUMENTS bypass the gate -> silent garbage,
   because host packs binary64 (8 bytes) and struct{float hi,lo} is also 8 bytes.
C2: literal precision — wp.float64(0.1) is only float32-accurate.
C3: int conversion — float64(16777217) == 16777216.
C4: floor/int round hi+lo first -> floor(0.99999999907) == 1 (should be 0).
C5: inf arithmetic collapses to NaN instead of inf.
C6: == rounds hi+lo but < is lexicographic -> can disagree.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)

print("=== C1: scalar float64 kernel argument (claimed CRITICAL) ===")

@wp.kernel
def scalar_arg(s: wp.float64, out: wp.array(dtype=wp.float32)):
    out[0] = wp.float32(s)

out = wp.zeros(1, dtype=wp.float32, device=DEV)
try:
    wp.launch(scalar_arg, dim=1, inputs=[1.0, out], device=DEV)
    wp.synchronize_device(DEV)
    v = float(out.numpy()[0])
    print(f"  launched with 1.0 -> kernel sees {v}")
    print(f"  VERDICT: {'SILENT GARBAGE CONFIRMED' if abs(v - 1.0) > 1e-6 else 'ok (value correct)'}")
except Exception as e:
    print(f"  raised (good): {type(e).__name__}: {str(e)[:160]}")

print("\n=== C2/C3: literal and integer precision ===")

@wp.kernel
def lit_int(out: wp.array(dtype=wp.float32)):
    # 0.1 is not representable in binary; df64 from a float literal keeps only 24 bits
    tenth = wp.float64(0.1)
    out[0] = wp.float32(tenth * wp.float64(10.0))   # ideally 1.0 exactly
    big = wp.float64(16777217)                       # 2^24+1, needs 25 bits
    out[1] = wp.float32(big - wp.float64(16777216))  # ideally 1.0
    out[2] = wp.float32(big)

o2 = wp.zeros(3, dtype=wp.float32, device=DEV)
wp.launch(lit_int, dim=1, inputs=[o2], device=DEV)
wp.synchronize_device(DEV)
v2 = o2.numpy()
print(f"  0.1*10 = {v2[0]!r}  (ideal 1.0)")
print(f"  16777217-16777216 = {v2[1]!r}  (ideal 1.0) -> {'BUG' if v2[1] != 1.0 else 'ok'}")
print(f"  float32(16777217) = {v2[2]!r}")

print("\n=== C4: floor of a value just below an integer ===")

@wp.kernel
def floor_probe(out: wp.array(dtype=wp.float32)):
    # build 1 - 2^-30 via df64 arithmetic (no two-arg ctor available in Python)
    one = wp.float64(1.0)
    eps = wp.float64(1.0) / wp.float64(1073741824.0)  # 2^-30
    x = one - eps                                      # 0.9999999990686774
    out[0] = wp.float32(x)                             # shows the value
    out[1] = wp.float32(wp.floor(x))                   # ideal 0.0
    out[2] = wp.float32(wp.mod(x, wp.float64(1.0)))    # ideal ~0.9999999991

o3 = wp.zeros(3, dtype=wp.float32, device=DEV)
wp.launch(floor_probe, dim=1, inputs=[o3], device=DEV)
wp.synchronize_device(DEV)
v3 = o3.numpy()
print(f"  x = {v3[0]!r}")
print(f"  floor(x) = {v3[1]!r}  (ideal 0.0) -> {'BUG CONFIRMED' if v3[1] != 0.0 else 'ok'}")
print(f"  mod(x,1) = {v3[2]!r}  (ideal ~1.0, must be >=0) -> {'BUG CONFIRMED' if v3[2] < 0 else 'ok'}")

print("\n=== C5: inf arithmetic ===")

@wp.kernel
def inf_probe(out: wp.array(dtype=wp.float32)):
    big = wp.float64(3.0e38)
    inf = big * big                     # overflows float32 -> inf
    out[0] = wp.float32(inf)
    out[1] = wp.float32(inf + wp.float64(1.0))   # ideal inf
    out[2] = wp.float32(inf * wp.float64(1.0))   # ideal inf
    out[3] = wp.float32(wp.sqrt(inf))            # ideal inf

o4 = wp.zeros(4, dtype=wp.float32, device=DEV)
wp.launch(inf_probe, dim=1, inputs=[o4], device=DEV)
wp.synchronize_device(DEV)
v4 = o4.numpy()
names = ["inf", "inf+1", "inf*1", "sqrt(inf)"]
for n, val in zip(names, v4):
    bad = np.isnan(val)
    print(f"  {n:10s} = {val!r}  {'NaN (BUG)' if bad else 'ok'}")
