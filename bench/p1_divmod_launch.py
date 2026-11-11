"""floordiv/mod consistency + launch-bounds tests (review Q2/Q4).
Run with a FRESH cache dir passed as argv[1] (G1: never reuse a warm cache).
Tests the identity a == (a//b)*b + (a%b) for negative operands, and 4D launch.
"""
import sys
import numpy as np
import warp as wp

cache = sys.argv[1]
wp.config.kernel_cache_dir = cache
wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def divmod_k(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
             fd: wp.array(dtype=wp.int32), md: wp.array(dtype=wp.int32)):
    i = wp.tid()
    fd[i] = a[i] // b[i]
    md[i] = a[i] % b[i]


vals_a, vals_b = [], []
for a in range(-7, 8):
    for b in range(-3, 4):
        if b != 0:
            vals_a.append(a)
            vals_b.append(b)
A = np.array(vals_a, dtype=np.int32)
B = np.array(vals_b, dtype=np.int32)
da = wp.array(A, dtype=wp.int32, device=DEV)
db = wp.array(B, dtype=wp.int32, device=DEV)
fd = wp.zeros(len(A), dtype=wp.int32, device=DEV)
md = wp.zeros(len(A), dtype=wp.int32, device=DEV)
wp.launch(divmod_k, dim=len(A), inputs=[da, db, fd, md], device=DEV)
wp.synchronize_device(DEV)
FD, MD = fd.numpy(), md.numpy()

# numpy floor division / mod reference
ref_fd = A // B
ref_md = A % B
fd_match = np.array_equal(FD, ref_fd)
md_match = np.array_equal(MD, ref_md)
identity = (FD * B + MD) == A
print(f"floordiv matches numpy floor: {fd_match}")
print(f"mod matches numpy floor:      {md_match}")
print(f"identity a==(a//b)*b+(a%b):   {identity.all()}  ({identity.sum()}/{len(A)})")
bad = np.where(~identity)[0]
for i in bad[:6]:
    print(f"  BROKEN a={A[i]} b={B[i]}: fd={FD[i]} md={MD[i]} -> {FD[i]*B[i]+MD[i]} != {A[i]}")
np.save(f"{cache}/divmod.npy", np.stack([A, B, FD, MD]))


# 4D launch bounds (Q2): each thread writes a linear index
@wp.kernel
def launch4d(out: wp.array(dtype=wp.int32, ndim=4)):
    i, j, k, l = wp.tid()
    out[i, j, k, l] = ((i * 3 + j) * 5 + k) * 7 + l


out = wp.zeros((2, 3, 5, 7), dtype=wp.int32, device=DEV)
wp.launch(launch4d, dim=(2, 3, 5, 7), inputs=[out], device=DEV)
wp.synchronize_device(DEV)
ok4d = np.array_equal(out.numpy().flatten(), np.arange(210))
print(f"4D launch (2,3,5,7) == arange(210): {ok4d}")

# 1D non-multiple size with sentinel (Q2)
@wp.kernel
def fill1d(out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = i + 1


N = 1000003
buf = np.full(N + 1, -999, dtype=np.int32)
d = wp.array(buf, dtype=wp.int32, device=DEV)
wp.launch(fill1d, dim=N, inputs=[d], device=DEV)
wp.synchronize_device(DEV)
res = d.numpy()
sentinel_ok = res[N] == -999
fill_ok = np.array_equal(res[:N], np.arange(1, N + 1))
print(f"1D dim={N}: fill correct={fill_ok}, sentinel untouched={sentinel_ok}")
print("ALL_OK" if (fd_match and md_match and identity.all() and ok4d and sentinel_ok and fill_ok) else "SOME_FAILED")
