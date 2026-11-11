"""P0a fixedarray ATOMIC test (review Q5.2): wp.atomic_add on a fixedarray element,
forward and Tape backward. The cleanup dropped 1.18 slot-atomic helpers, so this
confirms the P0a fixedarray atomic overloads don't depend on them.
Fresh cache via argv[1].
"""
import sys
import numpy as np
import warp as wp

wp.config.kernel_cache_dir = sys.argv[1]
wp.init()
DEV = "metal:0"
wp.set_device(DEV)


@wp.kernel
def fa_atomic_fwd(out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    buf = wp.zeros(shape=4, dtype=wp.float32)
    buf[0] = 1.0
    # atomic_add on a fixedarray element (thread-local; plain RMW)
    wp.atomic_add(buf, 0, 2.0)
    wp.atomic_add(buf, 1, float(i + 1))
    out[i] = buf[0] + buf[1]


@wp.kernel
def fa_atomic_tape(x: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    buf = wp.zeros(shape=4, dtype=wp.float32)
    wp.atomic_add(buf, 0, x[0] * x[0])
    wp.atomic_add(loss, 0, buf[0])


o = wp.zeros(4, dtype=wp.float32, device=DEV)
wp.launch(fa_atomic_fwd, dim=4, inputs=[o], device=DEV)
wp.synchronize_device(DEV)
v = o.numpy()
expect = np.array([3.0 + (i + 1) for i in range(4)])
fwd_ok = np.allclose(v, expect)
print(f"[atomic forward] {v} expect {expect} -> {'PASS' if fwd_ok else 'FAIL'}")

x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEV, requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=True)
tape = wp.Tape()
with tape:
    wp.launch(fa_atomic_tape, dim=1, inputs=[x, loss], device=DEV)
tape.backward(loss=loss)
g = float(x.grad.numpy()[0])
bwd_ok = abs(g - 6.0) < 1e-4
print(f"[atomic tape backward] grad={g} expect 6.0 -> {'PASS' if bwd_ok else 'FAIL'}")
print("ALL_OK" if fwd_ok and bwd_ok else "SOME_FAILED")
