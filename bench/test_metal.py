import time
import numpy as np
import warp as wp
wp.init()

@wp.kernel
def scale(a: wp.array(dtype=float), s: float):
    i = wp.tid()
    a[i] = a[i] * s

# 1. basic kernel on metal
a = wp.array(np.ones(1000), dtype=float, device="metal:0")
wp.launch(scale, dim=1000, inputs=[a, 2.0], device="metal:0")
wp.synchronize_device("metal:0")
print("basic kernel:", a.numpy()[:3], "-> expected [2. 2. 2.]")
assert np.allclose(a.numpy(), 2.0)

# 2. differentiable backward tape (scalar loss)
@wp.kernel
def square_loss(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * x[i])

x = wp.array(np.array([3.0, 4.0], dtype=np.float32), dtype=wp.float32, device="metal:0", requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, device="metal:0", requires_grad=True)
tape = wp.Tape()
with tape:
    wp.launch(square_loss, dim=2, inputs=[x, loss], device="metal:0")
tape.backward(loss=loss)
g = x.grad.numpy()
print("tape grads:", g, "-> expected [6. 8.]")
assert np.allclose(g, [6.0, 8.0])

# 3. benchmark: nbody-ish workload, metal vs cpu
@wp.kernel
def nbody_step(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
               n: int, dt: float):
    i = wp.tid()
    pi = pos[i]
    acc = wp.vec3(0.0)
    for j in range(n):
        d = pos[j] - pi
        r2 = wp.dot(d, d) + 1e-3
        acc += d / (r2 * wp.sqrt(r2))
    vel[i] += acc * dt
    pos[i] += vel[i] * dt

for dev in ["cpu", "metal:0"]:
    n = 2048
    pos = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3, device=dev)
    vel = wp.zeros(n, dtype=wp.vec3, device=dev)
    wp.launch(nbody_step, dim=n, inputs=[pos, vel, n, 0.001], device=dev)  # warmup/compile
    wp.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(50):
        wp.launch(nbody_step, dim=n, inputs=[pos, vel, n, 0.001], device=dev)
    wp.synchronize_device(dev)
    t = time.perf_counter() - t0
    print(f"nbody 2048^2 x50 steps on {dev:8s}: {t*1000:.1f} ms")
print("OK")
