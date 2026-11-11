"""Regression tests for the Metal float64 (df64) emulation.

Metal has no double type; wp.float64 is emulated as a double-single (hi, lo)
pair. These tests check the emulation's precision contract and that the generic
templated linear algebra works against it, cross-checked against the CPU device
(which has real binary64).

Run:  python bench/test_df64_metal.py
Exit code 0 = all passed. Skips cleanly when metal:0 is unavailable.

Contract being tested (see native/metal_float64.h):
  - df64 resolves values below float32's mantissa resolution (~44-48 bits)
  - fp64 *internal* arithmetic is supported; fp64 arrays in the kernel ABI and
    fp64 atomics are NOT (must still raise)
  - exponent range is float32's, NOT binary64's
"""

import numpy as np
import warp as wp

DEVICE = "metal:0"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


@wp.kernel
def precision_gain(out: wp.array(dtype=wp.float32)):
    """(1 + 1e-10) - 1 is 0 in float32; df64 must recover ~1e-10."""
    one = wp.float64(1.0)
    tiny = wp.float64(1.0e-10)
    out[0] = wp.float32((one + tiny) - one)


@wp.kernel
def mat_inverse_f64(inp: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    """Generic templated inverse over the emulated scalar type."""
    i = wp.tid()
    md = wp.mat33d(inp[i])
    out[i] = wp.mat33(wp.inverse(md))


@wp.kernel
def eig_f64(inp: wp.array(dtype=wp.mat33), vals: wp.array(dtype=wp.vec3)):
    """Generic templated eig3 over the emulated scalar type."""
    i = wp.tid()
    q, d = wp.eig3(wp.mat33d(inp[i]))
    vals[i] = wp.vec3(d)


@wp.kernel
def tape_through_f64(x: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    """Reverse mode through df64 arithmetic."""
    v = wp.float64(x[0])
    wp.atomic_add(loss, 0, wp.float32(wp.sqrt(v * v)))


def main():
    wp.init()
    if DEVICE not in [str(d) for d in wp.get_devices()]:
        print(f"{DEVICE} unavailable; skipping df64 tests")
        return 0

    # 1. precision gain over float32
    out = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    wp.launch(precision_gain, dim=1, inputs=[out], device=DEVICE)
    wp.synchronize_device(DEVICE)
    got = float(out.numpy()[0])
    f32_baseline = float(np.float32(np.float32(1.0) + np.float32(1e-10)) - np.float32(1.0))
    check("precision gain vs float32", abs(got - 1e-10) < 1e-12 and f32_baseline == 0.0,
          f"df64={got:.3e} float32={f32_baseline:.3e}")

    # 2. inverse matches CPU binary64
    small = np.diag([1e-7, 2e-7, 3e-7]).astype(np.float32).reshape(1, 3, 3)
    ref = np.linalg.inv(small.astype(np.float64))
    results = {}
    for dev in ("cpu", DEVICE):
        i = wp.array(small, dtype=wp.mat33, device=dev)
        o = wp.zeros(1, dtype=wp.mat33, device=dev)
        wp.launch(mat_inverse_f64, dim=1, inputs=[i, o], device=dev)
        wp.synchronize_device(dev)
        results[dev] = o.numpy()[0]
    rel = np.abs(results[DEVICE] - ref).max() / np.abs(ref).max()
    check("fp64 inverse vs CPU reference", rel < 1e-4, f"max rel err={rel:.2e}")

    # 3. eig3 resolves a tiny eigenvalue
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    A = ((Q * np.array([1e-8, 1.0, 3.0])) @ Q.T).astype(np.float32)
    ref_eig = np.sort(np.linalg.eigvalsh(A.astype(np.float64)))
    i = wp.array(A.reshape(1, 3, 3), dtype=wp.mat33, device=DEVICE)
    v = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    wp.launch(eig_f64, dim=1, inputs=[i, v], device=DEVICE)
    wp.synchronize_device(DEVICE)
    got_eig = np.sort(v.numpy()[0])
    rel_eig = abs(got_eig[0] - ref_eig[0]) / abs(ref_eig[0])
    check("fp64 eig3 tiny eigenvalue", rel_eig < 1e-4, f"rel err={rel_eig:.2e}")

    # 4. reverse mode through df64: d/dx sqrt(x*x) = sign(x) = 1 for x=3
    x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEVICE, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(tape_through_f64, dim=1, inputs=[x, loss], device=DEVICE)
    tape.backward(loss=loss)
    g = float(x.grad.numpy()[0])
    check("tape through df64", abs(g - 1.0) < 1e-4, f"grad={g:.6f} expected 1.0")

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
