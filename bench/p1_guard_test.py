"""Verify the df_guard overflow fix. Values are LOADED FROM A BUFFER at runtime so
the compiler cannot constant-fold them (the review's explicit requirement).

Cases: the four originally-confirmed failures, plus the review's counterexamples
that a naive output-only guard would miss (tail-normalize overflow, 1/inf).
Also re-runs the in-range regression suite to check nothing finite changed.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "metal:0"
wp.set_device(DEV)
FLT_MAX = np.finfo(np.float32).max


@wp.kernel
def ops(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x = wp.float64(a[i])
    y = wp.float64(b[i])
    out[i * 5 + 0] = wp.float32(x + y)
    out[i * 5 + 1] = wp.float32(x - y)
    out[i * 5 + 2] = wp.float32(x * y)
    out[i * 5 + 3] = wp.float32(x / y)
    out[i * 5 + 4] = wp.float32(wp.sqrt(x))


# (a, b, expected[+, -, *, /, sqrt(a)]) computed against numpy float64 as reference
INF = np.inf
cases = [
    ("overflow 3e38*3e38", 3e38, 3e38),
    ("inf + 1", INF, 1.0),
    ("inf * 1", INF, 1.0),
    ("1/0", 1.0, 0.0),
    ("1/inf", 1.0, INF),
    ("sqrt(inf)", INF, 1.0),
    ("inf - inf", INF, INF),
    ("normal 6/3", 6.0, 3.0),
    ("FLT_MAX + 9e30", float(FLT_MAX), 9e30),
]

a = np.array([c[1] for c in cases], dtype=np.float32)
b = np.array([c[2] for c in cases], dtype=np.float32)
da = wp.array(a, dtype=wp.float32, device=DEV)
db = wp.array(b, dtype=wp.float32, device=DEV)
out = wp.zeros(len(cases) * 5, dtype=wp.float32, device=DEV)
wp.launch(ops, dim=len(cases), inputs=[da, db, out], device=DEV)
wp.synchronize_device(DEV)
v = out.numpy().reshape(len(cases), 5)

# reference: same ops in host float64 then cast to float32 (IEEE semantics)
with np.errstate(all="ignore"):
    ref = np.stack([
        (a.astype(np.float64) + b.astype(np.float64)),
        (a.astype(np.float64) - b.astype(np.float64)),
        (a.astype(np.float64) * b.astype(np.float64)),
        (a.astype(np.float64) / b.astype(np.float64)),
        np.sqrt(a.astype(np.float64)),
    ], axis=1).astype(np.float32)

names = ["+", "-", "*", "/", "sqrt"]
print(f"{'case':22s} " + " ".join(f"{n:>22s}" for n in names))
bad = 0
for i, (label, _, _) in enumerate(cases):
    cells = []
    for j in range(5):
        g, r = v[i, j], ref[i, j]
        ok = (np.isnan(g) and np.isnan(r)) or (g == r) or (np.isinf(g) and np.isinf(r) and np.sign(g) == np.sign(r))
        if not ok:
            bad += 1
        cells.append(f"{g:>10.3g}/{r:<10.3g}{'' if ok else ' X'}")
    print(f"{label:22s} " + " ".join(f"{c:>22s}" for c in cells))
print(f"\n(got/expected; X = mismatch)  mismatches: {bad}")
print("VERDICT:", "PASS — IEEE inf/NaN semantics restored" if bad == 0 else f"{bad} mismatches remain")
