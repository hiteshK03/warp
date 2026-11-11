# Decision: df64 is NOT justified on Newton's inertia path

## Measurements (bench/p1_decision.py, metal:0, 39 real bodies from 4 Newton models)

Real inertia magnitudes: min 5.68e-04, median 2.68e-03, max 3.84e+01.
Nowhere near the FLT_MAX guard-regression region, so that defect does not
contaminate these numbers.

| path | median rel err vs host float64 inverse of the same stored float32 input |
|---|---|
| (a) df64 hi+lo rebuilt in float64 | **2.26e-15** |
| (b) df64 -> float32 store (what Newton keeps) | **2.93e-08** |
| (c) float32 intermediates -> float32 | **6.21e-08** |

M-B: **34 / 39** real bodies have bitwise-different (b) vs (c); max relative
difference 1.63e-07.

So the reviewer's hypothesis "the float32 store discards df64's gain" is
**REFUTED for this path**: df64 does survive the store, giving a consistent ~2.1x
error reduction (it lands on the correctly-rounded float32 result where float32
intermediates are an extra ulp off). My expectation was wrong.

## But the review's structural checks decide it anyway

Verified by grep in newton-physics/newton @ main:

1. **Featherstone does NOT read `body_inv_inertia`** — zero references in
   `newton/_src/solvers/featherstone/`. My planned M-C (Featherstone rollout) would
   have been **null by construction**, and I would have wrongly concluded "no
   downstream effect". XPBD does read it (`body_inv_inertia_effective`).
2. **The producer is one-shot.** `compute_body_inverses_kernel` is launched from
   exactly ONE site: `builder.py:16241`, inside `finalize()`. No per-step or
   per-reset device recomputation. Solvers only read the stored value.
3. **Most inverses are computed on the HOST anyway** — `import_mjcf.py`,
   `import_urdf.py`, `import_usd_deformable_utils.py` all call `wp.inverse(...)` in
   Python at import time.

Per the review's revised decision rule #1: if the inverse is computed one-shot at
finalize(), a CPU float64 inverse rounded to float32 is always at least as accurate
as df64, keeps correct IEEE semantics, and has no guard regression. **df64 loses.**

The ~2.1x gain at the store is real but buys nothing that a host float64 inverse
wouldn't do better and more safely — and this is a one-time kernel where a 30-100x
slower softfloat binary64 would also be free in practice.

## Decision

- **Newton inertia path: do NOT rely on df64.** Correct approach is host/CPU
  float64 rounded to float32 (already what the importers do), or softfloat if a
  device path is required for correctness.
- **df64 remains useful only where fp64 must run on the device repeatedly** — e.g.
  Steel's per-step compliance/friction math. That case needs its own justification
  measurement, and must not inherit this result.
- **df64 must not be the silent default for `float64` on Metal.** Given the
  confirmed guard regression (finite/finite -> wrong-sign inf), the scale-dependent
  accuracy (rel err 1.0 at 1e-12), signed-zero loss and wrong `mod`, silently
  substituting it for IEEE float64 is indefensible. Make it opt-in, or replace the
  default with softfloat binary64.

## Not done (honest gaps)

- M-C was never run: the review showed Featherstone would make it null, and the
  one-shot producer made it moot. A contact-free asymmetric-top test with XPBD
  (which does read the inverse) is still the right experiment IF anyone wants to
  justify df64 for a repeated device path.
- The review's other design critiques (ulp-distance instead of bitwise, input
  rounding floor, CPU control for (c), diagonal vs full-tensor split, zero-inertia
  bodies, fast-math EFT sanity check) are unaddressed. They matter for a rigorous
  df64 justification, which this decision makes unnecessary for THIS path.
