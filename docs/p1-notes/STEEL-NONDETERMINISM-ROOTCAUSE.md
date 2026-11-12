# Steel Metal non-determinism — definitive root cause + fix path

## Root cause (verified by bisection + code inspection)
Steel scatters contact contributions onto shared per-body/per-DOF buffers using
**float `wp.atomic_add`**, e.g. in newton/_src/solvers/steel/kernels.py:
- `accumulate_contact_wrenches_from_impulses` -> `_add_point_force`:
  `wp.atomic_add(body_wrench, body, spatial_vector(...))`  (line ~2880)
- `diag_from_JY_par_art`: `wp.atomic_add(world_diag, world, c, val)`  (line ~6591)
- propagation velocity updates: `wp.atomic_add(propagation_body_qd, ...)` (~8611)

Float addition is NOT associative, so the result depends on the order threads
perform the atomic adds. On Metal's GPU that order varies run-to-run -> the
scattered sums differ in the last bits -> chaotic contact dynamics amplify it.

## Evidence chain
1. CPU: bit-identical across runs (0.0 mm), always settles (0/25 awake).
2. Metal: two identical runs diverge by 1.5+ m; awake count varies (11 vs 22).
3. Bisection (full 4-substep simulate, eager): substeps 0-81 bit-identical;
   first divergence at substep 82 (frame 20, substep 2), exactly when a new
   contact forms (48->49). Contacts going IN are identical; solver.step output
   differs -> the non-determinism is in the solve's atomic scatter, not collision.
4. Frictionless, point-friction, and patch-friction ALL non-deterministic on Metal
   -> the source is the shared normal-contact atomic scatter, not friction-specific.
5. Not the graph: eager full simulate() reproduces it. Not batching: WP_METAL_BATCH=1
   reproduces it. Not contact generation: frame-0 collide bit-identical.
6. Not precision: CPU with float32 friction moments still settles (earlier finding).

## Why Warp's deterministic mode doesn't fix it on Metal
Warp has DeterministicMode.RUN_TO_RUN / GPU_TO_GPU that lower atomics to
deterministic scatter records (warp/_src/deterministic.py). BUT the lowering is
CUDA-only: `kernel_locals` emits real det buffers only `if device == "cuda"`;
for Metal it emits `det_scatter_buf_t{nullptr,...}` (no-op), so the atomics fall
through to plain non-deterministic atomic_add. Verified: forcing RUN_TO_RUN on
all Steel modules left Metal non-deterministic (divergence frame 1, 0.012 mm).

## The fix (two options)
A. **Implement deterministic scatter for Metal in Warp** (the proper fix):
   extend warp/_src/deterministic.py kernel_locals/emit to emit the det scatter
   buffers + replay for Metal (the MSL side: det_scatter_buf_t, det_ctx, the
   record-then-reduce replay). This makes RUN_TO_RUN work on Metal and fixes
   Steel (and any atomic-scatter kernel) generally. Substantial but upstreamable.
B. **Newton-side deterministic reduction** for the specific Steel scatters:
   replace the float atomic_add scatters with a fixed-order reduction (e.g.
   per-body serial accumulation, or sort contributions by contact index then
   prefix-sum). Localized to Newton, but must be done for each scatter site and
   may cost performance.

## Recommendation
Option A is the right long-term fix (benefits all Warp Metal kernels, matches
CUDA's existing deterministic support). Option B is a faster Newton-only workaround.
Either restores run-to-run determinism; whether the towers then SLEEP like CPU
depends on whether the deterministic Metal trajectory matches CPU's basin of
attraction (likely close, since the only difference was atomic ordering).

## Status
Root cause fully verified. Softfloat binary64 is confirmed irrelevant (the issue
is atomic ordering, not precision). Fix not yet implemented.
