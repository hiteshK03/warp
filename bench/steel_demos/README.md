# Steel demos on Metal

The skild-ai/steel repo has NO runnable code (README + license + video only). The
runnable Steel examples live in Newton: newton/examples/steel/. Both hard-require
CUDA and use the GL viewer; these are Metal-port copies (only the CUDA device guard
relaxed to also accept Metal) run headless with the null viewer.

## Verified on Apple M5 Pro (warp-metal 1.17.0.4, Newton 1.9.0.dev0)

### steel_sleeping (run.py)
Four box towers fall, settle, and sleep; a thrown block wakes one tower.
Result: awake bodies 25 -> 24 -> 11 -> 12 over 400 frames (sleeping + wake works),
no NaN. Run: WARP_CACHE_PATH=/tmp/x python run.py

### steel_determinism_arms (run_determinism.py)
Two Franka arms sweep boxes; default vs deterministic Steel copies.
Result: 200 frames, no NaN; max divergence default=9.04mm, deterministic=0.79mm
(deterministic mode ~11x tighter). Needs GitPython, trimesh, pycollada for the
Franka asset download. Run: WARP_CACHE_PATH=/tmp/x python run_determinism.py

## Newton-source fix required
newton/_src/geometry/contact_reduction.py float_flip: reinterpret_cast<uint32_t&>
is invalid MSL; patched to as_type<uint32_t> under __METAL_VERSION__ (Newton fork
branch metal-steel-demos, commit c4a49084).

## Caveats
- df64 opt-in is required for Steel's fp64 per-step math (set on the solver module).
- Deterministic divergence is ~0.79mm not exactly 0 (Metal float atomics differ
  from CUDA); the determinism MODE works but bitwise-run-to-run is not guaranteed
  on Metal the way it is on CUDA.
- These are headless ports; the GL/RTX visual viewers are not exercised.
