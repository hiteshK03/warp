# Metal backend roadmap (fork work plan)

Goal: make Newton physics (incl. Skild's SolverSteel, Newton 1.8) fully usable on Apple
Silicon GPUs via the warp-metal overlay. Forked from innate-inc/warp (main = NVIDIA main +
Metal backend); companion overlay repo: hiteshK03/warp-metal.

## Verified baseline (M5 Pro, macOS 27, Python 3.14, warp-metal 1.17.0.3)

Newton 1.7 batched-worlds sweep (nv_ant / nv_humanoid MJCF, dt=0.01, env_steps/s):

| solver       | metal w=64  | metal w=1024 | cpu w=1024 | metal speedup @1024 |
|--------------|-------------|--------------|------------|---------------------|
| semiimplicit | 147K        | 2.35M        | 143K       | 16.4x               |
| xpbd         | 96K         | 1.33M        | 53K        | 25.3x               |
| mujoco-warp  | 19K         | 247K         | 63K        | 3.9x                |
| kamino       | 15K         | 40K          | 4.7K       | 8.6x                |
| featherstone | 8K          | 70K          | 17K        | 4.1x                |

Working on Metal today: kernels, tapes/backward, graph capture, MJCF-robot sims at scale.

## Work items (priority order)

### P0 — fixedarray_t on Metal (issue #1)
`warp/native/array.h:313` guards out fixedarray_t (`__METAL_VERSION__`, no inheritance in
MSL). Blocks `narrow_phase_gjk_mpr_*` -> all URDF robots with cylinder/capsule/mesh geoms
(cartpole, quadruped verified failing on every solver). Deliverable: standalone Metal
fixedarray_t matching codegen ABI; Newton URDF sweep green on metal:0.

### P1 — float32 fallback for float64 scalar builtins (issue #2)
`_metal_unsupported_reason()` (warp/_src/context.py:3606) refuses kernels touching float64.
Newton 1.8 moved inertia math to float64 (`_eig_inertia_0`, `_inverse_inertia_wp_0`), which
blocks ALL Newton 1.8 solvers on Metal, including SolverSteel (Skild AI, release-1.8).
Deliverable: Metal float32 variants of eig3/inv3 builtins + codegen mapping; SolverSteel
ant benchmark runs on metal:0; numerics within 1e-4 rel of CPU float64.

### P2 — Newton-on-Metal CI harness
Port of the benchmark suite used for the baseline above (sweep.py: robots x solvers x
world-counts x device, JSONL results, per-config error capture). Run against fork overlay
after every change; catches regressions in both directions (warp upstream merges, Newton
version bumps). Repo: keep alongside (bench/ dir) or separate repo — decide after P0.

### P3 — performance pass
mujoco-warp on Metal is launch-bound at small batch (0.8x CPU @ w=64). Investigate graph
capture for solver step loops (overlay supports capture incl. capture_if/capture_while);
Steel's CUDA-graph design is the reference. Target: >=1x CPU at w=64 for all working solvers.

## Repro environment
- Overlay wheel: innate-inc/warp-metal release v1.17.0.3 + `warp-lang==1.17.0`
- Newton 1.7 (PyPI) for P0; Newton `release-1.8` (git) for P1 (needs
  `finalize(skip_validation_inertia=True)` until P1 lands)
- mujoco/mujoco-warp pinned `~=3.15` for Newton 1.7 SolverMuJoCo
- Benchmarks: ~/dev/mujoco-test/{sweep.py,test_steel.py,results_sweep.jsonl}
