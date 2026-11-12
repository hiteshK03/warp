# Steel Metal jitter — final root-cause chain (verified)

The investigation pivoted twice as hypotheses were tested and refuted. Final chain:

## Layer 1: run-to-run non-determinism (SOLVED, no Warp port needed)
- Default Steel on Metal is non-deterministic run-to-run (1.5m divergence).
- Cause: integer row-slot reservation via ROW_SLOTS_ATOMIC (racy per-world counter)
  + non-deterministic contact order.
- FIX (already in Newton): SolverSteel(deterministic=RUN_TO_RUN) reserves rows in fixed
  order; CollisionPipeline(deterministic=True) sorts contacts. Verified: Metal becomes
  BIT-IDENTICAL run-to-run (0.0mm between two runs). NO Warp scatter port required.
- This refuted the earlier "float atomic scatter" hypothesis as the dominant source.
  (Claude review doubt #9 predicted exactly this: audit integer atomics, not just float.)

## Layer 2: CPU/Metal trajectory parity (NOT solved by determinism)
- Even with R2R, Metal differs from CPU by 1.67m at frame 599; Metal keeps 6/25 bodies
  awake (ang ~2-4 rad/s) indefinitely (still 6/25 at frame 2000); CPU settles 0/25.
- Divergence seeds at frame 1 (~1.7mm), present even FRICTIONLESS -> not df64/friction.

## Layer 3: the actual bug — Metal box-box contact manifold is degenerate (ROOT CAUSE)
- Initial collide, identical state: CPU generates 88 contacts, Metal 28.
- Same 22 shape pairs on both. CPU: EVERY pair has 4 contact points (full face manifold).
  Metal: most pairs have only 1 contact point (2 pairs have 4, twenty pairs have 1).
- Example pair (0,1): CPU = 4 face corners; Metal = 1 point.
- A single contact point cannot resist torque -> boxes rock/spin -> persistent motion ->
  never sleep. THIS is why Metal towers bounce and don't settle.
- It is a Newton collision-manifold generation bug on Metal (box-box clipping/reduction
  produces 1 point instead of 4), NOT a determinism or precision issue.

## What this means for the original "part A" (Warp Metal deterministic scatter)
- The Warp det-scatter port does NOT fix the Steel jitter. Run-to-run determinism is
  already provided by Steel R2R; the jitter is a degenerate contact manifold.
- The partial Warp implementation (deterministic_metal.h + codegen hooks) is a legitimate
  standalone backend feature (Metal lacks CUDA's deterministic atomics) but is orthogonal
  to the user-visible problem. It also has unresolved issues per Claude review (MSL address
  spaces B1, os_log B2, cache key B6, graph capture doubt #8) and was never compiled/tested.

## Recommended next step
Debug Newton's box-box contact manifold on Metal: find why narrow_phase_manifold_kernel /
contact clipping yields 1 point instead of 4 for face-face box contacts. Likely a tile/
shared-memory or reduction issue in the Metal path of the manifold generator. Fixing that
should let Metal towers settle like CPU (with R2R for determinism).

## Status
Root cause verified by execution. The Steel jitter is a Metal collision-manifold bug.
Determinism is separately solved by Steel R2R. Warp det-scatter port reassessed as
orthogonal standalone feature, paused pending decision.
