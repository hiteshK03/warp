# Steel jitter — CORRECTED root cause (supersedes STEEL-JITTER-INVESTIGATION.md)

## The softfloat question is moot
The earlier hypothesis (df64 precision in friction-patch moments causes the jitter)
was DISPROVEN by experiment:

1. **CPU with float32 friction moments still settles perfectly** (0/25 awake, zero
   velocity). Degrading the moment precision does NOT break CPU settling. So
   friction-moment precision is NOT the lever. Softfloat binary64 would not help.

## The real issue: Metal is NON-DETERMINISTIC run-to-run
- CPU: bit-identical across runs (0.0 mm), always settles (0/25 awake).
- Metal: TWO identical runs differ by 1.5+ METERS, with variable awake counts
  (11/25 vs 22/25). Metal does not even reproduce itself.

## Bisection of the non-determinism
- Initial collide (frame 0): bit-identical across Metal runs (28 contacts, 0.0 mm).
- Step 1: bit-identical across Metal runs.
- Steps 1-20: bit-identical.
- **Step 21: first divergence (0.045 mm)**, then chaotic amplification.

So contact generation and the first 20 solve steps are deterministic. The
non-determinism enters at step 21, when the settling contact configuration changes
(a contact forms/breaks or a body wakes), triggering a kernel whose parallel
execution order varies on Metal's GPU.

## What it is NOT
- Not df64 precision (CPU float32 moments settle).
- Not the core contact/normal solve (frictionless: both devices settle perfectly).
- Not contact generation at frame 0 (bit-identical).
- Not the GS solve (one thread per world, serial; demo has 1 world).
- Not fixed by CollisionPipeline(deterministic=True): Metal still non-deterministic
  (1968 mm between runs). The contact sort does not cover the step-21 source.

## Remaining suspect
A parallel kernel in the Steel step path (friction-patch _build floods one thread
per body-pair; impulse/contact accumulation) whose reduction or write order varies
with Metal thread scheduling once the contact set changes at step 21. The two
atomics in friction_patches are _list_flood_pairs (CUDA-only) and atomic_max
(order-independent), so the source is elsewhere — likely a parallel sum/reduction
in the contact or friction impulse path, or the broad-phase pair ordering.

## Conclusion
The bouncing is chaotic amplification of run-to-run NON-DETERMINISM in Metal's
parallel contact/friction kernels, not a precision deficit. The fix is deterministic
ordering/reductions in the Metal step path, NOT softfloat binary64. CPU settles
because it is deterministic; Metal's varying thread order perturbs the chaotic
contact dynamics enough to keep bodies awake.

## Next step (if pursued)
Bisect step 21 to the specific kernel: snapshot contacts + per-body impulses before
and after each sub-kernel at step 21 across two runs, find the first non-deterministic
output, and make that kernel's ordering/reduction deterministic on Metal.
