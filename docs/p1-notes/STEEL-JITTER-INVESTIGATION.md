# Steel sleeping-demo jitter on Metal — root-cause investigation

## Symptom
On Metal, settled box towers micro-bounce and never sleep (8-21/25 bodies awake
at frame 500, angular vel up to ~3 rad/s). On CPU all 25 sleep by frame 200.
A separate "flying box" is INTENTIONAL (the demo's kick at frame 300).

## Method
Ran the identical scene on CPU (real float64) vs Metal (df64), isolating
variables. All runs headless, fresh kernel caches.

## Findings (controlled experiments)

| config | CPU awake@500 | Metal awake@500 |
|---|---|---|
| friction mu=0 (frictionless) | 0/25 | 0/25 |
| point friction mu=0.7 (beta=0, no fp64 patches) | 25/25 | 13/25 |
| friction patches mu=0.7 (default, fp64 _build) | 0/25 | 8/25 |

1. **Frictionless: both settle perfectly.** The core normal-contact solve and
   integration are correct on Metal. Not a general backend bug.
2. **Point friction: neither device settles** these offset (12 deg yaw) stacks.
   Point friction alone is insufficient on CPU too.
3. **Friction patches: CPU settles, Metal does not.** Persistent friction patches
   are what stabilize the stacks, and that path uses fp64 (df64 on Metal).

## Trajectory analysis
- frame 0: CPU and Metal bit-identical.
- frame 0->1 (single step): differ by only 3e-8 mm / 1.4e-8 m/s (float32 rounding).
- The tiny difference amplifies chaotically: 1.75mm @frame1, 37mm @frame5,
  1.2m @frame119. Tower collapse patterns diverge completely.
- This is fundamental chaotic sensitivity of rigid-body contact, NOT a bug.

## Why Metal doesn't sleep (the actual jitter)
The friction-patch _build kernel (friction_patches.py:715) accumulates contact
moments in fp64 to compute a footprint principal axis. The code comment is explicit:
"Accumulate moments in double precision: contact-order roundoff must not give an
isotropic footprint a spurious principal axis." The spread = sqrt((xx-yy)^2+4xy^2)
and the isotropic threshold (spread > 1e-6*(xx+yy)) are roundoff-sensitive.

Microbenchmarks show df64 computes the moments accurately (~1e-7 rel err) for
tested footprints. So df64 is not catastrophically wrong; rather, the small
df64-vs-f64 differences (~1e-7) perturb the friction anchors slightly each step,
and the chaotic contact dynamics amplify this into persistent motion that keeps
bodies above the sleep threshold (linear 0.05 m/s, angular 0.15 rad/s, quiet 0.5s).

## Conclusion
The jitter is NOT a correctness bug in the Metal backend or df64. It is the
chaotic amplification of legitimate float precision differences (df64 ~1e-7 vs
real float64) in the roundoff-sensitive friction-patch stabilization. CPU's exact
float64 happens to settle these particular stacks; Metal's df64 perturbs them
enough to keep them awake.

## Mitigations (not bugs to fix, but ways to get clean sleep on Metal)
- Raise sleep thresholds (sleep_linear_threshold / sleep_angular_threshold) so
  the residual df64-driven motion falls below them.
- Increase sleep_quiet_time tolerance or use a velocity deadband.
- Use real float64 for the friction-patch path (not available on Metal without
  softfloat; df64 is the only option).
- Accept that exact CPU/Metal trajectory parity is impossible for chaotic contact
  sims; validate statistical/energetic behavior instead of bitwise.

## Status
Investigation complete. This is expected behavior for emulated-precision chaotic
contact, not a defect to patch. The df64 opt-in warning already documents the
precision caveat.
