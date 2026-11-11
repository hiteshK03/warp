# Decision measurement plan: is df64 justified on Newton's inertia path at all?

## The question

Reviewer's closing challenge, still unanswered: Newton stores inverse inertia as
float32 `mat33`. If the gain from df64 intermediates is discarded at that store,
df64 buys nothing downstream on this path — and given it has now produced a
confirmed wrong-answer regression (FLT_MAX/1.9999998 -> -inf), the cheaper and
*correct* alternatives (softfloat binary64, or running the fp64 setup kernels on
the CPU device) would be strictly better for a one-time finalize() kernel.

## What I will measure

### M-A: Newton's actual inertia regime, with the hi/lo metric
Sample REAL inertias from Newton models (quadruped.urdf, nv_ant.xml,
nv_humanoid.xml) rather than synthetic scales. For each body's inertia tensor:
- inverse via df64 intermediates, emitted as hi/lo words
- inverse via float32 intermediates (control), same plumbing
- reference: host float64 inverse
Report: distribution of |inertia| magnitudes, and rel err of
  (a) df64 hi+lo rebuilt in float64   -> the theoretical gain
  (b) df64 rounded to float32 (what Newton actually stores)
  (c) float32 intermediates rounded to float32 (today's CUDA/CPU behaviour)
If (b) == (c) bitwise for essentially all bodies, df64 is pointless HERE.

### M-B: does the store discard it?
Count, over all real bodies, how many have (b) != (c) bitwise, and the max
relative difference. This is the decisive number.

### M-C: downstream effect over N steps
Newton rollout (Featherstone, quadruped, N=200 steps) with df64 vs float32
inverses, comparing body_q trajectories against a CPU float64-inertia reference.
Single-inverse residuals are not sufficient evidence of downstream value.

## Decision rule (stated in advance, to avoid post-hoc rationalisation)

- If M-B shows no bitwise difference on real bodies AND M-C shows no trajectory
  improvement => df64 is NOT justified for Newton's inertia path. Recommend
  softfloat binary64 or CPU offload for correctness-critical setup, and demote
  df64 to an explicit opt-in for hot paths (Steel per-step), pending its own
  justification.
- If M-B/M-C show a real benefit => keep df64 for this path, but only after the
  confirmed guard regression and signed-zero/mod defects are fixed.

## Known confounds to control

- The df64 guard regression is near FLT_MAX; real inertias are ~1e-3, far away, so
  it should not contaminate this measurement. Verify the magnitudes.
- float32 output rounding (~6e-8) is the ceiling for (b) and (c) BY DESIGN — that
  is the point: it is what Newton actually stores.
- Must not compare against a reference computed from the same float32 inputs in a
  way that hides input rounding; use the stored float32 inertia as the input for
  all three paths and a float64 inverse of THAT as reference.
