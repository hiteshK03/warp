# Plan review: is df64 justified on Newton's inertia path?

**Verdict:** M-B and M-C would mostly measure float32 rounding and simulation noise, so they can't answer the question. The decision rule also has a loophole: it can come out "keep df64" on a meaningless signal. Fix the design before running anything.

## 1. The decision rule has a hole and the wrong criterion

- **The branches don't cover every outcome.** "M-B/M-C show benefit ⇒ keep" reads as OR. If M-B shows 1-ulp bitwise differences and M-C shows nothing, the rule says keep df64. That is the most likely outcome.
- **Bitwise difference is almost guaranteed, so it decides nothing.** A body built from several offset or rotated shapes has a full 3×3 inertia. Example: an ant leg capsule with r≈0.04 and L≈0.3 has a principal-axis condition number around 1e2. A float32 adjugate inverse then has relative error around cond·eps ≈ 1e-5, so (b) ≠ (c) on nearly every non-diagonal body.
- **Fix:** replace "bitwise" with ulp distance from a correctly rounded reference: `ulp((b), round32(ref64))` against `ulp((c), round32(ref64))`. M-B should only be diagnostic. M-C alone should be allowed to justify keeping df64, with a pre-stated threshold (see point 4).

## 2. The question asked is not the one that decides it

The real question is whether this inverse ever has to run on the device.

- **If it runs once on the host at `finalize()`:** a CPU float64 inverse rounded to float32 is always at least as accurate as df64, keeps correct IEEE semantics, and has no guard regression. df64 loses whatever M-B and M-C show.
- **df64 only wins if all three hold:**
  1. fp64-quality inverses measurably matter downstream;
  2. the inverse must run on the device repeatedly (for example inertia randomization per RL episode, graph-captured resets, or gradients through inertia);
  3. softfloat binary64 is too slow for that workload.

The plan measures (1) only. Add (2): grep for every writer of `body_inv_inertia`. Add (3) only if (2) is true.

## 3. Check the producer and consumer before measuring

- **Producer:** find out where Newton actually computes `body_inv_inertia`. Upstream warp.sim did it on the host in Python/NumPy. If Newton still does, there is no device kernel to justify and the whole question goes away.
- **Consumer:** I believe Featherstone does not read `body_inv_inertia`. It builds spatial inertia from `body_inertia` and factors the joint-space mass matrix. If so, M-C with Featherstone is null by construction. Use a solver that reads the inverse (XPBD or SemiImplicit `integrate_bodies`), and confirm it with grep before trusting a null result.

## 4. M-C is dominated by chaos and noise

- **Contacts make a 200-step quadruped rollout chaotic.** Any 1-ulp perturbation diverges. Both runs will differ from the reference by Lyapunov growth, not by inertia quality.
- **Metal float atomics in contact accumulation are likely non-deterministic.** Run (c) twice. If body_q differs between the two runs, that difference is your noise floor.
- **"float64-inertia reference" is not a separate thing.** The store is float32, so the best possible stored value is `round32(inv64(I32))`. That is just the ideal (b). All three runs feed float32 matrices that differ by ulps into float32 dynamics.
- **Required control:** an ensemble of runs with the inverse randomly perturbed by ±1 ulp per element, to measure the trajectory spread. df64 counts as a benefit only if its error beats float32 by more than k× that spread. Set k in advance; I'd use 3.
- **Add a contact-free test:** a torque-free asymmetric top spinning about a stable axis, with a non-diagonal body-frame inertia. Inertia error shows up as precession phase drift that grows linearly with time. This is the only setting where a one-time ulp difference is plausibly visible above the per-step float32 noise (each step's R·I⁻¹·Rᵀ adds about eps).

## 5. Input-rounding floor is missing

You fix the stored float32 inertia as truth. That isolates the kernel, but it misses the physical point. If the inertia was float64 upstream (from geometry and density) and then rounded to float32, the input already carries cond·eps of error. If error(c) ≤ ‖inv64(I64) − inv64(I32)‖, the float32 kernel is already within the input noise and df64 buys nothing physical. Report the condition number and this floor for each body.

## 6. Metal-specific confounds

- **(c) on Metal is not "today's CUDA/CPU behaviour."** Metal's fast-math float division is specified to about 2.5 ulp, not correctly rounded, and the compiler may contract to FMA. Run (c) on the CPU device as well. Otherwise you are measuring Metal's division, not float32 intermediates.
- **Fast-math can silently break df64's error-free transforms.** The compiler may simplify `(a+b)-a` and cancel TwoSum. Sanity check: df64 1/3 should give hi=0.33333334f and lo≈−9.9e-9. If lo == 0, the EFTs were optimised away and every df64 number in the run is invalid.
- **Define (b) as what the kernel actually writes.** That is probably `hi`, which equals the correctly rounded value only if (hi, lo) is normalized, |lo| ≤ ½ulp(hi). Report it separately from the ideal `round32(double(hi)+double(lo))` so normalization bugs show up.

## 7. Sample coverage

- **The population is about 35 bodies**, so "essentially all" means little. Report each body with its condition number, not a pooled distribution. Many MJCF/URDF bodies are diagonal (`diaginertia`), where the inverse is just elementwise division and df64 equals correct rounding. Split diagonal from full-tensor bodies.
- **Include zero-mass and static bodies.** Input: zero inertia. Expected: inverse = 0, Newton's special case. Check the df64 path doesn't produce NaN or inf before that special case, given the guard history.
- **The FLT_MAX guard regression is irrelevant at ~1e-3 magnitudes.** But tiny bodies (~1e-6) make the determinant tiny. det ≈ I³ ≈ 1e-18 is still normal float32, so it is probably fine. Log the minimum determinant anyway.

## Revised decision rule

1. If the inverse is computed on the host or one-shot at `finalize()`, use CPU float64 rounded to float32. Done; df64 is not used here.
2. Otherwise, keep df64 only if the M-C benefit exceeds k× the ±1-ulp ensemble spread on the contact-free top, with a solver confirmed to read the inverse, and softfloat is shown too slow for the per-episode workload.
3. In every other case, use float32 on the device, or softfloat if correctness matters.