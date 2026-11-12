# FIXED: Steel Metal jitter — Metal fixedarray copy-semantics bug

## The fix (one line, warp/native/array.h)
The P0a Metal `fixedarray_t` had an explicit copy constructor that DEEP-copied
(`data = local storage`). CPU's `fixedarray_t` has NO copy constructor, so the
compiler-default memberwise copy copies the `data` POINTER, which aliases the
SOURCE's storage. Newton's contact-manifold code passes fixedarrays by value through
`Any`-typed `@wp.func` params and mutates them in place; that only works with alias-on-copy.

Fix: `fixedarray_t(const fixedarray_t& other) = default;` (match CPU), and copy `grad`
in `operator=` (CPU copies it; Metal previously nulled it).

## Why this caused the jitter (verified chain)
1. By-value fixedarray param on Metal was a deep copy -> in-place writes lost.
2. multicontact.add_avoid_duplicates_vec2 read arr[0]/arr[last] as (0,0) -> wrongly
   rejected the 4th face corner as a "duplicate" (distsq 0 < eps^2).
3. Box-box contact manifold degenerated to 1 point instead of 4 (CPU=4).
4. A single contact point can't resist torque -> boxes rock/spin -> never sleep.

## Verification (all by execution, fresh caches)
- fixedarray_nested_probe (by-value Any through 2 func layers): Metal reads correct (was 0,0).
- box_manifold_repro (two stacked boxes): Metal box-box pair = 4 points (was 1); total 8 (was 5);
  matches CPU exactly.
- Steel sleeping demo, R2R + det collision: Metal 0/25 awake @599 (was 6/25), zero velocity,
  BIT-IDENTICAL run-to-run (0.0 mm). Settles like CPU.
- Steel sleeping demo, DEFAULT mode (df64, no R2R): Metal 0/25 awake @599 (was 8-22/25).
  The kick wakes 8 bodies briefly (frame 450, max 0.002 m/s) then they re-sleep. Jitter gone
  even without determinism mode.
- Review probes: (a') same-frame write-then-read OK; (e) return-then-clobber OK (no dangling;
  codegen declare-then-assign uses operator= deep copy). Both match CPU.
- Regressions: P0a fixedarray suite 5/5 PASS; atomic fixedarray fwd+bwd ALL_OK;
  warp.tests.test_metal 18 tests OK (skip 1). divmod identity 90/90 (floordiv/mod-vs-numpy
  mismatch is pre-existing df64 floored-modulo, unrelated).

## Claude review
Two adversarial reviews guided this:
- METAL-DET-SCATTER-REVIEW.md: doubt #9 pivoted away from a needless Warp det-scatter port
  (Steel R2R already gives run-to-run determinism) toward the real collision-manifold bug.
- METAL-FIXEDARRAY-COPY-REVIEW.md: confirmed the alias fix, flagged the blocking precondition
  (accessors must use .data not .storage — verified they do), the dangling-pointer hazard
  (return-by-value; tested safe), and grad handling (fixed). Recommended `= default` over a
  hand-written alias ctor; implemented as recommended.

## Note on the paused Warp det-scatter work
codegen.py / deterministic.py / deterministic_metal.h contain a PARTIAL, uncompiled Metal
deterministic-scatter port (standalone backend feature, orthogonal to this fix). It is NOT
needed for the Steel jitter and is left uncommitted/paused pending a separate decision.