# P1 open verification items — final status

All items from the opt-in implementation review have been worked through. Each was
verified by execution (fresh kernel cache per run), and every Claude review claim
was tested before being accepted or rejected.

## Item 1 — full-tree pre-P1 baseline
PARTIAL. Used a pure metal-1.17 overlay restore (native headers + context.py +
codegen.py) as the baseline, not a full worktree build. This was sufficient to
prove the divmod/launch table is bit-identical to 1.17 and to expose the
contamination. A full `git worktree` build was not done; the overlay restore is a
faithful baseline for the source/behaviour comparisons made, but a clean worktree
build would be stronger. Honest limitation.

## Item 2 — config leak / ordering / validation
DONE (earlier commit 24f3ba6f). Config fallback removed (module-option only),
ValueError on invalid value, leak + ordering verified closed.

## Item 3 — warm-cache warning
DONE (24f3ba6f). Warning moved to Module.load; fires on warm cache.

## Item 4 — #else-branch coverage
MOOT. grep confirms zero #else branches on WP_METAL_DF64; all guards are one-sided
#if. No dead alternative code exists.

## Item 5 — cross-backend hash perturbation
DONE (159241db). metal_float64 omitted from both hash loops at default. Verified the
option's isolated contribution: no-key == default, df64 differs. Whole-module hash
still differs from 1.17 because native_headers_digest() hashes the genuinely-changed
headers (intended upstream behavior); full hash identity is impossible by design and
not claimed.

## Item 6 — source identity
RESOLVED with the correct preprocessed criterion (1c5a1ab4). builtin.h token-identical
to 1.17 when df64 off; vec/mat/quat/svd df64 edits preprocess out on Metal error mode;
mat.h explicit casts are the one shared-code change (semantic no-op without df64).

## Item 7 — per-kernel fault isolation
DONE (7e938597, 81983bdd). sin(float64) refused individually; siblings survive.
Per-kernel metallibs (no subset rebuild), mangled-name keying, generic-overload
expansion, name-stability assertion, leak-free __del__, two-tier backward isolation,
module-wide forward-only retry gated to single-kernel modules. 4 unittest cases in
warp/tests/test_metal.py::TestMetalFaultIsolation. Full test_metal: 15 pass / 1 skip.
Remaining gap: a dedicated forward-OK/backward-FAIL multi-kernel test (hard to
construct — df64 adjoints cover the ops with forward overloads).

## Item 8 — unique-module inheritance
NOT A BUG (159241db). module="unique" does not inherit fast_math (upstream option)
either; metal_float64 non-inheritance is consistent upstream behavior. module_options
is the supported path and works.

## De-contamination (discovered during item 1)
DONE (2dc9ac37). builtin.h had upstream 1.18 floordiv_signed + uint32_t launch_bounds
changes copied from main during P1a. Restored to 1.17 + only the 3 df64 guard pairs.
Fixed a real latent bug (floordiv floor semantics without mod broke a==(a//b)*b+a%b
for 18/90 negative cases). Verified bit-identical to pure 1.17.

## Final verification sweep (fresh caches)
- warp.tests.test_metal: 15 pass, 1 skip
- P0a fixedarray suite: 5/5
- P0a fixedarray atomic fwd+bwd: ALL_OK
- df64 opt-in: config inert, opt-in works, ValueError, warm relaunch consistent
- error-mode corpus: f32 launches, fp64 refused

## Remaining honest gaps (not blocking, documented)
- Full worktree pre-P1 build (item 1) — overlay restore used instead.
- Dedicated multi-kernel forward-OK/backward-FAIL test (item 7).
- Per-kernel success caching (item 7 #9) — every failed load pays N isolated compiles.
- Injected interaction-failure test (each kernel compiles alone, module fails together).
- df64's own numerical defects remain when opted in (wrong-sign inf near FLT_MAX,
  signed zero, floored mod, scale-dependent accuracy) — documented in the warning.
