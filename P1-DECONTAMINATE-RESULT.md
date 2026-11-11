# De-contamination: DONE and verified

## What was contaminated (and what wasn't)

Full sweep `git diff --stat metal-1.17 HEAD -- warp/` (review G2): 10 files.
- codegen.py, context.py: clean (1.17 + only my P0a/df64 edits)
- array.h, vec.h, mat.h, quat.h, svd.h, metal_crt.h, metal_float64.h: clean
- builtins.py, exports.h, types.py, tile.h, range.h: UNCHANGED vs 1.17 (verified)
- **builtin.h: CONTAMINATED** with two upstream 1.18 changes copied from main
  during P1a (commit 4e651fdb):
  - `floordiv_signed` rewrite — upstream 7784b3e3 "Floor integer division toward
    negative infinity [GH-1918]"
  - `launch_bounds_t` `int shape[N]` -> `uint32_t shape[N]` — upstream 172ac26c
    "Validate wp.tid() launch extents [GH-1815]"
  Provenance confirmed via `git log -S` (review G3): both are genuine upstream
  commits, NOT Metal fixes I needed.

I initially mis-diagnosed array.h as contaminated too; corrected: array.h's
1.17-vs-HEAD diff is ONLY the P0a fixedarray block (prefix/suffix byte-identical to
1.17). The slot-atomic helpers I saw were in main, not HEAD.

## Latent bug the contamination caused (review Q4, CONFIRMED)

Porting `floordiv` to floor semantics WITHOUT porting `mod` broke the identity
`a == (a//b)*b + (a%b)` for negative operands. Measured on the contaminated tree:
18/90 cases broken, e.g. -7//2=-4, -7%2=-1 -> -4*2+-1 = -9 != -7.

## Fix (mechanical restore, review G4)

`git show metal-1.17:warp/native/builtin.h` restored as the base, then re-applied
ONLY the 3 df64 guard pairs (adj_float64, DECLARE_FLOAT_OPS(float64),
DECLARE_ADJ_APPROX_DIV(float64)). Forward fix-up commit; history not rewritten
(earlier commits cite measurements on the contaminated tree — those are flagged).

## Verification (fresh cache per run, review G1)

- divmod/launch table on cleaned HEAD vs PURE metal-1.17 overlay: **bit-identical**
  (np.array_equal True). Identity now 90/90. 4D launch (2,3,5,7)==arange(210);
  1D dim=1000003 fill correct + sentinel untouched.
- P0a fixedarray suite: 5/5 PASS
- P0a fixedarray ATOMIC forward + Tape backward (new test, review Q5.2): PASS
  (confirms no dependency on the dropped 1.18 slot helpers)
- df64 opt-in: config inert, module opt-in launches 0.33333334, ValueError on bad
  value, warm relaunch consistent
- error-mode corpus: f32 launches, all fp64 cases refused, mixed-module clean
  kernel launches while its fp64 sibling is refused

## Note on earlier measurements

Commits e210cc65 / acd9ff0d cite source-differential and decision measurements
taken on the contaminated tree. The decision measurement (Newton inertia, ~1e-3
scale) is unaffected by floordiv/launch-bounds contamination. The source-identity
differential WAS affected and is re-done below with the correct preprocessed
comparison.
