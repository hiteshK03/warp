# Differential corpus result: my "identical to pre-P1" claim was FALSE

The review demanded generated-source identity, not just matching behaviour. I
built the 15-case corpus (bench/p1_msl_corpus.py) and ran it against a real pre-P1
overlay and against HEAD in error mode, then compared results, module hashes, and
MSL text (bench/p1_compare_corpus.py).

## First run: claim falsified

- Launch outcomes / values / exception messages: **15/15 MATCH**
- Module hashes: **14/14 DIFFER** — including `c_f32`, which contains no float64
- MSL text: **13/13 files differ**

Cause: `expand_includes` inlines headers TEXTUALLY. My `#if defined(WP_METAL_DF64)`
guard around `#include "metal_float64.h"` excluded the code at *compile* time, but
the ~282 lines of header text were still inlined into every module's source,
changing every hash. `_resolve_metal_conditionals` only evaluates
`__METAL_VERSION__`, not arbitrary defines.

## Fix applied

Removed the include from `metal_crt.h` entirely; codegen now appends the df64
header text AFTER base-header expansion, and only for opted-in modules.

## Second run: behaviour and pairing fixed, source still differs

- Results: 15/15 MATCH; file sets now pair 13/13 (no spurious only-in-one-side)
- Module hashes: still 14/14 DIFFER
- MSL text: still 13/13 differ

Remaining cause is NOT the opt-in mechanism — it is **P1a's edits to shared
headers**, which are unconditional:
- `builtin.h`: `#if !defined(WP_NO_FLOAT64)` -> `... || defined(WP_METAL_DF64)`
  (guard text changes even when the branch is inactive), `adj_float64` guard,
  `DECLARE_FLOAT_OPS(float64)` / `DECLARE_ADJ_APPROX_DIV(float64)` guards,
  and a `floordiv` relocation
- `vec.h` / `mat.h` / `quat.h` / `svd.h`: alias guards and the `mat_t` cross-type
  explicit-cast change
- `metal_crt.h`: new comment block

So error mode is **behaviourally** identical to pre-P1 on all 15 cases, but NOT
source-identical. The honest claim is the weaker one.

## What full source identity would require

Making every P1a shared-header edit conditional in a way that leaves the
non-opted-in text byte-for-byte unchanged — e.g. moving all df64-specific
declarations into the injected header and reverting the shared headers to their
original text, using overloads/specializations that only exist when injected.
The `mat_t` explicit-cast change is the awkward one: it is a genuine improvement
but it edits shared code, so it either stays (and source identity is abandoned)
or moves behind a traits hook.

## Status

- Opt-in mechanism: works, module-scoped, no global leak, warning fires once.
- Error mode: behaviourally pre-P1 on 15 cases; **not** source-identical.
- Claim corrected here rather than left standing.
