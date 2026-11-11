## Verdict

Record-over-source pivot is right. Plan still has one loop bug, one staleness hole that hits this branch directly, a filename/key mismatch, and a cost model that doesn't match the steps. Without measurements, Q5 may not justify building this.

## Bugs

**B1. Step 4 delete condition loops on backward-only refusals.**
- Input: module with one kernel X. X's forward compiles; its adjoint uses `sin(float64)`. Isolation gives `refused=[]`, `backward_refused=[X]`.
- Step 4 sees `refused` empty and deletes the record.
- Next load: whole-module compile fails, full re-isolation runs, record is written, then deleted again.
- Expected: steady state with the record present. Actual: every load pays full isolation.
- Fix: delete only when `refused ∪ backward_refused` is empty and `bundle_failed` is false.

**B2. "Regenerating source each load removes codegen staleness" is false for the record.**
- The record is a verdict produced by old codegen. `get_module_hash` covers kernel source and options. It does not cover `codegen.py`, Metal prelude headers or the fork's backend files.
- Input on this branch (metal-df64): record says `backward_refused=[X]` for an adjoint using `sin(float64)`. You then teach codegen to emit df64 for that adjoint. Module hash is unchanged, so the record hits.
- Expected: backward compiles. Actual: X stays forward-only until someone deletes the cache.
- `refused` heals itself because it is retried each load. `backward_refused` and `bundle_failed` are never retried, so they stay stale forever.
- Fix: add a fork codegen fingerprint to the key. Hash the contents of `codegen.py`, Metal headers and backend `.py`/`.h` once at import, or use a hash of the built native lib.

**B3. Filename does not contain `compiler_id`, but the key claims to.**
- File is `wp_metal_iso_<modulehash16>.json`. Two compiler_ids map to the same file.
- Input: shared `WARP_CACHE_PATH` (network home, CI cache restore) used by an M1 and an M3 host, or the same box across an OS update with a rollback snapshot. Each host sees a compiler_id mismatch, treats it as a miss, re-isolates and overwrites the other's record.
- Expected: one record per compiler. Actual: perpetual ping-pong, the cross-process loop you were trying to kill.
- Fix: `wp_metal_iso_<modulehash16>_<sha(compiler_id+codegen_fp)[:16]>.json`.

**B4. `bundle_failed` is permanent and set on any failure.**
- Input: MTLCompilerService XPC interruption or timeout under memory pressure during the one bundle compile.
- Expected: bundle retried next load. Actual: `bundle_failed=true` until the hash or compiler changes, so per-kernel compiles happen forever.
- "Only disables an optimization" is true, but it makes the cache net-negative.
- Fix: set it only on a deterministic compiler diagnostic, meaning a non-nil `NSError` with a compile-error domain and code. Never set it on XPC or interruption errors. Or use retry-N-then-stick.

**B5. Refused kernels have no stored diagnostic.**
- On a warm run, launching a refused kernel can only say "refused", because the error text came from a compile that never ran in this process.
- Store a truncated first error line per mangled name. Raise it at launch.

**B6. Adjoint launch of a `backward_refused` kernel must hard-fail.**
- The forward-only bundle has no `_backward` function, so `newFunctionWithName` returns nil.
- Input: `tape.backward()` over X.
- Expected: `RuntimeError` naming X plus the stored diagnostic. Possible actual: nil pipeline, then a skipped dispatch and silently zero gradients, or a crash.
- Add a test asserting the raise.

## Q1: whole-module key, any-kernel edit re-isolates

- Correct. The cost lands exactly in the edit loop, where the cache was supposed to help.
- `refused` and `backward_refused` are really per-kernel properties. Key them by mangled name, which already embeds the kernel hash including dependencies, plus module options hash, compiler_id and codegen fingerprint.
- Only `bundle_failed` is module-level.
- For v1, whole-module keying is acceptable if B2 and B3 are fixed. Don't over-engineer before Q5 is answered.

## Q2: compiler_id

- `platform.mac_ver()` is wrong. It returns the product version: two betas of 27.0 (you are on Darwin 27 beta) return the same string while the compiler differs.
- `os.uname().version` is the xnu string, not the OS build.
- Use `sysctlbyname("kern.osversion")` (e.g. `25A5xxx`, which changes per beta and per Rapid Security Response) plus `MTLDevice.architecture.name` (macOS 14+, e.g. `applegpu_g15s`). The architecture name is more precise than the marketing name.
- Also include whatever `MTLCompileOptions` you set (`languageVersion`, `fastMathEnabled`/`mathMode`) if they are chosen at runtime.
- The runtime compiler ships with the OS, so this set is sufficient.
- Exception: if the fork shells out to `xcrun metal` to build `.metallib` offline, the toolchain is separate from the OS. Then add the `xcrun metal --version` output.

## Q3: forward-only override and the mangled name

- Your verification holds only for module-level `enable_backward`.
- Adversarial inputs to test:
  - `@wp.kernel(enable_backward=True)` with an explicit per-kernel option.
  - Generic kernel overloads.
  - A kernel calling an overloaded `wp.func`.
- If `hash_kernel` folds in a non-empty `kernel.options`, the per-kernel case changes the name.
- Required test: for every kernel in the module, the mangled name from `ModuleHasher(good_roots)` with override equals the name from the full-module hasher.
- Note for tiles: the forward source genuinely differs with `enable_backward=False`, because grad tile storage is not allocated. Fine since you regenerate, but threadgroup-memory refusals can flip between the two tiers. That is another reason to keep the two-tier result per kernel.

## Q4: `_compile` short-circuit

- Yes, this is a risk if the failed whole-module `.metal` is promoted into the final module dir.
- Input: delete the record JSON, or change compiler_id. The normal path's `_compile` sees the source, returns `cached`, and skips codegen.
- Expected: fresh codegen and a compile attempt. Actual: stale source compiled, or a missing library loaded.
- Rules:
  - Failed builds never get renamed into the final dir. Upstream builds in a temp dir and renames only on success; check that the fork preserves this.
  - The record lookup happens before `_compile` is called for the whole module.
  - The bundle is either in-memory only, or its path includes `sha(sorted(good_set))`.
- Otherwise, once a retried refused kernel joins the good set, a cached bundle without it short-circuits and `newFunctionWithName` returns nil.

## Q5: is it worth it

Your "8 vs 2" counts the wrong things. For N=7, K=1 (K fails both tiers):

| Path | Compiles |
|---|---|
| No record | 1 whole (fail) + 7 fwd+bwd (6 ok, 1 fail) + 1 fwd-only (fail) = 9, plus 1 bundle validation = 10 on the first run |
| Record, as planned | 1 bundle + 2 retry failures for K = 3, not 2 |

- More important: Apple's system Metal shader cache already makes identical-source `newLibraryWithSource` calls warm on the second run.
- So the real warm-run saving is roughly: one failed whole-module compile, plus 6 cache-hit lookups, plus 5 extra `MTLLibrary` objects.
- Failed compiles are probably not cached by Apple, so the per-load retries of K cost about what you save.
- If the `double` error surfaces in Sema, as it usually does, the failed whole compile is cheap and the net win may be tens of ms.

Do this first: measure second-run wall time of the current no-record path against a hand-built single good-kernel library. If the gap is under about 50 ms per module, don't build this.

If you do build it:
- Drop per-load retries of `refused`. They are deterministic given compiler_id and codegen fingerprint, which is the point of keying on them.
- Provide `WARP_METAL_REISOLATE=1` to force re-isolation.
- Then warm cost is one cache-hit compile, and the feature pays for itself.