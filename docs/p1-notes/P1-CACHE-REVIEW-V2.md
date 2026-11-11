**Verdict:** the direction is right, but the plan caches the wrong artifact. Compile time dominates, and what saves compiles is knowing which kernels are refused, not the bundle source text. Caching the source adds staleness and keying bugs and saves only about 100ms of codegen. Persist an isolation record and rebuild the bundle source in-process every time.

## Core design flaw: the bundle key can't be computed at lookup time
The proposed key is a hash of the good-kernel set, but you only learn that set by running isolation, which is the cost you're trying to avoid. That makes the key circular. The lookup key has to be the whole-module hash (it already covers every kernel, the options and the headers) plus the compiler identity. Store this record at that key:

`{version, refused:[mangled], backward_refused:[mangled], bundle_failed:bool, compiler_id}`

with `compiler_id` = OS build (`kern.osversion`) + `MTLDevice.name`.

## Biggest perf miss: the plan still compiles the known-bad whole module first
As written, the next load is: failed whole module → bundle → K singles. That is two large compiles plus K. If the record exists, skip the whole-module attempt and go straight to bundle + K. The whole-module entry on disk then doesn't interfere (Q4), because you never compile it while the record exists. When a refused kernel compiles in its single retry, rewrite the record. When `refused` becomes empty, delete the record so the normal path comes back.

## Q1: subset build validity
`ModuleBuilder(ModuleHasher(good))` is the same machinery as your per-kernel isolation with |S| = N instead of 1. It is not materially riskier than the warned-against subset rebuild: any symbol collision between good kernels would already break ordinary all-good modules. The real risks are state and naming:

- **Hasher state leaking into the module.** If constructing `ModuleHasher(subset)` writes into `module.hashers`, the per-kernel hash fields or anything the mangled name reads, the full-module hash gets corrupted.
  - Test: build the bundle, then check that `module.get_module_hash()` is unchanged and every `kernel.get_mangled_name()` is identical before and after.
  - Failure: a later load keys the subset source as the whole module, and `newFunctionWithName` returns nil for the refused kernel.
- **Forward-only variants changing names.** If you make the forward-only variant by setting `kernel.options["enable_backward"]=False`, the per-kernel hash covers options, so the mangled name changes.
  - Test: generic kernel `k` with an int overload and a float overload, where the float backward is refused. Bundle-launch `k[float]`.
  - Expected: launches. Actual: the function lookup misses, and the user's kernel has also lost backward for good.
  - Fix: pass a codegen-time override; never mutate `Kernel.options`.

## Q2: key uniqueness
Sorted names plus per-kernel hashes don't capture `backward_refused`: the same good set with different backward availability produces different source. A deeper problem: `warp.__version__` in a dev fork doesn't change when `codegen.py` changes.
- Failure: edit the Metal codegen (say, the argument-buffer layout), and a cached bundle source written by the old codegen loads against the new runtime. You get silent garbage or a launch mismatch, not a compile error.
- Regenerating the source each load removes this whole class of bug.

## Q3: loop
No infinite loop in-process if isolation takes a "no bundle" flag instead of re-entering `load()`. Across processes there is a perpetual loop:
- Setup: kernels A and B each compile alone but fail together. Run 1 isolates them and writes the record. Run 2 tries the bundle, fails, deletes the record, isolates and writes the same record again. This repeats forever.
- Each run pays failed bundle + N singles, which is worse than having no cache.
- Fix: set `bundle_failed=true` keyed by `compiler_id` and go straight to per-kernel libs. That is a cached negative, but a safe one: it only disables an optimisation and is never a correctness verdict.
- Better still, compile the bundle in the first process before persisting the record, and write the record only if that compile succeeds.

## "Never cache refusals" is violated by `backward_refused`
Kernels refused outright get retried every load, but backward-refused kernels sit in the bundle forward-only and their backward is never retried.
- Failure: backward is refused on macOS X. Upgrade to X+1, where it compiles. `tape.backward()` still raises "backward unavailable".
- Fix: put `compiler_id` in the key so an OS or device change invalidates the record, or retry backward singly (K_b extra compiles).

## Q4: storage details
A separate namespace is fine. Requirements:
- Write atomically (temp file + `os.replace`).
- Treat a JSON parse error or `FileNotFoundError` as a miss. Failure otherwise: process A deletes the record after its bundle fails while process B is reading it, and B crashes.
- Lifetime: the bundle `MTLLibrary` must be owned by the module exec next to the per-kernel libs. Failure otherwise: unload and reload the module in a loop, and you get either a leak (bundle lib not in the free list) or a use-after-free (freed while hooks still hold its `MTLFunction`). You've fixed a metallib leak once already.

## Q5: bundle vs memo
Agreed, the memo is worthless here: it saves part of the 12% codegen, and in-process reloads already hit the module exec cache. Two points to check, though:
- Your warm compile (797ms) shows no system-level hit. Check whether a second process compiling byte-identical source is fast. If it is, keep the bundle source byte-deterministic (stable kernel order) to get that hit.
- If you want to avoid compile entirely, the only real path is a binary: an offline `xcrun metal` → `.metallib` loaded with `newLibraryWithURL` (needs the Metal toolchain installed). `MTLBinaryArchive` only skips the backend; you still compile source to get the library.

## Q6: the meta format
`.metal.meta` wasn't shown, so I can't confirm its structure. Whatever you extend needs:
- a schema version field;
- readers that ignore unknown fields and treat a missing record as a miss;
- an atomic write;
- no field the loader treats as "compiled OK", since this record is advisory.

## Cheaper alternative worth measuring
Map the compiler error log's line numbers back to kernels, with errors in shared functions assigned to every kernel that reaches them. That finds the refused set in about 2 compiles even on a cold cache. It's safe because the per-kernel fallback catches any wrong attribution.