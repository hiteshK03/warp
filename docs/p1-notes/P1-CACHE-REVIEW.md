## Verdict

Don't build this as planned. Caching the source only removes codegen. The dominant cost is the MSL frontend parsing the prelude (`builtin.h` plus df64) N times, and that still happens on every load. The plan also has two real correctness bugs, described below.

## Q1: Does source caching save the dominant cost?

Mostly no. Per-kernel `wp_metal_load_library` re-parses the whole prelude. Python codegen for a single kernel is usually a small fraction of that. Measure it before writing any code: per-kernel `codegen_ms` against `load_library_ms`, cold and warm. Warm means the second process run, because macOS `MTLCompilerService` already keeps its own on-disk cache keyed by source and options. Two possible results:
- **OS cache hits on identical source:** recompiles are already cheap and your cache only saves codegen, which is marginal.
- **OS cache misses:** compile dominates, and caching source does nothing about it.

Either way the payoff is small.

No usable metallib serialization path exists:
- `newLibraryWithData` needs a metallib from offline `xcrun metal`. That compiler ships with Xcode or a separate download, so it can't be a runtime dependency.
- `MTLDynamicLibrary serializeToURL` can't hold compute kernels.
- The prelude is mostly inline templates, so it can't be split into a dylib.
- `MTLBinaryArchive` skips the backend for pipeline states, but the library is still built from source, so the frontend parse stays.

Two alternatives would actually save work:
- **Bundle the good kernels.** After isolation, write one combined source containing only the kernels that compiled. On the next load, compile that once, then retry only the K refused kernels individually. Cost drops from 1 failed whole-module compile + N single-kernel compiles to 1 + K, and K is usually 1. This keeps the never-cache-refusals rule, because refused kernels are retried every load and a kernel that later succeeds rejoins automatically. Validate mangled names against the bundle.
- **Reuse the codegen already done in this load.** Codegen has just run for the whole module before the failed compile, and isolation then runs it again N times. Slice the per-kernel pieces from that in-memory output, or memoize by per-kernel hash within the process. This handles the repeated `mark_modified` case with no disk at all. The handles a memo keeps must outlive module unload, so the memo has to own its own references.

## Q2: Hash API

As far as I recall, upstream `ModuleHasher(module)` takes a `Module`, not `([kernel], options)`. I can't confirm the fork's signature, so check it. Building a temporary Module to get a hash breaks the no-temp-Module rule. Build the key explicitly instead:

`sha256(per-kernel hash from the existing module hasher + options + native_headers_digest + Warp version + variant tag)`

- **Generic kernels:** the key must include the overload's argument types. Breaking input: generic `k(a: Any)` overloaded for `wp.array[wp.float64]` and `wp.array[wp.vec3d]`. If both overloads hash the same, they share one cache entry and the second loads the first's source. The mangled-name assert should catch it, but only if the assert also runs on the reuse path.
- **Codegen globals:** every `wp.config.*` or fork global that changes MSL output must be in the key: `mode`, `verify_fp`, any df64 or Metal type-mapping toggle. Breaking input: populate the cache, flip the df64 toggle, reload. Expected fresh codegen; actual stale source, possibly with wrong numerics.

## Q3: Forward-only vs forward+backward

`enable_backward` is in the options, so it is hashed. `metal_forward_only` is probably an internal codegen flag and not hashed. Add the variant tag explicitly.

**Bug:** the reuse path must also restore the two-tier *outcome*, not just the source. Breaking input: kernel K with `enable_backward=True` where forward+backward fails and forward-only succeeds. On a second process run, if the code takes the cached source and assumes the adjoint exists, `tape.backward()` asks for `K_..._backward`. `newFunctionWithName` returns nil, so you get a null handle or a crash. Expected: the same "backward unavailable" error the first run raises. Fix: record the variant in meta and set backward availability from meta.

There is also a cost you can't avoid: since refusals aren't cached, every two-tier kernel pays one failing forward+backward compile on every load. The cache can't remove that.

## Q4: Atomicity and cache poisoning

Write source and meta into a temporary directory, then rename the whole directory with `safe_rename`, the same pattern as module builds. Treat meta as the presence marker.

**Bug:** a cached source that fails to compile poisons the cache. Breaking input: truncate the cached `.metal` file (crash mid-write without directory rename), or upgrade macOS so the new compiler rejects a source the old one accepted. Each load reads the cache, the compile fails, the kernel is refused, and the entry stays, so the failure repeats on every load. In effect that is a cached refusal. Fix: when a cached source fails to compile, delete the entry, regenerate, and compile once fresh. Refuse only if the fresh source also fails, and persist nothing.

## Q5: Staleness

The content hash covers called `@wp.func`s through recursive reference hashing (structs, constants, native snippets), assuming the fork still does what upstream does. So redefining a function is safe. The risks are the globals listed in Q2 and the generic overload types. One test would settle it: redefine a called function, toggle each codegen global, and assert the cache misses each time.

## Q6: Cost and benefit

This only runs when a module contains an fp64 kernel Metal can't compile. That is a user-visible limitation users normally fix, not a steady state. It recurs in practice in the test suite and in `mark_modified` loops. The saving per load is codegen × N, which is small, and you add a new cache namespace, an invalidation path, a variant tag, and a poisoning risk.

Recommendation: measure first. If isolation shows up in timings, do the in-process codegen reuse first (cheap, no disk), then the good-kernel bundle if compile time dominates. Skip the per-kernel source cache.