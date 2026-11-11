# Plan: per-kernel success caching for Metal fault isolation (item 7 #9)

## Problem
When a df64 module's whole MSL compile fails, _metal_isolate_kernels recompiles
every kernel from scratch (codegen + wp_metal_load_library) on EVERY load and every
mark_modified. Metal caches generated SOURCE (.metal), compiled at load time, so
each isolated compile re-parses the full Warp MSL prelude (builtin.h + df64) — N
times the prelude cost, every failed load.

## Current caching model (verified by reading the code)
- _compile writes the generated .metal source + .metal.meta to
  kernel_cache_dir/wp_<name>_<hash7>/, keyed by the MODULE hash.
- For Metal, the source IS the cached artifact; wp_metal_load_library compiles it
  at load time. There is no persisted metallib.
- _metal_isolate_kernels currently builds each kernel's source in-memory and
  compiles it directly, writing NOTHING to cache.

## Proposed design
Cache each successfully-isolated kernel's generated SOURCE, content-addressed by
its single-kernel hash, in a per-kernel subdirectory. On a later failed load,
reuse the cached source (skip codegen) and only recompile (wp_metal_load_library).

Concretely, in _metal_isolate_kernels, per root kernel:
1. Build a single-kernel hasher; get its module_hash (content-addressed, includes
   the kernel source + options + native_headers_digest).
2. Cache path: kernel_cache_dir/wp_metal_iso_<hash16>/<hash16>.metal (+ .meta).
   Use a distinct prefix so it can never collide with a whole-module entry.
3. If cache_kernels and the cached source + meta exist: read them, skip codegen.
   Else: codegen, write source + meta to the cache path (atomic rename).
4. wp_metal_load_library(source). On success: libs[name]=handle, cache the source.
   On failure: refuse (do NOT cache the refusal or the failing source).
5. Two-tier: if forward+backward fails, try forward-only; cache whichever source
   succeeded under a key that distinguishes backward-enabled vs forward-only.

## Key invariants (from the prior reviews)
- Refusals are NEVER cached (transient compiler-service failures must not persist).
- The cache key MUST include native_headers_digest (already in module_hash) so a
  header change invalidates it. Verify the single-kernel module_hash includes it.
- The cache key MUST distinguish forward-only vs forward+backward (different source).
- Per-kernel cache entries must not collide with whole-module entries (distinct prefix).
- mangled-name stability still asserted.
- Do not create temporary Module objects (global name registration leak).

## Questions for review
1. Is caching the SOURCE (not a metallib) the right granularity, given Metal
   compiles at load time? The recompilation (wp_metal_load_library) still happens
   every load — does caching source actually save the dominant cost, or is the
   compile itself the bottleneck? Should I instead cache the compiled metallib to
   disk (is there a wp_metal_serialize_library / metallib file API)?
2. The single-kernel module_hash: does ModuleHasher([kernel], options).get_hash()
   exist and include native_headers_digest + options? Confirm the API.
3. forward-only vs forward+backward cache keying: the options differ
   (enable_backward, metal_forward_only). Are those folded into the hash? If not,
   how do I key them apart without collision?
4. Atomic write: reuse warp._src.build.safe_rename, or write source+meta together?
   What happens if the process dies mid-write (partial cache entry)?
5. Is there a risk that a cached per-kernel source becomes stale relative to the
   kernel's current adjoint state (e.g. a @wp.func it calls was redefined)? The
   module_hash should capture this — verify.
6. Cost/benefit: is this worth the complexity given isolation only runs on the
   FAILURE path? Quantify: how often does a df64 module fail to compile as a whole
   in practice (only when it contains an unsupported fp64 kernel)?
