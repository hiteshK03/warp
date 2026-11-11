# Plan v2: good-kernel bundle caching (replaces per-kernel source cache)

## Measurement-driven decision
Measured (bench/p1_cache_bench.py, 6 good + 1 bad kernel):
- cold: total 1347ms, codegen 157ms, compile 1140ms (compile 7.3x codegen)
- warm: total 932ms, codegen 117ms, compile 797ms (compile 6.8x codegen)
Source-caching saves only ~12% (codegen). Compile dominates and is NOT avoided by
caching source. So per-kernel source caching is low-value (review agreed).

## The bundle approach (review's recommendation)
After isolation identifies the good kernels, write ONE combined .metal source
containing only those kernels (a "bundle"), cached content-addressed. On the next
failed load:
1. Try the cached bundle first: ONE wp_metal_load_library for all good kernels.
2. Retry only the K refused kernels individually (K usually 1).
Cost drops from (1 failed whole-module + N single compiles) to (1 bundle + K).
Refusals are still never cached: refused kernels are retried every load, so a
kernel that later compiles rejoins automatically.

## Design details
- Bundle key: sha256(sorted good-kernel mangled names + their per-kernel hashes +
  options + native_headers_digest + warp version + variant tag). Must change if the
  good-kernel SET changes (a kernel newly refused or newly compiling).
- Bundle source: concatenate the per-kernel codegen sources. BUT each single-kernel
  source includes the full prelude (builtin.h etc.) — concatenating N preludes would
  redefine symbols. Need to emit the prelude ONCE + the N kernel bodies. Investigate
  whether ModuleBuilder can codegen a subset of kernels sharing one prelude (it can:
  ModuleHasher(good_kernels, options) yields exactly those roots, one prelude).
- So the bundle is just ModuleBuilder with a hasher over the GOOD kernels only.
  That is a subset build — but the prior review warned against subset rebuilds
  because they can re-fail on interaction errors. Mitigation: the bundle is built
  from kernels ALREADY proven to compile in isolation; if the bundle compile fails
  (interaction), fall back to per-kernel libs (the current path). Never loop.
- Cache the bundle SOURCE + meta (which kernels, which are backward-refused).
- On load: if bundle cache exists, compile it; if it succeeds, use it for the good
  kernels and retry refused ones individually. If bundle compile fails, delete the
  cache entry and fall back to full per-kernel isolation.
- Backward-refused kernels: their forward-only variant must be recorded in the
  bundle meta and restored (review Q3 bug).

## Questions for review
1. Building the bundle via ModuleHasher(good_kernels) reuses the existing
   one-prelude codegen — confirm this emits a valid single translation unit with
   all good kernels and no duplicate symbols. Is this materially different from the
   "subset rebuild" the earlier review warned against, given the kernels are
   pre-validated in isolation?
2. The bundle key must capture the good-kernel SET. If I key on sorted mangled
   names + per-kernel hashes, does that uniquely determine the bundle source?
3. Interaction fallback: if the bundle fails to compile but each kernel compiled
   alone, I fall back to per-kernel libs. Is there a risk of an infinite loop
   (bundle fails -> isolate -> rebuild bundle -> fails)? How do I prevent caching a
   bundle that just failed?
4. Where to store the bundle: a separate cache namespace (wp_metal_bundle_<hash>)
   so it never collides with whole-module or per-kernel entries. The whole-module
   entry still exists (the failed source) — does its presence interfere?
5. Is the bundle worth it vs the simpler in-process codegen memo (review's other
   suggestion)? The bundle saves compiles across processes; the memo only saves
   codegen within a process. Given compile dominates, the bundle is the one that
   matters — confirm.
6. meta must record per-kernel backward availability. How is the existing
   .metal.meta structured, and can I extend it for the bundle?
