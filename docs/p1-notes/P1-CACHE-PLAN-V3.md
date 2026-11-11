# Plan v3: isolation RECORD caching (not source caching)

## The pivot (review v2 core correction)
Caching bundle SOURCE is wrong:
- Circular key: the good-kernel set is only known AFTER isolation (the cost we avoid).
- Staleness: a cached source written by old codegen loads against new runtime ->
  silent garbage (warp.__version__ doesn't change when codegen.py changes in a fork).
Correct design: persist an isolation RECORD keyed by the WHOLE-MODULE hash (already
covers all kernels + options + headers) + compiler_id. Regenerate the bundle source
in-process every load (cheap codegen, ~12%). The record tells us which kernels to
skip compiling as a whole and which to retry.

## Record format (JSON, advisory, schema-versioned)
Key: whole-module hash (module.get_module_hash) + compiler_id
Value: {
  version: 1,
  refused: [mangled_name, ...],          # kernels that failed forward+backward AND forward-only
  backward_refused: [mangled_name, ...], # kernels whose forward compiles, backward doesn't
  bundle_failed: bool,                   # good kernels fail together (interaction) -> per-kernel only
  compiler_id: "<kern.osversion>|<MTLDevice.name>",
}
Stored at: kernel_cache_dir/wp_metal_iso_<modulehash16>.json (separate namespace).

## Load flow (metal, when whole-module compile fails OR a record exists)
1. Compute whole-module hash + compiler_id; look up record.
2. If record exists and not bundle_failed:
   - good_roots = all_roots - refused
   - Build bundle source in-process from good_roots (ModuleHasher(good_roots)).
     For backward_refused kernels, use a codegen-time forward-only override (do NOT
     mutate Kernel.options — that changes the mangled name).
   - Compile bundle ONCE. If success: use it for good kernels.
   - Retry each refused kernel individually (forward+backward, then forward-only).
     If a refused kernel now compiles, remove it from refused (rewrite record).
   - If bundle compile fails: set bundle_failed=true, fall back to per-kernel libs.
3. If no record: run full isolation (current path), then:
   - Try compiling the bundle of good kernels; only persist the record if it succeeds
     (or persist with bundle_failed=true if it fails). This avoids the cross-process
     perpetual loop (review Q3).
4. If refused becomes empty: delete the record (normal path resumes).

## Invariants
- compiler_id in the key => OS/device upgrade invalidates (fixes backward_refused
  staleness, review Q3 "never cache refusals" violation).
- bundle_failed is a cached NEGATIVE but safe: it only disables an optimization,
  never a correctness verdict; refused kernels are still retried every load.
- Regenerating source each load removes the codegen-staleness bug class entirely.
- Bundle library owned by ModuleExec (freed with per-kernel libs) — no leak/UAF.
- Atomic write (temp + os.replace); JSON parse error / missing file = miss.
- Hasher state must not leak into module.hashers / kernel mangled names (test).
- Forward-only override is codegen-time, never mutates Kernel.options (test).

## Questions for review
1. The whole-module hash as key: when a refused kernel's SOURCE changes (user edits
   it), the whole-module hash changes, so the record misses and we re-isolate. Good.
   But if only a GOOD kernel changes, the hash also changes -> re-isolate. Acceptable?
2. compiler_id: kern.osversion via platform.mac_ver() or os.uname()? MTLDevice.name
   via wp_metal_device_get_name. Is OS build + device name sufficient to invalidate
   on a Metal compiler change, or do I need the GPU compiler version too?
3. Forward-only codegen override without mutating Kernel.options: the two-tier path
   currently builds ModuleHasher([kernel], fo_options) where fo_options has
   enable_backward=False. Does that change the mangled name? If the hash covers
   options, yes. How do I keep the name stable while emitting forward-only source?
4. When the record exists, I skip the whole-module compile. But the whole-module
   .metal source is still on disk (from the first failed build). Does its presence
   cause _compile to short-circuit (return False, "cached") and skip codegen, leaving
   me without fresh source? Need to ensure the record path bypasses that.
5. Is persisting the record worth it vs always re-isolating? The saving is: skip 1
   failed whole-module compile + (N-K) good-kernel compiles, replaced by 1 bundle
   compile. For N=7,K=1: from 8 compiles to 2. Quantify the expected win.
