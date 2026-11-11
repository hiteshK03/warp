# Per-kernel success caching — IMPLEMENTED (record-cache design)

## Design (after 3 review rounds + measurement)
- v1 per-kernel SOURCE cache: rejected — measurement showed compile dominates
  (7.3x codegen), source-caching saves only ~12%.
- v2 good-kernel bundle SOURCE cache: rejected — circular key (good set known only
  after isolation) + codegen-staleness bug class.
- v3 isolation RECORD cache (implemented): persist {refused, backward_refused,
  bundle_failed, compiler_id, codegen_fp} keyed by whole-module hash + compiler_id
  + codegen fingerprint. Regenerate bundle SOURCE in-process each load. On a record
  hit, skip the failed whole-module compile; compile ONE bundle of good kernels +
  retry refused kernels individually.

## Measurement justification (bench/p1_q5_real.py, 6 distinct good + 1 bad)
- warm per-kernel isolation: 854ms (6 distinct sources, each full prelude parse;
  Apple's shader cache does NOT help because each source differs)
- warm bundle: 46ms (one source, Apple-cache hot)
- saving ~808ms/module >> 50ms gate. WORTH BUILDING.
- e2e (bench/p1_record_e2e.py): cold 1267ms -> warm 685ms (~46% faster), record path
  confirmed via "(record-cache)" timer tag.

## Implementation
- Helpers: _metal_codegen_fingerprint (hash codegen.py/context.py/build.py),
  _metal_compiler_id (kern.osversion + machine), record read/write/delete (atomic
  os.replace, JSON, version+compiler+codegen guarded).
- Module._metal_build_bundle: one library for good kernels (full + forward-only
  groups), codegen-time forward-only override (no Kernel.options mutation).
- Module._metal_load_with_record: record-driven load; retries refused kernels
  (healing), bundles good ones, falls back to per-kernel on bundle interaction
  failure, persists updated record, deletes only when fully clean (B1).
- Module._metal_isolate_kernels_single: two-tier single-kernel isolation.
- First isolation writes the record after validating the bundle compiles (avoids
  cross-process loop, B4/Q3).
- ModuleExec.metal_bundle_libs owned + freed in __del__ (unique handles, no
  double-free); per-kernel libs dedup against bundle libs.
- WARP_METAL_REISOLATE=1 forces fresh isolation.

## Review bugs addressed
- B1 delete only when refused ∪ backward_refused empty and not bundle_failed.
- B2 codegen fingerprint in key (codegen.py change invalidates).
- B3 compiler_id + codegen_fp in the filename (no cross-host ping-pong).
- B4 bundle_failed set only after a real bundle compile attempt; validated before
  recording False.
- B5 refused diagnostics stored (truncated) and raised at launch.
- B6 backward_refused kernels: forward-only bundle has no _backward symbol;
  metal_backward_unsupported set so adjoint launch raises.
- Q2 compiler_id uses kern.osversion (not mac_ver product version).
- Q3 mangled-name stability verified (enable_backward is a module option).
- Q4 record lookup before whole-module compile; bundle in-memory (no disk source).

## Tests (warp/tests/test_metal.py::TestMetalFaultIsolation, 6 cases)
sibling survival, generic overload, backward preserved, no-recompile-on-relaunch,
record persisted+reused (spy confirms record path), WARP_METAL_REISOLATE bypass.
Full test_metal: 17 pass / 1 skip. fixedarray 5/5.

## Remaining honest gaps
- bundle_failed is sticky until hash/compiler/codegen changes (a transient XPC
  failure during the validation compile could set it; review B4 suggested
  distinguishing NSError domains — not implemented, native API returns no domain).
- No dedicated injected interaction-failure test (bundle fails, kernels pass alone).
- backward_refused not retried per-load (relies on compiler_id/codegen_fp
  invalidation); a same-OS compiler fix would need a fingerprint change to heal.
