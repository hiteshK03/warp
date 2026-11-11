# Caching measurements (drive the design)

## Cost breakdown (bench/p1_cache_bench.py, 6 good + 1 bad df64 kernel)
- cold: total 1347ms = codegen 157ms + compile 1140ms (compile 7.3x codegen)
- warm: total 932ms = codegen 117ms + compile 797ms (compile 6.8x codegen)
=> Source-caching saves only ~12% (codegen). The MSL compile dominates and is NOT
   avoided by caching source. Per-kernel source caching is low-value.

## Bundle concept verified (bench/p1_bundle_probe.py)
ModuleHasher(good_subset) -> ModuleBuilder emits ONE prelude + all subset kernel
bodies; compiles as a single library; both good kernels' symbols present; bad
kernel absent. So a good-kernel bundle is a valid single translation unit.

## Mangled-name stability (bench/p1_name_stability.py)
forward+backward vs forward-only override: mangled name IDENTICAL (k_c827c0a8).
enable_backward is a MODULE option, not folded into the per-kernel hash
(kernel.options is empty for this kernel). So forward-only kernels can live in the
bundle under their original names — no name drift.

## Design conclusion
Cache an isolation RECORD (refused set + backward_refused + bundle_failed +
compiler_id), keyed by whole-module hash + compiler_id. Regenerate bundle source
in-process each load. This skips the failed whole-module compile and reduces N
single compiles to 1 bundle + K refused retries. See P1-CACHE-PLAN-V3.md.
