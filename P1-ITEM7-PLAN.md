# Plan: item 7 fault isolation via per-kernel compile retry

## Goal
When a df64-opted-in module's MSL compile fails (e.g. a kernel calls sin(float64),
which has no df64 overload), refuse ONLY the offending kernel(s); let sibling
kernels compile and run. Currently the whole module fails.

## Mechanism discovered
- Module.load (metal branch) calls runtime.core.wp_metal_load_library(source).
  On failure it already does a forward-only retry for adjoint problems.
- ModuleBuilder(module, options, hasher=...) accepts a custom hasher. A
  ModuleHasher built from a single kernel yields a single-kernel build. So I can
  recompile each kernel in isolation to find which fail.

## Proposed implementation (minimal, fault-isolation first)
In Module.load's metal branch, when wp_metal_load_library returns no handle AND
the failure is not the already-handled adjoint case:
1. For each live kernel, build a single-kernel ModuleBuilder (hasher with just that
   kernel), codegen it, and try wp_metal_load_library on its source.
2. Kernels whose isolated compile succeeds are "good"; those that fail are "bad",
   with the compiler error captured.
3. Rebuild the module containing ONLY the good kernels (plus the functions they
   need), load it, and record the bad kernels in metal_unsupported so their
   launches raise a clear RuntimeError naming the kernel + compiler error.
4. If ALL kernels fail, raise the original whole-module error (nothing to salvage).

This is the review's "belt and braces" fault isolation: it guarantees the property
regardless of whether type prediction is complete, and costs nothing on the
success path (only runs when the module compile already failed).

## Questions for review
1. Is rebuilding a subset module via a single-kernel ModuleHasher sound? Functions
   (@wp.func) are emitted per-module based on what the kernels reference — will a
   subset build emit exactly the needed functions, and avoid emitting a bad
   function that only a refused kernel used?
2. Caching: the subset module has a different hash than the full module. How do I
   cache it without colliding with the full module's cache entry, and without
   leaking temporary modules? Should the retry result be cached at all, or just
   the final good-subset binary under the original module's identity?
3. metal_unsupported is parsed from `// wp_metal_unsupported` markers in source.
   For compile-failure refusals (not codegen-predicted), how do I inject the bad
   kernel names so ModuleExec refuses their launches? Is there a cleaner hook than
   fabricating markers?
4. The good-subset rebuild changes the module's kernel set. Does anything downstream
   (kernel.module back-references, ModuleHasher caches, generic-kernel dedup) assume
   the module contains all originally-registered kernels? Risk of breaking a later
   launch of a good kernel?
5. Is there a simpler approach I'm missing — e.g. codegen each kernel into its own
   translation unit always (per-kernel metallib), so one failure never affects
   others? What would that cost vs the retry-on-failure approach?
6. enable_backward interaction: a kernel might compile forward but fail backward.
   The existing forward-only retry handles module-wide adjoint failure. How does
   per-kernel isolation compose with that?
