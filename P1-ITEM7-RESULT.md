# Item 7: per-kernel fault isolation — IMPLEMENTED and verified

## Problem
In a df64-opted-in module, a kernel calling sin(float64) (no df64 overload) failed
the WHOLE module's MSL compile, taking down sibling float32 kernels.

## Implementation (per the review: keep per-kernel libs, never rebuild a subset)
- ModuleExec gains metal_per_kernel_libs (mangled_name -> metallib handle) and
  metal_refused (mangled_name -> compiler error).
- Module._metal_isolate_kernels: on whole-module compile failure, build each live
  kernel in its own translation unit via ModuleHasher([kernel], options) +
  ModuleBuilder(..., hasher=...). Good kernels get a metallib; bad ones are recorded
  with their compiler error. Uses the exact resolved options from the failed build
  (no config re-read). Refusals are NOT cached (transient compiler failures).
- Module.load metal branch: on failure, isolate; if any kernel compiles, build a
  ModuleExec with per-kernel libs + refused dict; if none compile, raise the
  original error.
- get_kernel_hooks: refused kernels raise a clear RuntimeError keyed by mangled
  name (so generic overloads refuse individually); good kernels look up their own
  metallib. __del__ unloads the per-kernel libs.

## Verified (review's minimum tests)
- sin(float64) kernel refused individually; clean f32 sibling LAUNCHES (value 6.0).
- Generic overload pair: f32 overload launches sin(0.5)=0.479426; refusal is per
  mangled name, not per key.
- Two launches of a good kernel: both 6.0, ModuleExec cached (no re-isolation).
- Mixed backward: f32 kernel with working adjoint next to a refused df64 kernel
  still differentiates (grad=6.0). The good kernel keeps its backward.
- Regressions: fixedarray 5/5; df64 opt-in (config inert, opt-in works, ValueError,
  warm relaunch); error-mode corpus (f32 launches, fp64 refused).

## Scope limits (honest)
- Isolates MSL-compiler faults only. A WarpCodegenError while building a kernel
  still propagates (codegen runs before MSL exists) — though _metal_isolate_kernels
  catches per-kernel codegen exceptions and refuses that kernel, so a codegen error
  in one kernel during isolation does not kill the others.
- The pre-existing module-wide forward-only retry still runs before isolation; in
  the mixed-backward test the good kernel kept its backward, but a systematic
  per-kernel two-tier (forward+backward, then forward-only) is not implemented.
  The review flagged this as a remaining refinement.
- Interaction failures (each kernel compiles alone but the module fails together,
  e.g. duplicate helper symbols) are handled: isolation keeps per-kernel libs, so
  no subset rebuild can re-fail. Not explicitly tested with an injected case.
