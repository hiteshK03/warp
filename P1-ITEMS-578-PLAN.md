# Plan: fix item 7 (sin(float64) breaks the whole opted-in module) + items 5, 8

## Confirmed problems (measured)

- Item 7: in a df64-opted-in module, a kernel calling wp.sin(wp.float64(x)) has no
  df64 sin overload, so MSL compile fails for the WHOLE module, taking down a clean
  float32 kernel in the same module. Expected: the sin kernel is refused
  individually; the clean kernel runs.
- Item 5: the metal_float64 option sits in every module's options dict, so it
  perturbs module hashes on CPU/CUDA too (measured: CPU hash differs default vs
  df64), invalidating existing caches on upgrade.
- Item 8: module="unique" does NOT inherit the parent's metal_float64 option, so an
  fp64 kernel there is refused even if the parent opted in. Fails safe, but
  surprising.

## Proposed fixes

### Item 7 (priority — correctness/robustness)
The codegen already has a mechanism: METAL_UNSUPPORTED_BUILTIN_PREFIXES +
metal_unsupported_builtins, which replaces a builtin call with a
`metal_unsupported<...>` stand-in and marks the kernel refused (source prefixed
`// wp_metal_unsupported`). Currently the prefix tuple is empty.

Approach: when df64 is active, detect builtin calls whose argument/return type is
float64 AND which have no df64 overload (the transcendentals: sin, cos, tan, exp,
log, pow, atan2, etc. — metal_float64.h defines only add/sub/mul/div/sqrt/abs/
min/max/floor/ceil/rint/mod/copysign). Mark those kernels unsupported so they are
refused individually, leaving sibling kernels compilable.

Question: is it cleaner to (a) populate a float64-specific unsupported set keyed
on (builtin, has-float64-arg), or (b) add df64 overloads for the transcendentals
so they actually work? (b) is more work and each needs a numerically-correct df64
implementation; (a) is honest refusal. I lean (a) for now.

### Item 5
Contribute metal_float64 to the module hash ONLY for Metal builds, or only when
non-default. Need to find where options feed the hash (ModuleHasher / resolve_options)
and whether device is known at hash time.

### Item 8
Decide: should unique modules inherit the parent option? If yes, copy it at unique-
module creation. If no (fail-safe), document it. Lean: document fail-safe, since
unique modules are meant to be isolated.

## Questions for review
1. Item 7: is the metal_unsupported_builtins mechanism the right hook, and how do I
   detect "float64 arg with no df64 overload" precisely at codegen (func.input_types)?
   Will marking it unsupported actually prevent the MSL compile failure, given the
   stand-in substitution?
2. Item 5: where exactly is the option hashed, and is device available there? What
   is the least invasive way to make it Metal-only without breaking df64 cache
   invalidation?
3. Item 8: inherit or document? Any correctness risk either way?
4. Are there other fp64 builtins beyond transcendentals that lack df64 overloads and
   would hit the same whole-module failure (e.g. wp.atan2, wp.pow, wp.log)?
