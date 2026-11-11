# df64 is now opt-in; default restores pre-P1 behaviour (VERIFIED)

## Design (per the plan review)

- **Module option**, not just a global knob: `wp.set_module_options({"metal_float64": "df64"})`.
  `wp.config.metal_float64` supplies only the default (`"error"`). Module-scoped so a
  library cannot enable emulation for its users' kernels, and hashed via
  `resolve_options` so the kernel cache cannot serve a df64 binary to an error module.
- **Native gating**: `metal_crt.h` no longer defines `WP_METAL_DF64` unconditionally;
  codegen emits it per module only when opted in, and only then includes
  `metal_float64.h`. In error mode the df64 typedef/ops do not exist, so the
  generated MSL matches the pre-df64 backend.
- **Gate**: `_metal_unsupported_reason` short-circuits to the original
  `float64|double` refusal when the module has not opted in.
- **Warning**: `log_warning` (not `warnings.warn`, which user filters can silence),
  once per module, naming the module and stating the precision/range contract and
  the known defects. Fires only for modules that actually use float64.

## Verification (not asserted)

Captured a real pre-P1 baseline by restoring the metal-1.17 overlay files, then ran
the same 5-case matrix against the new build in error mode
(`bench/p1_fp64_mode_matrix.py`, results in `bench/fp64_mode_*.json`):

| case | pre-P1 | error mode |
|---|---|---|
| internal_fp64 | refused (float64) | refused (float64) — MATCH |
| mat33d_inverse | refused | refused — MATCH |
| fp64_array_abi | refused | refused — MATCH |
| fp64_scalar_arg | refused | refused — MATCH |
| plain_float32 | launched | launched — MATCH |

Differential: **identical on all 5**, including exception type and message.

Module scoping (`bench/p1_optin_scope.py`): module A with the option launches and
returns 0.33333334; module B without it is still refused; the warning names module
A only. **No global leak.**

## Correction to an earlier claim

The pre-P1 baseline shows `fp64_scalar_arg` **was refused** before P1. So the
"silent garbage: scalar fp64 arg reads 1.875 instead of 1.0" hole I reported was
**introduced by my gate opening**, not pre-existing. Error mode closes it again.

## Still outstanding (from the review, not done)

- The full 15–20 kernel differential corpus with **MSL text diffs** and module-hash
  comparison (I verified behaviour on 5 cases, not generated-source identity).
- Cold vs warm kernel cache, and the knob-toggle sequence within one process.
- df64's own defects remain when opted in: wrong-sign infinity near FLT_MAX,
  signed-zero loss, floored `mod`, public isnan/isinf using foldable FP predicates,
  ambiguous int64/float16 constructors, no transcendental overloads.
