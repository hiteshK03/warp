# Implementation-review fixes (all verified by execution)

## Item 2 — global config leak: CONFIRMED, then removed

Measured before the fix (bench/p1_verify_leak.py):
- 2a LEAK: setting `wp.config.metal_float64="df64"` before a user module is created
  made that module's fp64 kernel LAUNCH under emulation. My earlier "no global
  leak" claim was only true for the module-option path, not the config path.
- 2b ORDER: kernel defined first, config set afterwards -> REFUSED, because
  `Module.__init__` captured the default at construction time.
- 2c: an invalid value ("DF64") silently meant error mode, with no diagnostic.

Fix (the review's preferred option): **drop the config fallback entirely.** The
policy is now module-option only; `wp.config.metal_float64` no longer exists.
`set_module_options` validates the value and raises ValueError otherwise.

Verified after (bench/p1_postfix.py):
- config-only: REFUSED (config is inert; `hasattr(wp.config,'metal_float64')` False)
- explicit module option: LAUNCHED, 0.33333334
- invalid value: ValueError raised
- warm relaunch: consistent

## Item 3 — warm cache swallowed the warning: CONFIRMED, then fixed

Measured: a second process with a warm kernel cache launched df64 with NO warning,
because the notice lived in codegen and codegen is skipped on a cache hit.

Fix: moved the warning to `Module.load`, keyed once per module name. Verified: the
warning now fires on a warm cache (count 1).

## Item 6 — warning text understated the defects: fixed

Now states the measured defects explicitly: wrong-sign infinity from division near
FLT_MAX, spurious overflow of sqrt(FLT_MAX)^2, relative error up to 1.0 for tiny
inputs, lost signed zero, positive copysign(x,-0), floored mod instead of fmod, and
isnan/isinf possibly folded under fast math.

## Bug I introduced and fixed while verifying

Injecting the df64 header AFTER base-header expansion broke df64 mode entirely:
`builtin.h` instantiates `DECLARE_FLOAT_OPS(float64)` / `adj_float64`, which need
`wp::float64` to already exist -> "unknown type name 'float64'". The header is now
injected BEFORE the base headers, still only for opted-in modules.

## Regression status

- error-mode matrix: all four fp64 cases refused, plain float32 launches (unchanged)
- fixedarray suite: 5/5 PASS

## Still open from this review (not done)

- Item 1: the baseline was an overlay-file restore, not a full pre-P1 worktree
  build, so changes outside those files were invisible to the differential.
- Item 4: `#else` branches of WP_METAL_DF64 have only ever compiled on the df64
  side since P1; each needs a float32 kernel exercising it.
- Item 5: the option sits in every module's options dict, so it perturbs hashes on
  CPU/CUDA too and invalidates existing caches.
- Item 7: in df64 mode a kernel using sin(float64) may fail MSL compile and take
  down the whole module instead of being refused individually.
- Item 8: `module="unique"` option inheritance untested.
- Source identity: still NOT achieved (P1a's shared-header edits are unconditional).
