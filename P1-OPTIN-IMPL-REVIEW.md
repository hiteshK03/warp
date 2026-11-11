# Review: df64 opt-in step

**Verdict:** Making df64 opt-in is the right shape. The `metal_crt.h` gate is correct as far as it goes. The test with module A passing shows that `WP_METAL_DF64` is emitted before the include, so the define takes effect. But the headline claim is "error mode is behaviourally identical to pre-P1, no global leak", and the evidence does not support it. Two holes are real (items 1 and 2 below), and three more are unverified paths that the claim depends on.

## 1. The "pre-P1 baseline" is probably not pre-P1 (methodology bug, highest priority)

The matrix docstring says the baseline came from restoring the "metal-1.17 headers+context" overlay. If P1 also touched `codegen.py`, `types.py`, `builtins.py`, or anything else outside those files, both runs shared those changes. The differential then cannot see them.

- All four refusals match for a trivial reason: the new error branch was written to return the literal `"float64"`. Matching that string proves the branch was copied, not that the rest of the pipeline is unchanged.
- **Fix:** run `git worktree add ../prep1 <pre-P1 sha>` and do a full clean build there. Then run the matrix from that tree, not from an overlay.

## 2. The global config is still a global leak, and it depends on ordering

`Module.__init__` captures `getattr(warp.config, "metal_float64", "error")` at construction time.

**Breaking input (leak):**
1. A library does `wp.config.metal_float64 = "df64"` at import.
2. The user's script runs `import lib`, then defines `@wp.kernel def k(out): out[0] = wp.float32(wp.float64(1.0)/wp.float64(3.0))`.
3. The user's `__main__` module is created after the import, so it inherits `"df64"`.

- Expected: k is refused.
- Actual: k launches under emulation, with only a log line.

This is exactly the leak the module option was meant to prevent. `p1_optin_scope.py` never touches `wp.config`, so it cannot catch it.

**The reverse is also broken:**
1. Define the kernel at the top of the script.
2. Then set `wp.config.metal_float64 = "df64"`.
3. Launch.

- Expected (per the config docstring, "supplies the default"): k launches.
- Actual: k is refused, because the default was captured too early.

**Fix:** either drop the config fallback entirely (module option only), or resolve it lazily in `resolve_options` the way `mode` is. Dropping it is better for the stated safety goal.

**Also:** the value is not validated. `"DF64"` or `True` silently means error mode. That fails safe, but it should raise.

## 3. Warm cache: the warning disappears, and the refusal path was never tested warm

The warning is emitted inside codegen. On a warm kernel cache, `Module.load` finds the binary and skips codegen.

**Breaking input:**
1. Run `p1_optin_scope.py` twice without clearing the cache.
2. Expected on the second run: the df64 warning fires.
3. Actual: no warning. Module A silently runs emulated code.

The same question applies to `_metal_unsupported_reason`, if it only runs on the generated source.

- **Test:** run `p1_fp64_mode_matrix.py error_mode` twice back to back and diff the two JSON files.
- **Failure would look like:** on the second run, a refused kernel raises a different exception (a missing function or KeyError instead of the clean `RuntimeError`), or launches.
- The result file admits this case was not done, so "identical" is unproven.
- **Fix:** move the warning to load/launch, keyed on module hash.

## 4. `#else` branches of `WP_METAL_DF64` have been dead code since P1

Before this commit, `WP_METAL_DF64` was defined unconditionally. Any `#if defined(WP_METAL_DF64)` / `#else` split in `builtin.h`, `vec.h`, `mat.h`, `quat.h` or elsewhere has therefore only had its df64 side compiled since P1. Error mode is now the first time the `#else` side builds.

- **Check:** `grep -rn WP_METAL_DF64 warp/native`. For each `#else` branch, compile a float32 kernel that instantiates the code it guards.
- **Breaking input, if any `#else` side was edited or deleted in P1:** a float32 kernel using that builtin. Expected: it compiles. Actual: MSL compile error, which takes down the whole module, including the other float32 kernels.
- The 5-case matrix exercises one trivial float32 multiply, so it cannot catch this.

## 5. The new option changes the module hash on every backend

`metal_float64` now sits in the options dict of every module, on CPU and CUDA too.

- Every existing kernel cache is invalidated on upgrade.
- The review's own criterion, "module hash identical in error mode", is impossible to meet by construction.
- In df64 mode, CUDA and CPU modules get different hashes for byte-identical code.

**Fix:** contribute the option to the hash only when it differs from the default, or only for Metal builds.

## 6. The warning text understates the known defects

The warning says values "within an ulp of FLT_MAX may overflow spuriously". The measured defects are worse:

- **Wrong-sign infinity:** `wp.float64(3.4028234e38)/wp.float64(1.9999998)` gives `-inf`; expected `+1.70e38`.
- **Spurious overflow far from an ulp:** `sqrt(FLT_MAX)^2` gives `inf`.
- **Not mentioned at all:** `isnan`/`isinf` can be folded away by fast math, and `copysign(x, -0.0)` returns `+|x|`.

Either fix the guard (the previous review gave a fix: if `naive` is finite and `r` is not, return `naive`) or state "can return wrong-sign infinity". The current text misleads the user who is opting in.

## 7. df64 mode: one bad kernel breaks the whole module (not new, still open)

1. In an opted-in module, define float32 kernel A and kernel B with `wp.sin(wp.float64(x))`.
2. If the df64 gate does not refuse B before MSL compile, compilation fails because there is no `sin(float64)` overload.
3. Expected: B is refused and A runs. Actual: neither runs.

## 8. `module="unique"` inheritance is untested

1. Call `wp.set_module_options({"metal_float64": "df64"})` in a file.
2. In the same file, define an fp64 kernel with `@wp.kernel(module="unique")`.

If unique modules do not copy the parent's options, the kernel is refused. That fails safe but will surprise users. It needs one test either way.

## Fine as-is

- The include gate in `metal_crt.h`.
- `set_module_options` going through `mark_modified`, so toggling the option within a process forces a reload.
- The opted-in vs not-opted-in split shown by the scope test.
- `log_warning` instead of `warnings.warn`.

**Minor:** `config.metal_float64` uses `#` comments where AGENTS.md asks for an attribute docstring.

## Minimum before calling this verified

1. A full-tree pre-P1 worktree baseline.
2. An MSL text diff (comments/hash stripped) over a corpus that includes: struct with a float64 field, a generic `@wp.func` at float64, an unused float64 func, `sin(float64)`, a mixed module, and one float32 kernel per `#else` branch.
3. The matrix run cold then warm.
4. The `wp.config` ordering cases from item 2.