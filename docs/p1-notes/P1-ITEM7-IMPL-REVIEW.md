# Review: item 7 per-kernel Metal fault isolation

**Verdict:** the per-kernel-lib design is right. The implementation has two likely correctness bugs, a probable leak, and a test suite that doesn't exercise most of what `RESULT.md` claims. Not mergeable as is.

## Blocking issues

**1. Generic kernels break isolation, and test 1 never reached isolation.**
- In `_metal_isolate_kernels`, `name = kernel.get_mangled_name()` runs outside the `try`. `live_kernels` holds the generic kernel itself, and its `hash` is `None`. Only its overloads get hashed. So this line raises the "Missing hash" RuntimeError, and the whole load dies.
- Even if it didn't raise, `ModuleHasher([generic])` puts every overload into one translation unit. That lib gets keyed by the generic's name, but `get_kernel_hooks` looks up each overload's own mangled name. So per-overload refusal can't work. This is exactly the failure the review described in Q3.
- **Failing input:** a df64 module containing `g(a: wp.array(dtype=Any), o: wp.array(dtype=Any))` that computes `o[i]=a[i]*2`, a float32 overload of `g`, and `k_bad` that calls `wp.sin(wp.float64(0.5))`. Launch the f32 overload.
  - Expected: it runs.
  - Actual: a "Missing hash" RuntimeError from inside isolation, or "no compiled Metal library after fault isolation".
- **Why test 1 passed:** its f64 overload is refused earlier by the float64-ABI marker gate. So the whole module compiled, isolation never ran, and the f64 overload was never launched. `RESULT.md`'s line "refusal is per mangled name, not per key" is unverified.
- **Fix:** expand overloads into separate units (skip `adj.skip_build`), and build one hasher per overload.

**2. The per-kernel libs probably leak.**
- The isolated `ModuleExec` is created with `handle=None`. Upstream `ModuleExec.__del__` is guarded by `if self.handle is not None:`. If the fork kept that guard, the loop that unloads `metal_per_kernel_libs` never runs, and every good kernel's metallib leaks on every reload.
- If there is no guard, `wp_metal_unload_library(None)` runs first. It either passes NULL to native code (crash risk if the native side does `CFRelease`) or raises `TypeError`. The `except (TypeError, AttributeError)` then skips the per-kernel loop, so the libs leak anyway.
- **Test:** isolate a module, `del module.execs[...]`, `gc.collect()`, then check a native library counter or Instruments. Expected: N unloads. Likely actual: 0.
- **Fix:** unload the per-kernel libs outside that guard and outside that `try`.
- **Second leak:** any exception partway through the loop (such as the generic-kernel raise in #1) or in `ModuleExec(...)` leaves the already-loaded libs with no owner. Wrap the loop in `try/except` and unload `libs` on failure.

**3. Mangled-name stability is assumed, never checked, and test 2 is missing.**
- The docstring of `p1_iso_tests.py` lists test 2, but the file doesn't contain it.
- `ModuleHasher([kernel], options)` mutates `kernel.hash` (upstream sets `kernel.hash = self.hash_kernel(kernel)`). `name` is captured before that mutation, while codegen emits `{key}_{new_hash}_metal_forward` and launch looks up the new `get_mangled_name()`.
- If any part of the hash depends on the set of sibling kernels, the lib is stored under the old name and launch looks up the new one. Every good kernel would then fail with "no compiled Metal library", and `module.hashers` would be stale for other devices.
- **Fix:** after building, assert `kernel.get_mangled_name() == name`. If they differ, refuse loudly. Also add the actual test: in a module with df64 kernel A and f32 kernel B, compare B's mangled name from the full build with its name after isolation.

**4. The forward-only retry interaction is untested, and behavior depends on unrelated siblings.**
- Test 4 doesn't cover the review's scenario. Its bad kernel fails in the *forward* pass (`sin(f64)`), so the module-wide forward-only retry also fails and isolation runs with backward on. The scenario that matters is kernel B with a working forward and a broken df64 adjoint.
- Three cases:
  - **{A clean, B with a broken adjoint}:** the forward-only retry succeeds and A loses its backward. This is the hole the review flagged in Q6, still open.
  - **{A, B, C=`sin(f64)`}:** forward-only fails because of C. Isolation then builds B with forward+backward, which fails, so B is refused entirely, *forward included*. Expected: B's forward runs and its backward is refused. Actual: launching B raises.
  - **Net effect:** whether A keeps its backward and B keeps its forward depends on whether an unrelated kernel C exists. That makes the behavior non-monotonic, which is a correctness smell.
- **Verify which `options` isolation gets after the retry ran.** If the retry rebinds `options` with `enable_backward=False`, isolation builds forward-only and A silently loses its backward. If the retry is gated on matching the error string, that's the fragile detection the review warned about.
- **Fix:** add the per-kernel two-tier build (forward+backward, then forward-only with a "backward refused" mark). Also run isolation *instead of* the module-wide forward-only retry whenever the module has more than one kernel.

## Non-blocking but wrong

**5. The "all kernels refused" error isn't the original error.** `runtime.get_error_string()` is read after isolation, so it holds the last isolated compile's error. If the last kernel failed at codegen, it's a stale string from an earlier compile. Capture the whole-module error before calling `_metal_isolate_kernels`.

**6. The isolated `ModuleExec` drops `metal_backward_unsupported` and `metal_backward_error`.** Only `metal_unsupported` is re-parsed.
- **Failing input:** a tile kernel marked `wp_metal_backward_unsupported` in the same module as `k_bad`, then `tape.backward` through the tile kernel.
  - Expected: the clear "backward unsupported" error.
  - Actual: the backward symbol lookup returns null, and you get the generic "Failed to find backward kernel" error, or worse.
- Parse both markers from each isolated source.

**7. `if refused:` tests truthiness.** If the native error string is empty, the refusal is skipped and the user gets "no compiled Metal library after fault isolation", with no compiler message. Use `if name in self.metal_refused`. The error also doesn't name the overload's argument types, which the review asked for.

**8. The `except Exception` around codegen is effectively dead code.** A codegen error would already have killed the full build before isolation could run. The `RESULT.md` claim about codegen-error isolation describes a path that can't happen in practice. It also catches hasher bugs that should be loud. Narrow it or delete it.

**9. No caching at all.** Every process and every `mark_modified` pays for a failed full compile plus N prelude-heavy compiles. The review recommended caching successes by per-kernel hash, and `RESULT.md` doesn't mention skipping that. A single-kernel module also compiles the same source twice; short-circuit when there is only one live kernel.

**10. Early `return` skips the normal path's tail.** Check that nothing after the normal metal branch (timers, meta writes, any bookkeeping) is required. Nothing in the diff shows it isn't.

## Test-suite problems

- `p1_iso_tests.py` only prints; it asserts nothing. It isn't `unittest` (AGENTS.md requires `unittest`) and isn't in `warp/tests`, so the review's minimum tests are not in CI.
- Test 3's "exactly one compile" check only shows that `execs` holds an object. Count calls with a monkeypatched `wp_metal_load_library` or `_metal_isolate_kernels` instead.
- Not tested at all: name stability, the injected interaction failure, a transient failure not being persisted, an f64 generic overload that actually goes through isolation, and B with a forward that works but a broken adjoint.
- `bench/corpus_reg.json` hard-codes an absolute local path (`/Users/viani/dev/...`), and the `P1-*.md` notes are committed at the repo root. Keep both out of the branch.

## Genuinely fine

- Keeping per-kernel libs instead of rebuilding a subset is correct, and handles interaction failures without looping.
- Refusals are not persisted.
- `module.kernels` and `module.hashers` are left untouched, apart from the `kernel.hash` side effect in #3.
- The exact resolved `options` are passed through.
- The success path adds no cost.

**Fix order:** #1, #2, #3 (with real assertions), then #4, then everything else.