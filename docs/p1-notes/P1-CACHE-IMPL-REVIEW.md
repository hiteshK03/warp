# Review: per-kernel success caching via isolation record and good-kernel bundle

**Verdict:** the design is reasonable. The implementation has two bugs that break the interaction-failure path and one staleness bug. All three are covered by "gaps" the notes call minor, so treat these as blocking.

## Blocking

**1. Bundle fallback sets `metal_bundle_libs = None`, so `__del__` raises and leaks every library.**
- On an interaction failure, `_metal_build_bundle` returns `(None, {})`. That assigns `bundle_handles = None`, and the code then sets `module_exec.metal_bundle_libs = bundle_handles`.
- In `__del__`, the first branch collects the per-kernel libs and then runs `for lib in self.metal_bundle_libs`. That raises `TypeError: 'NoneType' object is not iterable` before any unload runs. The `try` only wraps the unload call, not the loop.
- Result: every per-kernel lib leaks, and you get "Exception ignored in `__del__`".
- **Repro:** use a record with `bundle_failed=False`. Monkeypatch `Module._metal_build_bundle` to return `(None, {})`. Clear `m.execs`, launch a good kernel, then run `del` and `gc.collect()`.
  - Expected: libs unloaded silently.
  - Actual: `TypeError` in `__del__`, and spying on `wp_metal_unload_library` shows 0 calls.
- **Fix:** use `bundle_handles or []`.

**2. A record with `bundle_failed=True` never loads the good kernels.**
- The per-kernel fallback only runs inside `if good_roots and not bundle_failed:` when `bundle_handles is None`. When the record already says `bundle_failed=True`, the code skips both the bundle and the per-kernel compile of the good kernels.
- Case (a), the common one: refused retries all fail again, so `libs` is empty and the function returns `None`. You then get a failed whole-module compile, full isolation, a failed validation bundle, and the record rewritten with `bundle_failed=True`. That repeats every load, and it is strictly slower than having no cache (two extra failing compiles per refused kernel, plus the validation bundle).
- Case (b): a refused kernel heals, so `libs` is non-empty and the exec is returned. The good kernels are in neither `metal_per_kernel_libs` nor `metal_refused`. Launching one fails with whatever lookup error you get for a missing name, instead of running.
- **Repro, pure interaction case:** two kernels that each compile alone. Wrap `wp_metal_load_library` to return 0 when the source contains both kernel names.
  - The cold run writes `refused={}` and `bundle_failed=True`.
  - The second load (`execs.clear()`) should take the record path with per-kernel libs.
  - Instead the spy shows `_metal_load_with_record` returned `None` and the whole-module path ran again.
- **Fix:** when `bundle_failed` is set, isolate `good_roots` per kernel unconditionally.

**3. `backward_refused` only ever grows, so the record can never be deleted or healed.**
- The success branch runs `new_backward_refused |= backward_refused` unconditionally. B1 deletes only when that set is empty, so any record with a backward refusal is permanent. The record path also never tries the whole module again.
- The fingerprint hashes only `codegen.py`, `context.py` and `build.py`. A fix to any of these does not invalidate the record:
  - `builtins.py`, where df64 builtins and adjoint rules live.
  - The native MSL prelude and df64 headers under `warp/native/`.
  - The `.mm` loader.
  - Any other Metal backend `.py`.
- The module hash covers user functions recursively, not builtin implementations. A dev fork's kernel-cache dir does not change with header edits.
- **Repro:** module with kernel K whose adjoint fails to compile, so the record gets `backward_refused={K}`. Fix the adjoint in the native header and rebuild. Relaunch and run `tape.backward()` through K.
  - Expected: works.
  - Actual: raises backward-unsupported until someone sets `WARP_METAL_REISOLATE`.
- **Fix:**
  - Fingerprint `builtins.py` plus the Metal native headers, or a version string embedded by the native library.
  - Re-test `backward_refused` kernels in the full bundle, or at least don't union stale entries into the record you write.

## Significant

**4. A healed refused kernel compiles twice and leaks a handle every load.**
- In the retry loop, a healed kernel gets `libs[name] = sub_handle` and is then added to `good_roots`. The bundle mapping overwrites `libs[name]` with the bundle handle. The `sub_handle` is in neither dict nor list, so it is never freed.
- In the fallback path it is compiled a third time, which leaks again.
- The comment says the kernel will "rejoin the bundle next time", but the code adds it this time.
- **Repro:** record with `refused={K}` where K now compiles (simulate a transient failure by hand-editing the JSON). Count `wp_metal_load_library` against `wp_metal_unload_library` across load and destroy: there is one more load than unload.
- Also, adding a newly healed kernel to the bundle in the same load risks turning a working bundle into `bundle_failed`, which then hits bug 2.

**5. Refused kernels are retried on every load, which defeats the cache and contradicts the comment.**
- The header comment says retries happen "unless WARP_METAL_REISOLATE is unset and the fingerprint is unchanged". The code always retries.
- Each retry is two failing compiles (full, then forward-only), and failed compiles don't hit Apple's shader cache.
- Your own numbers suggest this is where the time goes: the warm bundle is 46ms, but warm end-to-end is 685ms. Measure it.
- **Recommendation:** trust the record when the fingerprint matches. Retry only under `REISOLATE`, which is the point of keying on the fingerprint.

**6. The device is not in the key, and `compiler_id` does not identify the GPU.**
- The docstring says "GPU architecture", but `platform.machine()` is just `arm64`, or `x86_64` under Rosetta.
- On an Intel Mac, `metal:0` (an Intel integrated GPU) and `metal:1` (an AMD discrete GPU) share the same record, yet MSL support differs per device.
- **Fix:** include the device name or GPU family (`device.name` / `MTLDevice.name`) in both the key and the record.
- If `sysctl` exits non-zero with empty stdout, `osversion` is `""` and no fallback runs.

## Fragile, not currently wrong

- **Handle mapping (full vs forward-only):** correct today. All three cases work: full only (`[full]`), forward-only only (`[fo]`, where `[-1]` is right), and mixed (`[full, fo]`). But the caller re-derives the partition in an O(N²) loop instead of receiving it.
  - If the two partitions ever diverge (for example a different set passed), a full kernel gets mapped to the forward-only lib, and adjoint symbol lookup fails.
  - **Fix:** have `_metal_build_bundle` return a `name → handle` dict.
- **`__del__` dedup:** it uses `id(lib)`. That works only because the same int object is stored in both places. Two ctypes calls returning equal addresses would give distinct objects and a double unload. Dedup by `int(lib)` value instead.
- **Cold-path validation bundle:** it compiles a bundle and then discards it while keeping N per-kernel libs. Use the bundle it just compiled. It is the same work and avoids the extra compile on the cold path.
- **Temp file name:** it uses only the PID, so two threads in one process can collide. Add the thread ID or use `tempfile`.
- **Diagnostic truncation:** diagnostics are cut to 500 chars in the record but 2000 in-process, so the error message differs between cold and warm runs.
- **Record files:** they are never garbage-collected. Every kernel edit leaves a new JSON file.

## Test gaps

- **`test_isolation_record_persisted_and_reused` doesn't clean up its record.** On a second suite run against the same cache, the first launch already takes the record path. The test still passes, so it no longer tests the cold-to-warm transition. Delete the record in `setUp`, or use a per-test cache dir.
- **No test for the mixed full and forward-only bundle through the record path.** Add one: K1 plus K2 with a failing adjoint, plus a bad kernel. On the warm load, `backward` through K1 should work and K2 should raise. This is the only test that would catch a mapping regression.
- **No injected interaction-failure test.** Bugs 1 and 2 both live in that path, so add one, along with a launch/unload balance assertion for bug 4.

## Fine as is

- Atomic `os.replace` write.
- Version, compiler and codegen guards on read.
- Using `kern.osversion` rather than the product version.
- Never mutating `Kernel.options`.
- B1's delete condition, apart from the union in bug 3.