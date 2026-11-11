**Verdict:** isolating the failure is the right idea, but step 3 is a mistake. Rebuilding a "good-subset" module costs one extra compile, can fail on its own, and pushes you to change the module's kernel set. After step 1 you already hold a working metallib for every good kernel. Keep those libraries and map each good kernel to its own library handle. Never rebuild a subset.

## Q1: Is the subset/single-kernel build sound?

**Function emission is mostly fine.** `ModuleBuilder` emits only functions reached through `adj.build()` from the kernels it builds. A bad `@wp.func` that only a refused kernel calls will not be emitted. Generic funcs work per overload: an `f(float32)` and an `f(float64)` overload are separate emissions.

**Plan gap: interaction failures.** Some modules fail as a whole while every kernel compiles alone. Two causes:
- Two kernels each pull in a helper with the same mangled name but different bodies (closures, or different captured constants), giving a redefinition.
- A df64 helper that gets emitted once per user instead of being guarded.

For such a module, step 1 marks every kernel good, step 3 rebuilds the same module, and it fails again. Your plan only handles "all bad", not "none bad". The same thing happens with a failing pair: A+B fails together, A and B each pass alone, so the subset fails. Per-kernel libraries avoid this class of failure completely.

**Verify that mangled names don't depend on sibling kernels.** `get_kernel_hooks` looks up `kernel.get_mangled_name()`, and that name comes from the hash the hasher computed. Your latest commit fixed option hash perturbation. If any part of that perturbation depends on the module's contents (for example "module has any df64 kernel"), a single-kernel hasher gives a different hash suffix. Then the function lookup in the isolated library returns null.
- Test: a module with a df64 kernel A and a pure-float32 kernel B. Isolate B and compare its mangled name with the one from the full build.

**`ModuleHasher(module)` iterates `module.live_kernels`.** You'll need a constructor that takes a kernel list, not a fake module.

**Pass the exact resolved `options` object from the failed build** to every isolated builder. Don't re-read module or global config: you just closed a global-config leak, and re-reading would reopen it.

**Scope limit.** This only isolates MSL-compiler errors. `ModuleBuilder.__init__` builds every kernel before any MSL exists, so a `WarpCodegenError` in kernel B still kills A. The plan's claim that this "guarantees the property regardless of type prediction" is overstated. Say "MSL-compile faults only".

## Q2: Caching

**Don't persist refusals.** Metal compile failures can be transient: MTLCompilerService crashes or XPC interruptions under memory pressure. If a refusal gets cached, for example as fabricated markers in a cached `.metal` file, a good kernel stays refused across processes.
- Failing case: compiler service dies during A's isolated compile. A is cached as bad and stays refused forever.

Refusals also depend on the OS/Metal toolchain version, which the cache key doesn't include.

**Cache only successes, content-addressed.** Each single-kernel hasher's `module_hash` is a natural key for a per-kernel artifact, so it can't collide with the full module entry. Never write the subset result under the full module's hash. If you do, the next process loads that entry as if the module were complete.

**On the next process:** the full compile fails again, isolation runs again, and good kernels hit their per-kernel cache entries. You pay the cost only on the failure path.

**Don't create temporary `Module` objects.** Use a builder plus a hasher only. Modules register globally by name, and that is your leak.

## Q3: Injecting refusals

**Don't fabricate markers.** Give `ModuleExec` an explicit `refused: dict[mangled_name, error_str]` at construction. Check it in `get_kernel_hooks` before the native lookup. Otherwise a missing symbol returns a null hook, which may get cached as `None` and fail obscurely.

**Key by mangled name, not `kernel.key`.**
- Failing input: a generic kernel `k(a: wp.array[Any])` that calls `wp.sin(a[i])`, with `wp.overload(k, [wp.array[wp.float32]])` and `wp.overload(k, [wp.array[wp.float64]])`.
- Expected: the float32 launch works and the float64 launch raises.
- With keying by `key`: both raise.

Check whether the existing marker parser already has this bug.

**Truncate the compiler error** in the `RuntimeError`, and name the overload's argument types.

## Q4: Downstream assumptions

**Do not touch `module.kernels`/`live_kernels` or `module.hashers[block_dim]`.** If the subset hasher replaces the stored hasher, the next `load()` sees a hash mismatch, rebuilds, fails, and isolates again on every launch.
- Test: launch good kernel A twice and assert exactly one compile.

`kernel.module` back-references stay valid as long as you only change what `ModuleExec` holds.

**Cost under `mark_modified`.** Defining a new kernel later in the same Python module reloads the module, so the full compile fails again and N isolated compiles run again. Per-kernel caching keeps this bearable. Without it, N compiles happen on every module change.

## Q5: Always one metallib per kernel?

No as the default. Every compile re-parses the full Warp MSL prelude (builtin.h plus df64), and `newLibraryWithSource` has no precompiled header. On a cold load that is N times the prelude cost, plus N cache entries, plus N libraries on every success path.

Use the hybrid: compile the whole module first, and on failure use per-kernel libraries.

Bisection or mapping error line numbers to kernels is cheaper in theory but unreliable. df64 template errors point into builtin.h lines, not the kernel. Per-kernel compiles are simpler and give each kernel its exact error.

## Q6: Interaction with `enable_backward`

**The existing module-wide forward-only retry has a hole.** If the module compiles forward-only, it strips backward from every kernel, including ones whose adjoint was fine.
- Input: kernel A is pure float32 with a working adjoint, kernel B is df64 with a broken adjoint.
- Expected: `tape.backward()` through A works.
- Actual: A loses its backward too.

Verify what an adjoint launch does when the backward symbol is missing. If it is a no-op that leaves grads at zero, that is silent wrong gradients, a correctness bug and not a refusal.

**Per-kernel tiers:** try forward+backward first. On failure, try forward-only and mark that kernel "backward refused" with the error. On failure again, mark it bad.

**Ordering options:**
- Full → per-kernel two-tier, and drop the module-wide forward-only step. This is correct but costs up to 2N compiles.
- Keep module-wide forward-only as a fast path only when every kernel's backward fails. You can only know that after isolation, so the first option is simpler.

The plan doesn't say how "the adjoint case" is detected. If it relies on parsing the error string, it is fragile. Detecting it structurally, by retrying, is safer.

## Minimum tests before merging

1. **Overload test:** the float32/float64 overload pair from Q3.
2. **Name stability:** a good kernel's mangled name is the same in the full build and the isolated build.
3. **Interaction failure:** an injected one where each kernel compiles alone but the module doesn't. It must not loop.
4. **Two launches:** launching a good kernel twice after isolation triggers exactly one compile.
5. **Mixed backward:** a pure-float32 kernel with a working adjoint next to a df64 kernel with a broken adjoint. The first still differentiates; the second refuses backward with a clear error.
6. **Cache state:** a simulated transient compile failure is not persisted.