# Design review: deterministic float scatter on the Metal backend

**Verdict:** the core idea is sound. If you record each contribution, put the records in a fixed order and reduce them sequentially on the host, the result is deterministic. Your minimal repro shows that. The weak points are elsewhere: the MSL in this diff probably won't compile, the host-pointer handling is unresolved, and graph capture is likely to break things silently. Test the Steel hypothesis cheaply first (doubt 9).

## Likely bugs in the diff

**B1. Pointers and references without an address space (MSL compile error).** These are all written without a qualifier:
- `int64_t* keys`, `T* vals` and `int* count` in `det_scatter_buf_t`
- `int* overflow` and `array_t<int>* counter_targets` in `det_ctx`
- `int* _wp_det_overflow` in the `wp_args` struct
- `det_ctx& ctx` and `det_scatter_buf_t<T>& buf` as parameters of `scatter()`
- `auto& helper = buf;` in `kernel_locals`

MSL requires `device`/`constant`/`thread` on pointer and reference types. Use whatever macro your fork's `array_t` uses.
- **Repro:** compile any `wp.atomic_add(a, i, 1.0)` kernel for Metal with RUN_TO_RUN.
- **Expected:** it compiles.
- **Likely actual:** "pointer type must have explicit address space qualifier".

`det_counter_buf_t` has the same problem, so it breaks compilation even though it is unused.

**B2. `os_log("Warp deterministic scatter overflow")` is not valid MSL.** In `<metal_logging>`, `os_log` is a type with a `(subsystem, category)` constructor. You log with `os_log_default.log(...)`, which needs MSL 3.2 and logging enabled at compile time. The `ctx.debug` check happens at runtime, so this line is compiled in every kernel. Remove it and report overflow only through the overflow flag the host reads.

**B3. `_wp_args->` in the external-constant-params entry.** `metal_args_unpack()` is added to `forward_body` for every Metal kernel, including the case where `forward_func_type == "function"` (`is_external_constant_params_entry`). If that variant doesn't receive `_wp_args`, or receives it as a reference rather than a pointer, you get "undeclared identifier" or "member reference type is not a pointer".
- **Repro:** a deterministic-scatter kernel compiled through that entry path.

**B4. The counter path doesn't fail at a defined point.** You say kernels that consume the atomic's return value are refused, but there is no refusal in the diff. Two things happen instead:
- `metal_args_struct_fields()` still emits counter fields.
- `kernel_locals("metal")` never declares the counter helper, so `WP_DET_COUNTER_OR_FALLBACK` references an undeclared name. The MSL error is loud but cryptic.

Worse: if someone later declares the helper, the macro does `out = 0; cpu_expr;`, which silently falls back to a plain atomic.
- **Repro:** `slot = wp.atomic_add(counter, 0, 1)` with RUN_TO_RUN on Metal.
- **Expected:** a Python `RuntimeError` naming the kernel.

Put the refusal in Python, before codegen.

**B5. Possible clash with `deterministic.h` fallback macros.** If `deterministic.h` defines `WP_DET_*` fallbacks outside its `!__METAL_VERSION__` gate (Metal must get *some* definition today), your header redefines them. Clang only warns about a macro redefinition, and the last definition wins because of include order. That works by accident. Gate one set explicitly.

Also check that the Metal module actually uses `cpu_module_header`. If it has its own header, your `#include` never reaches Metal and the deterministic path compiles out to plain atomics. Your end-to-end test would then catch it as "still nondeterministic", not as an error.

**B6. Kernel cache key.** The args struct changes shape depending on `needs_deterministic`. If the module hash doesn't include the deterministic mode for Metal, a kernel cached with mode off gets reused under RUN_TO_RUN (or the reverse). The host then packs a struct that doesn't match the compiled one.
- **Repro:** launch with mode off, switch to RUN_TO_RUN, launch the same kernel again.
- **Expected:** a recompile or deterministic results.
- **Likely actual:** garbage in the fields after the args, or a GPU fault.

## Your doubts

**1. Thread id.** For RUN_TO_RUN, the ID only has to be stable from run to run, not equal to CUDA's `_idx`. `task_index` is fine provided three things hold:
- **Single dispatch.** The launch is one dispatch whose `task_index` is global. If you split big launches into chunks with a per-chunk offset, keys collide across chunks.
- **No tile kernels.** With `block_dim > 1`, all lanes of a block share one `task_index`. Equal keys from different threads then sort in slot order, and slot order comes from the racy counter, so the result is nondeterministic.
  - **Repro:** a tile kernel with `block_dim=64` where every lane does `atomic_add(out, 0, x[lane])`.
  - **Expected:** identical results run to run.
  - **Actual:** differences around 1e-6.
  - **Fix:** use the linear thread id, or refuse tile kernels.
- **Several adds to one target from one thread.** These produce equal keys. That is still safe, because one thread's counter increments come back in program order, but only if the host sort is stable. LSD radix sort is stable. Don't swap in `std::sort`.

**2. Key packing.**
- 32-bit tid is safe if the host launch rejects `dim.size > 2^32`, as CUDA does. Add that check; it isn't there yet.
- `dest_flat_idx` must be the logical row-major index, and the host reduce must map it through the destination's strides, not its byte layout. Test with a strided view (`a[::2]`) as the destination.
- A negative or out-of-bounds index gives a negative key, which sorts first. The host then writes `dest[-k]`, which is silent host memory corruption. Bounds-check in the host reduce and raise.

**3. Binned accumulator.** You're right that it isn't needed for RUN_TO_RUN. A fixed record order plus a sequential scalar add is bit-exact, as long as you don't compile the reduce with `-ffast-math` (NumPy `np.add.at` is fine). Two caveats:
- Don't let GPU_TO_GPU succeed silently on Metal. The bits won't match CUDA's. Refuse it or document that it falls back to RUN_TO_RUN.
- To maximise the chance of matching the CPU trajectory, accumulate exactly like the CPU backend: `dest += v`, one record at a time in (tid, k) order, starting from the destination's value before the kernel. Don't sum each group and then add it to dest; that differs in the last bit from CPU's serial order.

**4. Vector and composite values.**
- Get the record stride from Warp's Python dtype size, not from any MSL type. This only goes wrong if your Metal `vec3` maps to MSL `float3`, which is 16 bytes, versus 12 on the host; that would corrupt the vals stride.
  - **Repro:** tid `i` scatters `vec3(i, 2i, 3i)` to dest 0.
  - **Expected:** `(Σi, 2Σi, 3Σi)`.
  - **Test the same way for `spatial_vector`** (6 floats, 24 bytes).
- float64 targets: if the fork turns `double` into df64, `value_ctype` must be the df64 struct, and the host reduce must use df64 arithmetic or convert explicitly. Otherwise you're adding pairs of floats as raw bits.
  - **Repro:** `atomic_add` on `wp.array(dtype=wp.float64)` under RUN_TO_RUN.
- Avoid moving the value payload through the sort at all: sort a permutation of `int32` slot indices, or skip the sort (point 9).

**5. Pointers inside the args struct.** This is the biggest open question.
- Host pointers are not GPU addresses. `MTLBuffer.contents` is not `gpuAddress`.
- Nested pointers need translating. The existing translation probably only rewrites `array_t.data` at known offsets. The `keys`, `vals`, `count` and `overflow` pointers sit inside nested structs, so they must be translated too.
- Residency: each of those buffers also needs `useResource:usage:` or a residency set. Without it, an indirectly referenced buffer faults or reads zero.
- **Repro:** one launch with dim=1 and one scatter, then read `count` on the host.
- **Expected:** 1.
- **Likely actual:** 0, a GPU page fault, or a write into some random host address.

**6. Capacity and overflow.**
- A static `records_per_thread` can't bound an `atomic_add` inside a loop whose trip count is only known at runtime.
  - **Repro:** a kernel with dim=1 that does `for k in range(n): wp.atomic_add(out, 0, 1.0)` with `n=4` at runtime.
  - **Expected:** 4.0.
  - **Actual:** 1.0 if records_per_thread is computed as 1. The extra records are dropped and only the flag is set.
- So the host must check the overflow flag after every launch and either raise or re-run with a larger capacity.
- `count` keeps growing past capacity, so clamp it before reducing.
- Look at how the CUDA path handles loops. If it refuses them, refuse them too.

**7. Backward.** Leaving it out of scope is acceptable only if it fails loudly. If the adjoint body emits the deterministic macros without a `det_ctx`, you get a compile error, which is ugly but loud. If it emits plain atomics, you get gradients that change from run to run, silently. Raise in Python when an adjoint launch has scatter targets under RUN_TO_RUN on Metal.

Also check where the det fields sit in the args struct when the adjoint fields are appended. The host packing offsets have to match the backward struct as well.

**8. Graph capture.** This is the most likely silent failure. Three specific failure modes:
- **Host op runs once.** If `_host_call` executes immediately during capture instead of recording a host op, the reduce runs once against empty buffers, and replays never reduce. Steel bodies would then get zero contact wrench, which looks like "it settles".
- **Count never resets.** `count` must be zeroed inside the captured sequence. If it is zeroed only when the buffers are allocated, the first replay works and the second starts at the old count and overflows immediately.
  - **Repro:** capture one scatter launch, replay it twice with different inputs, and compare each replay with an eager launch.
- **Buffer lifetime.** Scatter buffers allocated during capture must be owned by the graph. If they are freed after capture, replays write into freed buffers.

Even if all three are fixed, each deterministic launch forces a commit, a wait for completion, a host reduce, and a new command buffer. That throws away most of what the graph buys. It's acceptable for an experiment; just know it.

Also, in eager mode the reduce must wait for the command buffer to complete before reading. Otherwise it reads `count == 0`.

## 9. Is the Warp-side port the right call?

Not as the first step. Your decision rule says (b) decides it, so test (b) cheaply before doing any of the codegen, args-packing, residency and capture work.

**No sort is needed for a sequential host reduce.** Give each record a deterministic slot `tid*rpt + k` instead of an atomic counter slot, then reduce sequentially over the slots in order. That is already (tid, k) order. For each destination it gives exactly the same bits as sorting by (dest, tid) and reducing in order. It costs O(n), needs no sort and no slot atomic.

**Prototype:** hand-edit Steel's three scatters (`body_wrench`, `world_diag`, `propagation_body_qd`) into this form plus `np.add.at`, with no capture. If Steel still doesn't settle, the Warp port solves determinism but not the jitter, which you only learn afterwards.

**Audit every atomic in the solver, not just the float ones.** An integer `slot = atomic_add(count, 1)` used to build per-body contact lists makes the list order nondeterministic, and any float loop over that list inherits it. That is exactly the consumed-return path you plan to refuse. Your bisection only showed the *inputs* to the solver are identical.

**Longer term**, a Newton-side gather (a per-body CSR built once per collide, looping over contacts in a fixed order) is deterministic on every backend, works under graph capture, and has no host round trip. If (b) passes, weigh that against the Warp port. The port is more general but costs a host sync per deterministic kernel and carries all the risks in point 8.