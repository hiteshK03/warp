# Design review: deterministic atomic scatter for the Warp Metal backend

## Goal
Make Warp's deterministic mode (DeterministicMode.RUN_TO_RUN) actually work on the
Metal backend, so float atomic_add scatters produce bit-identical results run-to-run.
Today the deterministic scatter-sort-reduce lowering is CUDA-only; on Metal it is a
no-op (nullptr buffers), so float atomics stay non-deterministic.

## Measured motivation (verified by execution, not speculation)
- The Newton "Steel" sleeping demo on Metal is non-deterministic run-to-run: two
  identical runs diverge by 1.5+ meters; awake-body count varies (11 vs 25). CPU is
  bit-identical across runs and settles (all bodies sleep).
- Bisection: frame-0 collide and steps 1-81 are bit-identical on Metal; first
  divergence at substep 82 when a new contact forms. Contacts going INTO the solver
  are identical; the solver.step OUTPUT differs. So the non-determinism is in the
  solve's float atomic scatters, not collision.
- Steel scatters contact contributions with float wp.atomic_add onto shared buffers:
  body_wrench (spatial_vector), world_diag (float), propagation_body_qd (float).
  Float addition is non-associative; Metal GPU thread order varies run-to-run.
- Minimal repro confirmed: a 4096-thread scatter into 64 destinations via plain
  wp.atomic_add differs by ~5.7e-6 between runs on Metal. The SAME scatter done as
  record -> host sort by (dest<<32|tid) -> host fixed-order reduce is BIT-IDENTICAL.
- Precision is NOT the issue: CPU with float32 friction moments still settles. So
  softfloat would not help; deterministic ordering is the fix.

## Existing CUDA mechanism (what I'm porting)
warp/_src/deterministic.py + warp/native/deterministic.h + deterministic.cu:
- Scatter atomics (return value unused): each thread records (key=dest<<32|tid, value)
  to a temp buffer via an int atomic counter slot. After the kernel, CUB radix-sorts
  records by key and reduces-by-key in fixed order into the destination.
- Consumed-return counter atomics (slot = atomic_add(counter,...)): two-pass execution.
- The whole deterministic.h is gated #if !defined(__METAL_VERSION__); the active
  WP_DET_SCATTER_OR_FALLBACK macros are gated #ifdef __CUDA_ARCH__. So Metal currently
  falls through to plain atomics.
- kernel_locals(device) emits real det buffers for "cuda", nullptr no-op buffers otherwise.
- launch_deterministic() is entirely CUDA-coupled (wp_cuda_launch_kernel, stream.cuda_stream,
  CUB sort/reduce via wp_deterministic_sort_reduce_device).

## My Metal design (the part to critique)
Key enabler: Metal arrays live in UNIFIED memory, and Warp already runs radix_sort_pairs
on the HOST for Metal (warp/_src/utils.py _runs_on_host / _host_call; sort.cpp
radix_sort_pairs_host supports int64 keys + arbitrary value_size). So I do NOT need to
port CUB or write a Metal sort. Plan:
1. MSL scatter record: new warp/native/deterministic_metal.h defines det_scatter_buf_t,
   det_ctx (layouts matching the CUDA ctypes structs), and scatter() that allocates a slot
   with wp_metal_atomic_add(buf.count, 1) (Metal int atomic) and writes key+value. Same
   key packing as CUDA: (dest_flat_idx<<32) | (thread_id & 0xFFFFFFFF).
2. Codegen: include deterministic_metal.h for Metal; kernel_locals("metal") emits REAL
   scatter buffers using task_index as the deterministic thread id (Metal has one thread
   per task, no grid-stride loop, so task_index == the linear thread id). The det params
   (phase, debug, overflow, scatter bufs) are appended to the Metal wp_args struct and
   unpacked at the top of the kernel body before det_ctx is declared.
3. Host launch: a Metal path that allocates scatter buffers, packs them into the args
   struct, launches the Metal kernel, then HOST sorts records (reuse radix_sort_pairs_host)
   and HOST reduces in fixed sorted order into the destination (unified memory).
4. Scope: ONLY the scatter/reduce path (return value unused). The consumed-return counter
   two-pass path and store-guarding stay CUDA-only; kernels needing them are refused on
   Metal in deterministic mode. Steel's scatters are all return-value-unused.

## What I've implemented so far (in this diff)
- deterministic_metal.h (new): the MSL structs + scatter() + active WP_DET_* macros.
- deterministic.py: kernel_locals("metal") real-buffer branch; metal_args_struct_fields()
  and metal_args_unpack() helpers.
- codegen.py: include deterministic_metal.h in cpu_module_header; append det fields to the
  Metal wp_args struct; unpack det params before kernel_locals in forward_body.
- NOT yet done: the host Metal launch path (launch_deterministic is still CUDA-only) and
  the host sort/reduce. So nothing is wired end-to-end yet.

## Specific doubts I want attacked (numbered)
1. Thread-id determinism: I use task_index as det_ctx.idx. On Metal one thread per task,
   task_index in [0, dim.size). Is task_index guaranteed to equal the linear thread id the
   CUDA path uses (_idx), so the sort order is meaningful and stable? Any grid-stride or
   block-dim subtlety on Metal that breaks this?
2. Key packing collisions: key = dest<<32 | tid. If dim.size > 2^32 the tid truncates; the
   CUDA path rejects >2^32. Steel launches are small. Is 32-bit tid safe for Metal, and is
   the dest flat index computed the same way (row-major strides) as CUDA?
3. Host sort/reduce correctness: I plan to sort records by the packed int64 key then reduce
   consecutive equal-dest records in order. The CUDA path uses a "binned float accumulator"
   for GPU_TO_GPU reproducibility across different CUB reduction trees. For RUN_TO_RUN on
   Metal with a FIXED host sort + sequential reduce, is a plain sequential float reduce
   sufficient for bit-exact run-to-run, or do I still need the binned accumulator? I claim
   plain sequential is enough because the order is fixed; attack this.
4. Spatial_vector / multi-component values: Steel scatters wp.spatial_vector (6 floats) and
   vec3. The scatter buffer vals are typed T; the host reduce must reduce component-wise.
   Does my plan handle non-scalar T correctly, and does the radix sort's value_size handle
   sizeof(spatial_vector)=24 bytes?
5. Struct layout / address spaces: det_scatter_buf_t has int64_t* keys, T* vals, int* count.
   On Metal these are device pointers passed in the constant wp_args struct. Is passing
   raw device pointers in the args struct + the pointer-offset translation (Metal rewrites
   host pointers to GPU VA) going to work for the scatter buffer pointers, or do they need
   special residency/import handling? The keys/vals/count buffers are warp arrays allocated
   on the Metal device.
6. Counter overflow / capacity: scatter capacity = dim_size * records_per_thread. If a thread
   does multiple atomic_adds to the same target (Steel does), records_per_thread must count
   them. The CUDA codegen computes scatter_records_per_thread statically. Does my Metal path
   reuse that correctly, and what happens on overflow (I set an overflow flag)?
7. Backward pass: I only wired forward. Steel runs enable_backward=False mostly, but some
   kernels may need backward. Is leaving Metal backward deterministic unsupported acceptable
   for the stated scope, and will it fail loudly rather than silently produce wrong grads?
8. Graph capture: Steel uses wp.ScopedCapture / graph replay. The host sort/reduce runs on
   the host between dispatches. Does my plan work inside a Metal graph capture (the host
   sort must be recorded as a host_op via _host_call), or does it break capture? This is a
   real risk — Steel captures the whole simulate() into a graph.
9. Is porting the scatter path even the right call vs. a Newton-side fixed-order reduction?
   I chose the Warp-side fix for generality. Attack whether the complexity/risk is justified.

## Decision rule (pre-committed)
Keep the Warp-side Metal deterministic scatter ONLY IF: (a) a minimal scatter kernel is
bit-identical run-to-run on Metal with RUN_TO_RUN, AND (b) the Steel demo becomes
run-to-run deterministic AND settles (bodies sleep) like CPU. If (b) fails because the
deterministic Metal trajectory lands in a different chaotic basin than CPU, that is a
separate finding (determinism achieved, parity not) — report it honestly, do not claim the
jitter is "fixed" unless bodies actually sleep.
---

## PIVOTAL FINDING (after Claude review, verified by execution)

The Claude review's doubt #9 ("test the Steel hypothesis cheaply before the Warp port")
led to discovering that **Steel already has its own deterministic mode**:
`SolverSteel(deterministic=wp.DeterministicMode.RUN_TO_RUN)` sets `_ordered_rows=True`,
which reserves constraint rows in a FIXED order (joint rows in articulation order, contact
rows in contact-buffer order) instead of the racy `ROW_SLOTS_ATOMIC` per-world counter.
Combined with `CollisionPipeline(deterministic=True)` for reproducible contact order.

### Measured results (600 frames, fresh caches)
| config | CPU awake@599 | Metal awake@599 | Metal run-to-run |
|---|---|---|---|
| default (NOT_GUARANTEED) | 0/25 | 8-22/25 (varies) | NON-deterministic (1.5m) |
| R2R solver + det collision | 0/25 | 6/25 | **BIT-IDENTICAL (0.0mm)** |

- **Metal R2R is bit-identical run-to-run** (two runs, 0.0 mm diff). Run-to-run
  determinism is ACHIEVED on Metal with NO Warp scatter port — Steel's own ordered-row
  path does it. The dominant non-determinism was the integer row-slot reservation order,
  exactly as the review predicted (doubt #9: "audit every atomic, not just float ones").
- **But Metal R2R still differs from CPU by 1.67m and 6/25 bodies stay awake** (ang ~4 rad/s).
  CPU R2R settles (0/25). So determinism is achieved, CPU-parity settling is NOT.
- The float atomic scatters (accumulate_contact_wrenches_from_impulses, world_diag) are
  NOT changed by R2R yet Metal R2R is still bit-identical — so with fixed row+contact order
  the residual float-atomic variation does not amplify within 600 frames. The remaining
  CPU/Metal difference is a different chaotic basin (df64 friction-patch precision and/or
  float roundoff), not run-to-run non-determinism.

### Implication for the Warp port
The original goal ("make Metal run-to-run deterministic so Steel settles like CPU") splits:
1. Run-to-run determinism: ALREADY solved by Steel R2R + det collision. No Warp port needed.
2. CPU-parity settling: NOT solved by determinism alone; it's a chaotic-basin / precision
   difference. The Warp scatter port would NOT fix this either (it only adds determinism,
   which R2R already provides).

Therefore the Warp Metal deterministic-scatter port is NOT the right fix for the Steel
jitter. It is still a legitimate general backend feature (Metal lacks deterministic atomics
that CUDA has), but it does not address the actual user-visible problem. The partial
implementation (deterministic_metal.h + codegen hooks) should be evaluated as a standalone
feature, not as the Steel fix.

### Claude review claims verified
- Doubt #9 (integer row-slot atomics are the real source): CONFIRMED — R2R ordered rows
  make Metal bit-identical. The float-scatter-only hypothesis was incomplete.
- "Test (b) cheaply first": CONFIRMED valuable — saved a multi-day Warp port that would
  not have fixed the settling.
- B1 (MSL address spaces), B2 (os_log), B6 (cache key): still relevant IF the Warp port
  is pursued as a standalone feature; not yet verified by compilation.
