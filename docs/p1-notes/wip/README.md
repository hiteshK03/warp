# Paused WIP: Metal deterministic scatter-sort-reduce

This directory preserves a **paused, uncompiled** partial port of Warp's CUDA
deterministic atomic scatter to the Metal backend. It is NOT part of the Steel
jitter fix and is NOT wired end-to-end.

## Where the code lives
- Git branch: `wip/metal-det-scatter` (commit c2f5bc1c), branched from `metal-df64`.
- Patch: `metal-det-scatter.patch` (apply with `git apply` from the repo root).

## What it contains
- `warp/native/deterministic_metal.h` — MSL `det_scatter_buf_t` / `det_ctx` /
  `scatter()` record + active `WP_DET_*` macros.
- `warp/_src/deterministic.py` — `kernel_locals("metal")` real-buffer branch,
  `metal_args_struct_fields()` / `metal_args_unpack()` helpers.
- `warp/_src/codegen.py` — include the header, append det fields to the Metal
  `wp_args` struct, unpack det params before `kernel_locals`.

## What is missing (not done)
- Host Metal launch path in `launch_deterministic` (still CUDA-only).
- Host sort + reduce (plan: reuse `wp_radix_sort_pairs_int64_host` + fixed-order
  host reduce on unified memory).
- Never compiled or tested.

## Why it was paused
The Steel sleeping-demo jitter turned out to be a `fixedarray_t` copy-semantics
bug (fixed in `metal-df64` commit be0b2b22), NOT a determinism/precision problem.
Steel run-to-run determinism is already provided by
`SolverSteel(deterministic=RUN_TO_RUN)` + `CollisionPipeline(deterministic=True)`.
So this port is orthogonal to the user-visible problem. It remains a legitimate
standalone backend feature (Metal lacks CUDA's deterministic atomics) for later.

## Known issues to address before resuming (from Claude review)
See `../METAL-DET-SCATTER-REVIEW.md`:
- B1: MSL pointer/reference address spaces on the det structs/params.
- B2: `os_log(...)` is not valid MSL; report overflow via the flag only.
- B6: module cache key must include the deterministic mode for Metal.
- Doubt 8: graph capture — host sort/reduce must record as host_ops; reset count
  inside the captured sequence; keep buffers alive for the graph lifetime.
- Refuse consumed-return counter atomics on Metal in Python before codegen.

## Design + review docs
- `../METAL-DET-SCATTER-PLAN.md` — design, measured motivation, decision rule,
  and the pivotal finding that Steel R2R already gives determinism.
- `../METAL-DET-SCATTER-REVIEW.md` — Claude adversarial review.