# Metal fixedarray copy-semantics bug — the true root cause of Steel jitter

## Verified root-cause chain (all by execution)
1. Steel towers on Metal bounce / never sleep; CPU settles.
2. Run-to-run non-determinism is separately solved by SolverSteel(deterministic=RUN_TO_RUN)
   + CollisionPipeline(deterministic=True) -> Metal bit-identical run-to-run.
3. But Metal still differs from CPU because the INITIAL CONTACT MANIFOLD is degenerate:
   - minimal two-stacked-boxes repro: box-box pair CPU=4 contact points, Metal=1.
   - box-plane pairs are fine (4 points both). Only convex-convex (box-box) is broken.
4. Instrumented multicontact.build_manifold: CPU a_count=4 b_count=4; Metal a_count=4 b_count=3.
   Metal's shape-B feature-point collection drops the 4th face corner.
5. The 4th corner (0.1,0.1,0.102) is rejected by add_avoid_duplicates_vec2 because its 2D
   projection reads as (0,0) and arr[0] reads as (0,0) -> distsq=0 < eps^2 -> "duplicate".
6. Isolated repro (fixedarray_nested_probe.py): a wp.zeros fixedarray passed by Any through
   two nested @wp.func layers, written then read back:
   - CPU: reads return the written values (correct).
   - Metal: ALL reads return (0,0) — writes do not persist to the caller's array.

## The exact bug
Generated Metal inner_add_0 takes `wp::fixedarray_t<10,vec2> var_arr` BY VALUE.
- CPU fixedarray_t (array.h) inherits array_t<T> { T* data; } with T storage[Size].
  It has NO copy constructor (only operator=). The compiler-default copy ctor copies the
  `data` POINTER, which still points at the SOURCE object's storage. So a by-value parameter
  ALIASES the caller's storage: array_store writes through data -> caller's array. This is the
  semantics Newton's manifold code relies on (pass fixedarray by Any, mutate in place).
- My P0a Metal fixedarray_t (deterministic_metal... no, array.h Metal block) defines an EXPLICIT
  copy ctor: `data(storage)` + deep-copies storage[i]=other.storage[i]. So a by-value parameter
  is a DEEP COPY: writes go to the copy, caller's array stays zero. -> the manifold bug.

So P0a's Metal fixedarray copy ctor changed alias-on-copy (CPU) into deep-copy (Metal),
breaking in-place mutation through by-value Any params.

## Proposed fix
Make the Metal fixedarray_t copy constructor match CPU's alias-on-copy semantics:
copy the `data` pointer from `other.data` (which points at other.storage), NOT re-point to
local storage, and do NOT deep-copy storage. I.e. replicate the compiler-default memberwise
copy CPU gets. Concretely: remove the explicit Metal copy ctor (let it default) OR set
`data(other.data)` and skip the storage copy.

### Risks to attack (numbered)
1. Dangling pointer: alias-on-copy means the copy's data points at the source's storage. If the
   source is destroyed before the copy is used, dangling. CPU lives with this; does any Metal
   code path destroy the source while a copy is live? The by-value func param case is safe
   (caller outlives the call). Are there other copy sites (return by value, assignment to a
   longer-lived var) where aliasing would dangle or alias unexpectedly?
2. operator= vs copy ctor: CPU operator= DEEP-copies and re-points data=this->storage, but the
   default copy ctor ALIASES. So CPU is inconsistent between `a = b` (deep) and `a(b)` / by-value
   (alias). Should Metal mirror that exact inconsistency, or is there a cleaner uniform semantics
   that still fixes the manifold? Mirroring CPU exactly is safest for parity.
3. Does removing the deep copy break the P0a tests (test_fixedarray_metal.py) or the atomic
   fixedarray tests? Those may have relied on deep-copy. Need to re-run them.
4. Grad pointer: the copy ctor also sets grad(nullptr). CPU default copy copies other.grad.
   For correctness with requires_grad, should copy grad too. Match CPU.
5. Is by-value fixedarray passing even intended, or should codegen pass fixedarray by reference
   on Metal (WP_THREAD&)? That would be a cleaner fix (no aliasing hazard) but changes the ABI
   and must match how array_store/address resolve. CPU passes by value and relies on aliasing,
   so changing Metal to by-ref would diverge from CPU codegen. Prefer matching CPU.

## Decision rule (pre-committed)
Keep the fix ONLY IF: (a) fixedarray_nested_probe reads correct values on Metal, (b) the
two-stacked-boxes repro gives 4 contact points on Metal (matches CPU), (c) the P0a fixedarray
suite + atomic tests still pass, AND (d) the Steel sleeping demo with R2R settles (bodies sleep)
like CPU. If (d) fails but (a)-(c) pass, report determinism+manifold fixed but settling separately.