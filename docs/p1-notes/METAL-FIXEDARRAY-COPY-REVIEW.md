**Verdict:** the root cause is plausible, but the fix as written may not change anything. Before you trust the decision rule, check one blocking precondition and one new hazard.

## 0. Blocking: aliasing only helps if every Metal accessor goes through `data`

Metal `fixedarray_t` has no base class, so P0a had to write its own Metal overloads of `address`, `array_store`, `load`, `atomic_*`, `adj_*` and `view`. The kernel comment you pasted says `wp::address(var_arr, 0)` becomes `&var_arr.storage[0]`. If that is literally true, `address` reads the copy's own storage, which is a stale snapshot taken at call time. In that case changing the copy ctor to alias does nothing for those overloads.

Worse, the overloads could be mixed: `array_store` through `data` and `address` through `storage`. Then after the fix, writes go to the caller and reads come from the stale copy, which is the same "(0,0)" symptom.

Failing probe to run right after the change:
```python
@wp.func
def inner(arr: Any, i: int, v: wp.vec2):
    arr[i] = v
    return arr[i]            # read back in the SAME frame
```
Call it from a kernel with a `wp.zeros(shape=10, dtype=wp.vec2)` fixedarray and `v=(1,2)`. Expected: returns `(1,2)` and the caller's `arr[i] == (1,2)`. If `address` uses `storage`: it returns `(0,0)` while the caller's array is `(1,2)`.

Audit every Metal fixedarray overload in `array.h` for `.storage[` and rewrite them to use `.data[`. Keep `.storage` only in constructors and `operator=`.

## 1. Dangling pointers: one new hazard, and it depends on how codegen initializes locals

By-value params are safe, because the caller frame outlives the inlined call. The dangerous case is a fixedarray returned by value and then copy-initialized:
```python
@wp.func
def make():
    a = wp.zeros(shape=4, dtype=float)
    a[0] = 3.0
    return a

@wp.kernel
def k(out: wp.array(dtype=float)):
    b = make()
    clobber()        # another func with a local zeros(shape=4) filled with 99.0
    b[1] = 7.0
    out[0] = b[0]; out[1] = b[1]
```
Expected `out = [3, 7]`. With an aliasing copy ctor, `b.data` points at `make`'s dead `a.storage`, so this is undefined behavior: likely 99, 0 or garbage once MSL fully inlines and its stack-slot coloring reuses that alloca.

- **Safe:** if Warp codegen declares `wp::fixedarray_t<..> var_b;` first and then assigns `var_b = make(...)`, `operator=` copies `other.storage` (valid memberwise snapshot) and re-points `data`. That is why CPU survives.
- **Breaks:** any Metal codegen site that emits `T var_b = expr;` or `auto var_b = ...`, or builds tuples or structs containing a fixedarray. Copy elision does not apply to a named local return, so the aliasing copy ctor runs.

Grep the generated `.metal` for fixedarray copy-init sites and tuple construction. Add the kernel above to the test suite. The old deep-copy ctor was immune to this; the alias ctor is not.

## 2. Copy ctor vs `operator=`: mirror CPU, but know the inconsistency you're inheriting

The CPU semantics are internally inconsistent:
- The copy ctor aliases.
- `operator=` copies `other.storage` instead of `*other.data`. An aliased copy's `storage` is stale, so assigning from a by-value param loses any writes made through it.

Concrete case, the same on CPU and on Metal after the fix:
```python
@wp.func
def g(arr: Any):
    arr[0] = 5.0
    return arr
# kernel: a = wp.zeros(shape=4, dtype=float); x = g(a)
```
Expected (intuitive): `x[0] == 5`. Actual: `x[0] == 0`, because `operator=` reads the stale snapshot, while `a[0] == 5`.

Exact parity is the right call for this fix. If you want uniform semantics, change `operator=` on both backends to read `other.data[i]`. Do that upstream as a separate change, not here.

Simplest faithful implementation: delete the Metal copy ctor or write `= default`. The implicit copy ctor still exists alongside the user-declared `operator=`, and it copies `storage`, `data`, `grad`, `shape` and `strides`, which is exactly CPU behavior. Don't hand-write "`data(other.data)` and skip storage". Skipping the storage copy diverges from CPU in exactly the return-then-assign case above, where `operator=` then reads uninitialized storage.

## 3. Grad

Copy `grad` in the ctor (`= default` does this). Also, Metal `operator=` sets `grad = nullptr` while CPU copies it. That is another divergence; fix it too.

Real-world impact is near zero, since local fixedarrays have no grad. One caveat: under aliasing, the backward pass replays forward and re-mutates the caller's array. `add_avoid_duplicates` then sees the point as already present and takes the other branch, so the count diverges. CPU has this too, so it is parity, not a regression, but don't trust adjoints through this code on either backend.

## 4. Tests

Any existing P0a test that asserts the caller is unchanged after a func mutates a by-value fixedarray is encoding the bug. Re-derive its expected values from a CPU run rather than "fixing" the test. Atomic tests on a fixedarray passed into a func will now mutate the caller. That is correct, but their expectations may need the same CPU-diff treatment.

## 5. By-ref codegen

The "ABI change" worry is overstated. These functions are static inline in one translation unit, and the host can't pass fixedarrays into kernels, so nothing external sees the signature. By-ref would remove the dangling hazard and the dead `Size*sizeof(T)` copy per call, which costs register and stack pressure with the larger manifold buffers.

Costs of by-ref:
- A Metal-only codegen special case on `Any` args.
- Prvalue arguments like `inner(wp.zeros(...))` won't bind to a non-const `thread&`.
- An `operator=` on the param would write into the caller's storage, where CPU writes only into its local copy.

Net: it is cleaner but diverges from CPU. Do the alias fix now for parity. Consider by-ref as a later change on both backends.

## Decision rule additions

- **(a′)** The same-frame write-then-read probe from section 0 must pass, not just the nested probe.
- **(e)** The return-then-clobber kernel from section 1 must give `[3, 7]` on Metal.
- **(f)** A static check that no Metal fixedarray accessor indexes `.storage`.

If (d) fails while everything else passes, report it as you planned. Settling also depends on other float-vs-df64 differences.