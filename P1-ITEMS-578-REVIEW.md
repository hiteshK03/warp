# Review: plan for items 7, 5, 8 (df64 opt-in)

**Verdict:** the plan's direction is right: refuse what is unsupported, don't fake it. The mechanism and the detection rule are wrong, though, and as planned the item 7 fix both lets whole-module failures through and adds a silent wrong-result path.

## Item 7

### Problem 1: a list of transcendental names will be incomplete

Composite builtins call transcendentals internally, so their argument types hide the problem:

- `wp.quat_from_axis_angle(vec3d, float64)` (sin/cos)
- `wp.quat_to_axis_angle` (acos/atan2)
- `wp.quat_rpy`, `wp.quat_slerp`
- possibly `svd3`

Each of these resolves to a builtin named `quat_*`, not `sin`. A list of names like "sin, cos, exp" passes them through, and the MSL compile kills the whole module again.

**Failing input:** an opted-in module with kernel `q[0] = wp.quat_from_axis_angle(wp.vec3d(...), wp.float64(1.0))` next to a float32 kernel. Expected: only the quat kernel is refused. Actual: the module fails to compile.

**Fix:** use an allowlist, not a denylist. The header supports a small, closed set (add/sub/mul/div/neg, compare, sqrt, abs, min/max, floor/ceil/rint, mod, floordiv, copysign, casts, plus vec/mat templates built only on those). Any call whose resolved types contain float64 and whose builtin is not on the allowlist gets refused. Don't write the list by hand; generate it. For every builtin overload with Float/Scalar type variables, instantiate it at float64 (and vec/mat/quat-of-float64), put it in a one-kernel module, and compile it on Metal in df64 mode with `enable_backward` both on and off. The passing set is the allowlist. Check it in, with a test that fails when it drifts.

### Problem 2: the existing hook has the wrong shape and has never run

- `METAL_UNSUPPORTED_BUILTIN_PREFIXES` is keyed on builtin names. `sin(float32)` must stay legal, so refusal has to depend on types. Add a separate type-aware predicate at the same call site; don't overload the prefix tuple.
- The tuple is currently empty, so the stand-in path has never been exercised. Treat it as untested code.

### Problem 3: detection must use resolved types

`func.input_types` holds the declared signature (type variables like `Float`/`Scalar`), not the concrete types. Use the argument types and return type that `add_call` sees after overload resolution, and apply `type_scalar_type()` so vec3d, mat33d, quatd and transformd count as float64. Also check the return type: an overload can take float32 arguments and return float64.

### Problem 4: the adjoint pass escapes the forward-only stand-in (whole-module break)

`enable_backward` defaults to True, so the backward body is emitted into the same MSL translation unit. If only the forward call is replaced, `wp::adj_sin(float64…)` is still emitted and the module still fails. Check the adjoint symbol names separately: the header defines `adj_fabs`, so if Warp emits `adj_abs` for `wp.abs`, then a float64 `wp.abs` passes the forward check and breaks the backward compile.

**Failing input:** a default (backward enabled) opted-in module containing `y[i] = wp.sin(x[i])` (float64) plus a float32 kernel. Grep the generated MSL for `adj_sin`.

### Problem 5: `@wp.func` gives a silent wrong result (the most serious)

Function bodies are emitted once per module, not per kernel. Example:

```python
@wp.func
def f(x: wp.float64): return wp.sin(x)

@wp.kernel
def k1(a: wp.array[wp.float64]): a[0] = f(a[0])

@wp.kernel
def k2(b: wp.array[float]): b[0] = 1.0
```

The substitution would happen inside `f`'s body, while the refusal mark is placed on the kernel that contains the call. So `f` compiles with the stand-in, `k1` is never marked, and `k1` launches and writes whatever the stand-in returns.

- Expected: `k1` refused, `k2` runs.
- Actual (likely): `k1` runs and returns garbage, or zero.

**Fix:**
- Mark refusal per function overload.
- Propagate it transitively to every kernel that calls the function.
- Block launch of refused kernels in Python, not via the stand-in.
- Skip emitting refused kernel and function bodies entirely instead of substituting stand-ins. Then nothing needs to type-check (no worry about whether `metal_unsupported<T>` yields a usable value) and there is no runtime fallback value at all.

### Belt and braces

Prediction will always lag the real compiler. Add fault isolation as well: if the module's MSL compile fails, recompile each kernel on its own and refuse the ones that fail, with the compiler error attached. This costs nothing on the success path and guarantees the property item 7 is really about. Keep the codegen check for clear error messages.

### (a) vs (b)

Choose (a). In particular, reject the cheap version of (b), `float64(metal::sin(a.hi))`. It labels a float32-accurate result as float64, which is worse than refusing. A real df64 sin needs Cody-Waite reduction with a multi-part π and a df64 polynomial; that is a separate project with its own accuracy tests.

## Q4: other float64 gaps the header doesn't cover

| Category | Gaps |
|---|---|
| Transcendentals | sin cos tan asin acos atan atan2 sinh cosh tanh exp log log2 log10 pow cbrt erf erfc erfinv |
| Rounding and misc | round, trunc, frac, sign/step/nonzero (if they are explicit overloads), degrees/radians |
| Conversions | `float64(int64)`, `int64(float64)`, float16 ↔ float64 (only int/unsigned exist) |
| Atomics | `atomic_add`/`min`/`max`/`cas` on float64 arrays (no 64-bit float atomics on Metal) |
| Printing | `print` of a float64 |
| Adjoints | `adj_mod`, `adj_copysign`, `adj_floordiv`, `adj_min`/`adj_max` (if not generic), `adj_abs` naming |
| Composites | quat_* functions as listed above |

The generated allowlist sweep catches all of these. A hand-written list won't.

## Item 5: hash

Upstream's module hash is device-independent: options are hashed once, and binaries are separated by architecture afterwards. Don't make the hash depend on the device; that breaks the single-hash invariant and the cache layout.

Least invasive fix: when computing the hash, drop the option if its **effective** value is the default (False). Default modules then hash exactly as before on every device. Opted-in modules hash differently on CPU/CUDA too, which costs one harmless recompile.

Trap: base the check on the resolved effective value, not just whether the key is present in `module.options`. If any global default can flip the effective value to True while the key is omitted, Metal's cache will serve a non-df64 binary for a module that should now be df64. You say the global-config leak is closed; add a test that pins it.

The impact of item 5 is overstated. Warp's kernel cache directory is per version, so an upgrade normally invalidates caches anyway. The real value of the fix is "default mode is byte-identical to pre-P1", which is worth having, so do it.

## Item 8: inherit or document

Do what upstream does for the other module options (`enable_backward`, `fast_math`, `mode`). Check the unique-module creation code. If those are inherited and `metal_float64` isn't, that's a bug, not isolation: `module="unique"` isolates recompilation, not semantics.

Inheriting carries no numerical risk; the user already accepted df64 precision for the parent module. The real risk is order dependence, whichever way you go.

**Failing input:**
1. Decorate a `module="unique"` float64 kernel.
2. Then call `wp.set_module_options({"metal_float64": True})` on the parent.

- Expected: the kernel runs.
- With copy-at-creation: refused.

Test both orderings.

## Header bugs found while answering Q4

All of these give wrong results under df64.

1. **floor/ceil/rint/int conversion collapse to float before rounding.**
   `x = wp.float64(1.0) - wp.float64(1e-9)` is stored as (1.0, -1e-9). `wp.floor(x)` returns 1.0; expected 0.0. `mod(x, 1)` returns -1e-9; expected about 0.999999999. `int(x)` returns 1; expected 0.
   **Fix:** apply floor to the hi word, then correct using the sign of lo when hi is already an integer.

2. **`==` compares the float-rounded sum while `<` compares lexicographically, so they disagree.**
   With x = (1.0, 1e-10): `x == 1.0` is true and `x > 1.0` is also true. Expected: `==` is false.
   **Fix:** compare hi and lo word by word on canonical pairs.

3. **`float64(int)` truncates large values.**
   `wp.float64(16777217)` gives 16777216.
   **Fix:** set lo = float(v - int(hi)).

4. **`isnan`/`isfinite` use `metal::isnan`/`isfinite`,** which fast math can fold. This contradicts the header's own integer-bit-pattern rationale. `wp.isnan(wp.float64(0.0)/wp.float64(0.0))` may return False. Use `df_abits`.

5. **`_wp_native_fmin` does not propagate NaN when b is NaN.**
   `min(1, NaN)` returns 1, while `min(NaN, 1)` returns NaN. The comment claims NaN propagation. Compare against `builtin.h`'s float behaviour before claiming they match.

6. **Literals emitted as float lose precision** (outside this plan, but real).
   `wp.float64(0.1)` is only accurate to about 1e-9 relative, not df64 precision. **Fix:** split the literal into a hi/lo pair on the host when generating code.

## Suggested order

1. Generated allowlist sweep, covering forward and adjoint.
2. Type-aware refusal, propagated through `@wp.func`, with refused bodies not emitted.
3. Per-kernel compile retry when the module compile fails.
4. Header bugs 1–3.
5. Item 5 (omit the option from the hash when its effective value is the default).
6. Item 8 (match upstream semantics, plus the ordering tests).