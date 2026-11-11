**Verdict:** Making df64 opt-in is right. A launch-time refusal by itself is not enough. If `"error"` only refuses at launch while codegen still emits df64, it will not match pre-P1 behaviour. `"error"` has to turn off the df64 codegen and header path too, so the generated MSL is byte-identical to pre-P1. I'm reviewing only the header and plan shown, not `builtin.h` or codegen. Claims about those are things to check, not proven bugs.

## 1. Global knob vs module option

A global knob alone is wrong. Use a module option (`wp.set_module_options({"metal_float64": "df64"})`) and let `wp.config.metal_float64` supply only the default. This is how Warp already handles `mode` and `fast_math`.

- **Global leak:** Newton or Steel setting `"df64"` silently switches it on for every kernel the user writes.
- **Stale result at module scope:**
  - Input: set `"df64"`, launch kernel K, set `"error"`, launch K again.
  - Expected: the second launch raises.
  - Actual: it runs, if the refusal or the loaded module was cached on the Module object at first load.
  - The reverse order (`"error"` then `"df64"`) can also keep raising if a failed load is remembered.
  - A module option avoids this because it changes the module hash and forces a reload.
- **Disk cache:** if the knob changes codegen (it should, see 2), it must be in the module hash. Plain `wp.config` values are not hashed. Otherwise:
  - Input: process 1 runs with `"df64"` and compiles; process 2 runs with `"error"`.
  - Actual: process 2 can load the cached df64 binary, if the refusal check runs after cache lookup.
  - Fix: run the check before cache lookup, and hash the option.

## 2. Where the check must live

Put it at every branch point P1 added, not only in `_metal_unsupported_reason`. Each P1 condition `metal and <float64>` becomes `metal and <float64> and opt == "df64"`. In `"error"` mode, `WP_METAL_DF64` must not be defined, and float64 must reach codegen exactly as before P1.

If P1 changed struct emission or function dropping and those stay active, a pure-float32 kernel can break:

- **Transcendentals (likely bug):**
  - This header defines no `sin`, `cos`, `exp`, `log`, `pow`, `atan2` or `tanh` for `float64`.
  - Input: one module with kernel A (float32 only) and kernel B calling `wp.sin(wp.float64(x))`.
  - Pre-P1: B is refused or dropped, A runs.
  - `"error"` mode with only a launch-time check: MSL compile fails with no matching `sin(float64)`, the whole module fails, and A cannot launch either.
  - Check `builtin.h` under `WP_METAL_DF64` before trusting this. If nothing else defines these, it is a real failure in both modes.
- **Struct (verify):**
  - Input: `@wp.struct S` with a `wp.float64` field in the same module as an unrelated float32 kernel.
  - If P1 emits S as `wp::float64` and the header is gated off, that is a compile error.
  - If the header is not gated off, the behaviour differs from pre-P1.
  - Either way it needs testing.

## 3. Gate the aliases and `DECLARE_FLOAT_OPS(float64)`?

Yes, gate them with the knob through `WP_METAL_DF64`. "Types exist but launches refuse" is not inert, because the header itself has bugs that show up at compile time:

- **Ambiguous constructor, `int64`:**
  - Input: `wp.float64(wp.int64(5))` in a kernel; codegen emits `wp::float64(var)` with a `long` argument.
  - Expected: 5.0.
  - Actual: ambiguous overload. `long`→`float`, `long`→`int` and `long`→`unsigned` are all same-rank conversions. `uint64` fails the same way.
- **Ambiguous constructor, `float16`:**
  - Input: `wp.float64(wp.float16(1.0))`.
  - Actual: ambiguous. `half`→`float` is a conversion, not a promotion, so it ties with `half`→`int` and `half`→`unsigned`.
  - Fix: add `float64(long)`, `float64(ulong)` and `float64(half)` overloads, or route all casts through `cast_float64`.
- **Exposure:** if the overloads stay defined in `"error"` mode, any generic `@wp.func` overload that codegen still instantiates for float64 hits these. Gating removes that exposure.

## 4. Warning vs explicit acknowledgement

The string value `"df64"` is already the acknowledgement. A second flag adds ceremony, not safety. What matters:

- Fire the warning on the first load or launch of a module that actually uses float64, and name that module. Do not fire it when the knob is set.
- Use Warp's `warn(..., once=True)`-style helper, not a bare `warnings.warn`, which user filters can silence.

Opt-in does not excuse leaving a wrong-sign result in place. The division regression is cheap to fix:

- **Input:** `float64(FLT_MAX) / float64(1.9999998f)`.
- **Trace:**
  1. `q1 ≈ 1.70e38`, finite.
  2. `float64(q1)*b`: `two_prod` gives `p = +inf`, normalize gives NaN, and the guard returns naive `+inf`.
  3. `r = a - inf` gives `-inf`.
  4. `q2 = -inf`.
  5. `normalize(q1, -inf)` gives `hi = -inf`, and the guard keeps the infinite hi.
- **Expected:** `+1.70e38`. **Actual:** `-inf`.
- **Fix:** if `naive` is finite and `r` is not finite, return `float64(naive)`. Keep an infinite `r.hi` only when `naive` is also infinite. Apply the same rule in `df_guard` everywhere. You lose precision, never sign.

Two more defects belong in the warning text (or should be fixed):

- **Public NaN/inf tests:** `isnan`, `isinf` and `isfinite` use `metal::isnan` and friends. The header's own comment says fast-math can fold those, and Metal's default math mode is fast.
  - Input: `x = float64(0)/float64(0)`, then `wp.isnan(x)`.
  - Expected: True. Likely actual under fast math: False.
  - Fix: use the bit-pattern helpers (`df_abits`, `df_isinf`) in the public functions too.
- **`_wp_native_fmin` / `_wp_native_fmax`:** the NaN check `b == b` is an FP compare that fast math can fold.
  - Input: `min(1.0, NaN)`.
  - Expected: NaN propagated, per the code's own comment.
  - Actual: `b == b` folds to true, so it returns NaN only by accident of the `<=` path. `min(NaN, 1.0)` likely returns 1.0.
- **Signed zero in `copysign`:** `copysign(x, -0.0)` returns `+|x|` because `y < 0` is false. This belongs to the known signed-zero class, but list it.

## 5. Steel's legitimate per-step case

Set the module option on Steel's own module only. That is the strongest argument for module scope over a global knob. Also measure the per-step cost: the header estimates 5–15x a native float op, and per-step is exactly where the header says df64 is "not intended for hot inner loops". If Steel only needs accurate sums or dots, an explicit compensated-float32 helper in Steel is more honest than reinterpreting `wp.float64`.

## 6. Verify "error" == pre-P1, don't assert it

Build a differential corpus of about 15–20 kernels and modules covering:

- float32 only
- float64 scalar, `vec3d`, `mat33d`, `quatd`
- float64 struct field
- generic `@wp.func` instantiated at float64
- `wp.sin` on float64
- int64→float64 cast
- float64 array argument
- float64 atomic
- mixed modules where one kernel is clean

For each item, at the pre-P1 commit and at HEAD in `"error"` mode, record:

1. The generated MSL text: the diff must be empty. Strip only hash and path lines.
2. Which kernels launch successfully.
3. Exception type and message for refusals: must be identical.
4. The module hash: must be identical in `"error"` mode, and must differ under `"df64"`.

Also add the knob-toggle sequence from section 1 within one process, and run it with a cold and a warm kernel cache. Any MSL diff in `"error"` mode means the claim is false.