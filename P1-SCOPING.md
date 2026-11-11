# P1 — float64 on Metal: scoping findings

## Hard constraint (verified, not assumed)

Apple's MSL compiler rejects `double` outright:

```
error: 'double' is not supported in Metal
typedef double float64;
```

Probe: commented out `#define WP_NO_FLOAT64` in `native/metal_crt.h:17` and bypassed
the codegen gate (`ModuleBuilder._metal_unsupported_reason`). Compilation fails at
the first `double` token. So P1 is **emulation**, not gate removal. (Header restored.)

## Where the gate lives (two layers, both needed)

1. `native/metal_crt.h:17` — `#define WP_NO_FLOAT64` strips `float64`, `vec3d`,
   `mat33d`, `quatd`, and ~138 guarded blocks across native headers.
2. `_src/context.py` — `ModuleBuilder._metal_unsupported_reason()` regexes
   `\bfloat64\b|\bdouble\b` in generated source and (a) drops the function into
   `metal_skipped`, (b) replaces the kernel body with a `// wp_metal_unsupported`
   marker, which `ModuleExec` turns into a launch-time RuntimeError.

## Key enabler: the linear algebra is already generic

`mat_t<Rows,Cols,Type>`, `vec_t<N,Type>`, `inverse()` (mat.h:3842), `eig3()`
(svd.h:765) are all templated over `Type`. If `wp::float64` becomes a *struct* on
Metal instead of `double`, this existing code compiles against it unchanged —
no per-op Metal reimplementation of eig3/inverse needed.

## Newton's actual requirement (census on main @ 888670c3)

| area | fp64 mentions | notes |
|---|---|---|
| actuators | 144 | drive PID/PD linearization |
| geometry | 48 | inertia eig/inverse (finalize path) |
| solvers/steel | 28 | compliance law, friction patch moments, torsion |
| sim | 3 | |
| xpbd / featherstone / semiimplicit / mujoco | 0 | already Metal-clean |

**Crucially: fp64 is internal-only in the blocking kernels.**
`compute_body_inverses_kernel(body_mass: array[float32], body_inertia: array[mat33],
body_inv_mass: array[float32], body_inv_inertia: array[mat33])` — no float64 in the
ABI. Same for `validate_and_correct_inertia_kernel`. So **no host/device layout
change is required**; emulation can be confined to the scalar type + its ops.
This is what makes P1 tractable.

## Design options

### A. `float64` = float32 on Metal (precision downgrade)
`typedef float float64;` + unguard the type aliases. Smallest change; every generic
op just works. Cost: silently loses the precision Newton deliberately added
(its comments say float64 intermediates exist to avoid determinant underflow /
reciprocal overflow for small float32 inertias, and to resolve small negative
eigenmoments). Newton 1.7 used float32 and was fine on Metal, so it *works*, but it
re-introduces exactly the numerical failure modes upstream fixed. Also silently
wrong for user kernels that asked for fp64.

### B. Double-single (df64) emulation — recommended
`struct float64 { float hi, lo; }` implementing Dekker/Knuth two-sum, two-product
(FMA-based), giving ~2x float32 mantissa (~44-48 bits) — enough to cover the
underflow/overflow and eigen-resolution cases Newton cares about, at ~5-15x the
cost of a float op. All generic templated ops work if we provide the scalar
operators (+,-,*,/,comparisons) and the CRT/math functions they call.
Caveats: needs `metal::fma` for error-free transforms; range still float32's
(exponent is not widened), so true fp64 *range* cases remain unsupported —
document, don't pretend.

### C. Software IEEE-754 binary64 (full fidelity)
Correct range + precision, but slow (integer limb arithmetic) and a large surface
(every math function). Overkill for per-body 3x3 init kernels; not worth it now.

## Recommendation

**B**, staged:
1. P1a: `float64` df64 struct on Metal + scalar ops + casts; unguard type aliases
   and the codegen gate for these types; add numeric tests vs CPU float64.
2. P1b: whatever math/CRT functions the Newton blocking kernels actually pull in
   (sqrt, abs, min/max, comparisons, `eig3`/`inverse` come free via templates).
3. Validate: Newton main `finalize()` without `skip_validation_inertia`, then
   SolverSteel ant benchmark on metal:0; compare against CPU fp64 reference.

Range limitation (float32 exponent) must be stated in docs + a runtime guard where
feasible. Any op we cannot emulate faithfully should keep raising, not silently
degrade — preserve the existing "fail loudly" contract.

## Alternative worth noting

Newton could expose a float32 inertia path (as 1.7 had), making P1 unnecessary for
Newton specifically. That's a Newton-side conversation, not a Warp fix; the Warp
df64 work benefits any fp64 user kernel, so it is the more general investment.

---

# P1a — IMPLEMENTED AND VALIDATED (df64)

## What landed

- **`warp/native/metal_float64.h`** (new): `wp::float64` as a double-single pair
  `{float hi, lo}` with Knuth two-sum / FMA two-product error-free transforms;
  arithmetic, comparisons, `abs`/`sqrt`/`floor`/`ceil`/`rint`/`mod`/`copysign`,
  `isnan`/`isinf`/`isfinite`, `_wp_native_fmin/fmax`, and reverse-mode adjoints
  (`adj_sqrt`, `adj_fabs`, `adj_approx_div`, ...) with signatures matching
  `DECLARE_ADJOINTS` in builtin.h. All references carry `WP_THREAD` (MSL rule).
- **`metal_crt.h`**: keeps `WP_NO_FLOAT64` (raw `double` stays out) and adds
  `WP_METAL_DF64` + includes the new header.
- **`builtin.h`**: enables `DECLARE_FLOAT_OPS(float64)` and
  `DECLARE_ADJ_APPROX_DIV(float64)` under `WP_METAL_DF64` — reuses Warp's own
  macro rather than hand-writing 26 named ops (codegen emits `wp::add`/`mul`/...,
  not operators).
- **`vec.h` / `mat.h` / `quat.h`**: `vec3d`/`mat33d`/`quatd` aliases now use
  `float64` instead of raw `double` and are enabled under `WP_METAL_DF64`. Since
  `mat_t`/`vec_t`/`inverse`/`eig3` are templated over the scalar type, the whole
  linear-algebra stack works against df64 with **no per-op reimplementation**.
- **`svd.h`**: `_svd_config<float64>` specialization (epsilon 1e-12).
- **`mat.h` cross-type conversion**: `mat_t<OtherType>` ctor and `adj_mat_t` now
  cast scalars explicitly (`Type(...)`, `OtherType(...)`). df64's `operator float()`
  is intentionally explicit (no silent narrowing), so the generic converters had
  to say so. Behavior on CUDA/CPU unchanged (explicit cast of an implicitly
  convertible scalar is a no-op).

## Measured results (Apple M5 Pro, macOS 27, warp-metal 1.17.0.3)

| test | df64 on Metal | reference | verdict |
|---|---|---|---|
| `(1 + 1e-10) - 1` | 1.000000e-10 | float32 gives 0.0 | precision gain proven |
| `eig3` with eigenvalues {1e-8, 1, 3} | rel err 3.5e-8 on smallest | CPU float64 | resolves tiny moments |
| Newton `_inverse_inertia_wp`, inertia ~1e-7 | max rel err 1.2e-8 | CPU float64 | the blocking kernel now runs |
| `mat33d` inverse via generic template | diag [0.5, 0.25, 0.125] exact | analytic | forward + adjoint compile |

## Remaining (P1b)

- Lift the codegen gate (`_metal_unsupported_reason`) for the df64-supported
  types — currently the probes bypass it, so nothing is enabled by default yet.
  Must keep raising for genuinely unsupported things (true binary64 range,
  fp64 arrays in kernel ABI, atomics on fp64, transcendentals not yet emulated).
- Newton `finalize()` without `skip_validation_inertia` on metal:0, then
  SolverSteel ant benchmark; compare to CPU.
- Regression tests in the fork's test suite + docs entry stating the
  **range limitation** (float32 exponent; not IEEE binary64) and the ~5-15x cost.

## P1b — gate opened, acceptance PASSED

`_metal_unsupported_reason()` now permits internal fp64 (df64) and still raises for:
fp64 arrays in the kernel ABI (host binary64 vs two float32s — not marshallable),
fp64 atomics (no 64-bit float atomics; a df64 pair cannot update atomically), and
raw `double`. The struct-level `"float64" in struct_source` short-circuit was
removed so the single predicate decides. Also enabled `adj_float64` under
`WP_METAL_DF64` (fp32->df64 cast adjoint).

### Acceptance (Newton main 1.9.0.dev0 @ 888670c3, metal:0)

- `ModelBuilder.finalize()` with **full fp64 inertia validation** — no
  `skip_validation_inertia`, no monkeypatching: **OK**. This runs both
  `validate_and_correct_inertia_kernel` (fp64 eig3) and
  `compute_body_inverses_kernel` (fp64 inverse), the two kernels that previously
  refused to launch.
- `body_inv_inertia` matches the CPU device **exactly**:
  `[954.49884, 556.0499, 570.3378]` on both.
- **SolverSteel: 20 steps on metal:0, no NaN.** First time Steel has run on Apple
  Silicon in this work.
- Two artifacts in the output were checked against CPU and are NOT Metal issues:
  the Steel "dense constraint-row overflow" warning (config limit,
  `articulation_constraint_max`) and a Featherstone NaN on the ant MJCF appear
  identically on CPU.

### Regression tests

`bench/test_df64_metal.py` — 4/4 PASS: precision gain vs float32, fp64 inverse vs
CPU binary64 (1.17e-8), eig3 tiny eigenvalue (3.47e-8), tape through df64 (grad
exactly 1.0). `bench/test_fixedarray_metal.py` still 5/5 PASS; Warp
kernel/tape/nbody benchmarks unchanged (nbody 241.7ms CPU vs 23.3ms Metal).

### Branch note (important for anyone rebuilding the overlay)

Overlay Python must match the shipped `libwarp.dylib`. The fork's `main`
context.py is newer than the 1.17 binary (references
`wp_llvm_set_cpu_block_runtime`), so P1 work targeting the released overlay lives
on `metal-df64` (based on metal-1.17), not `main`.


---

# CORRECTION — df64 does NOT fix range-driven underflow/overflow

An earlier claim in this document (that df64 "covers the underflow/overflow cases
Newton cares about") is **wrong** and is retracted here. Measured on metal:0
(bench/p1_range_map.py): determinant of a scaled identity, then its reciprocal.

| scale | true det | df64 det | true 1/det | df64 1/det | verdict |
|---|---|---|---|---|---|
| 1e-3  | 1e-9  | 1e-9  | 1e9  | 1e9  | ok |
| 1e-7  | 1e-21 | 1e-21 | 1e21 | 1e21 | ok |
| 1e-12 | 1e-36 | 1e-36 | 1e36 | 1e36 | ok |
| 1e-15 | 1e-45 | **0.0** | 1e45 | **NaN** | BROKEN |
| 1e-20 | 1e-60 | **0.0** | 1e60 | **NaN** | BROKEN |

What df64 actually buys is **mantissa precision** (~44-48 bits: resolving small
eigenmoments, avoiding cancellation). It does **not** widen the exponent range.
Newton's own comment cites "determinant underflow" and "reciprocal overflow" —
those are *range* problems and df64 leaves them unchanged. The failure mode is
**NaN, silently**, not a clean inf and not a raised error.

Practical impact: Newton's real models are far inside the safe zone (quadruped
inertias ~1e-3 scale; `body_inv_inertia` matched CPU exactly), so the P1
acceptance result stands. But "float64 now works on Metal" is too broad a claim.

## Comparison-operator check (was a suspicion, now tested)

`operator<` compares hi then lo lexicographically, which is only valid for
canonical pairs. Tested (bench/p1_selfcheck.py): equal values built via different
arithmetic paths order consistently with subtraction, and large-magnitude /
tiny-delta ordering agrees with `(y-x)>0`. Also, Warp exposes **no two-arg
`float64(hi, lo)` constructor**, so user code cannot inject non-canonical pairs.
Risk is therefore narrower than feared, but not provably absent for all internal
intermediates.

## Open design question (do not ship as-is without deciding)

Silently substituting a non-IEEE type for `float64` is a footgun. Options:
1. **Opt-in** (`wp.config.metal_float64 = "df64"`), default keeps raising. Most
   honest; costs Newton-on-Metal out-of-the-box.
2. **Automatic + loud one-time warning** stating precision gained, range NOT
   gained, and the NaN failure mode.
3. **Range contract with detection**: keep df64, but detect subnormal/zero `hi`
   in division and raise/flag instead of returning NaN.
4. **Extended-exponent df64** (separate scale factor): actually fixes range, but
   substantially more work and slower.

Leaning 2+3 for a research fork, 1 for anything upstreamed.


---

# EXTERNAL REVIEW (Claude Opus) + EMPIRICAL VERIFICATION — P1 IS NOT SOUND AS WRITTEN

Full review: P1-CLAUDE-REVIEW.md. Repro: bench/p1_verify_review.py.
Every claim below was **executed and confirmed** on metal:0, not accepted on faith.

| # | claim | measured | severity |
|---|---|---|---|
| C1 | scalar `wp.float64` kernel arg bypasses the gate | launched with `1.0`, kernel reads **1.875** | **CRITICAL — silent garbage** |
| C3 | literals/int conversion capped at 24 bits | `float64(16777217) - float64(16777216)` = **0.0** | high |
| C4 | `floor`/`mod` round `hi+lo` first | `floor(1-2^-30)` = **1.0** (want 0); `mod(x,1)` = **-9.3e-10** (negative from positive operands) | high |
| C5 | inf arithmetic collapses to NaN | `3e38*3e38`, `inf+1`, `inf*1`, `sqrt(inf)` all **NaN** | high |

C1 is the worst: host packs binary64 into 8 bytes and `struct{float hi,lo}` is also
8 bytes, so nothing fails — the bits are silently reinterpreted. My regex gate only
inspected *arrays*, never scalar arguments or structs with fp64 fields. The gate
therefore does not deliver the safety property I claimed for it.

Also raised and NOT yet verified (treat as open):
- **Fast-math collapse**: Metal may compile with fast math; reassociation can fold
  `(a-(s-bb))+(b-bb)` to 0 and `fma(a,b,-(a*b))` to 0, silently degrading df64 to
  float32. My passing canary only proves this compiler/inlining context left it
  alone. Needs `mathMode = .safe` for df64 modules + a CI canary.
- **`==` vs `<` disagreement**: `==` rounds `hi+lo`, `<` is lexicographic.
- **`<=`/`>=` written as negated `<`** makes NaN comparisons true both ways.
- **Sloppy Dekker add**: error bounded by |a|+|b|, not |a+b| — weakest exactly in
  the cancellation case df64 exists for. Needs the accurate two-two_sum add.
- **Gate misses** `array_t<vec_t<3,float64>>`, mat/quat/transform arrays, fabric/
  indexed variants; the atomic regex almost never fires because codegen emits
  `atomic_add(var_a, var_i, var_v)` with no type in the text.
- **Validation was not discriminating**: a 1.2e-8 relative error on a float32
  output is just float32 output rounding (eps/2 ~ 6e-8). "Matches CPU exactly" on
  float32 outputs would also hold with float32 intermediates in a well-conditioned
  case. Each test needs a float32-intermediate control that visibly fails. The
  eig3 {1e-8,1,3} test does discriminate; the inverse-inertia test does not.

## Reviewer's recommendation (and my assessment)

The trade is backwards for the motivating case: `finalize()` is a **one-time setup
kernel**, so paying 30-100x for *correct* softfloat IEEE binary64 is free in
practice, while df64 buys speed we don't need in exchange for non-IEEE semantics.

Proposed tiering:
1. `float64` on Metal = **softfloat binary64** (bit-exact, full range, correct
   inf/NaN, layout-compatible so fp64 arrays/args work and most of the gate
   disappears; atomics stay blocked).
2. df64 becomes an explicit opt-in for hot paths (e.g. Steel per-step), only after
   discriminating validation with float32 controls.
3. **Power-of-two pre-scaling** inside `inverse`/`eig3` removes the range hazard
   for those ops at negligible cost — and is worth upstreaming to Newton
   independently, since it helps float32 too.
4. Alternatively/additionally: run the fp64 setup kernels on the **CPU device**
   (real binary64, unified memory makes the transfer nearly free). Doesn't help
   Steel's per-step fp64.

## Status

P1 stays on `metal-df64` (NOT proposed upstream). The Newton acceptance result
(finalize + Steel stepping, inverses matching CPU) is real but **narrow**: it holds
for well-conditioned, in-range models and does not establish that float64 is
correctly supported. Do not describe P1 as "float64 works on Metal".
