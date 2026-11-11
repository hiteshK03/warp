# Plan: make df64 opt-in; restore raise-by-default for float64 on Metal

## Why (all measured, see P1-DECISION-RESULT.md and the *-VERIFIED.md files)

df64 currently substitutes itself silently for IEEE float64 on Metal. Confirmed
defects while it is the default:
- finite/finite division near FLT_MAX returns a WRONG-SIGN infinity
  (FLT_MAX / 1.9999998 -> -inf; expected +1.70e38). This is a regression the guard
  introduced: it was NaN before.
- spurious overflow for finite values: sqrt(FLT_MAX)^2 -> inf; three-term sum -> inf
- relative error 1.0 (100% wrong) for both df64 and float32 at 1e-12 scale
- signed zero lost; 1/((-1)*0) -> +inf instead of -inf
- mod is floored-modulo, not Warp's fmod; mod(1,inf) -> NaN

And the decision measurement showed df64 is NOT needed for Newton's inertia path:
that inverse is one-shot at finalize() (single launch site, builder.py:16241) and
mostly computed on the host anyway. A CPU float64 inverse rounded to float32 is at
least as accurate with correct IEEE semantics.

So silently replacing float64 is indefensible. Default must go back to raising.

## Proposed design

Add a config knob, default OFF:

    wp.config.metal_float64 = "error"   # default: raise at launch (pre-P1 behaviour)
                            = "df64"    # opt-in: double-single emulation

- "error": `_metal_unsupported_reason` returns a reason for internal float64 use,
  exactly as before P1. The native df64 header can stay compiled (harmless) or be
  gated; the launch-time refusal is what matters.
- "df64": current behaviour, but emit a ONE-TIME warning naming the limitation
  (precision gained, exponent range NOT gained, known defects) so nobody can
  enable it without being told.

The existing hard refusals (fp64 arrays in the ABI, fp64 atomics, raw double) stay
in BOTH modes — they are unsafe regardless.

## Questions for review

1. Is a config knob the right mechanism, or should it be per-module/per-kernel
   (e.g. a kernel option) so a library can't flip it globally for its users?
2. Where exactly should the check live so that "error" mode reproduces pre-P1
   behaviour faithfully? Are there paths (struct codegen, function dropping) that
   would now behave differently from before P1 even in "error" mode?
3. Should the native aliases (vec3d/mat33d/quatd) and DECLARE_FLOAT_OPS(float64)
   also be gated on the knob, or is the launch-time refusal sufficient? What breaks
   if the types exist but launches refuse?
4. Is a one-time warning enough, or should "df64" require an explicit
   acknowledgement (e.g. a second flag) given it can return wrong answers?
5. What is the cleanest way to keep the df64 work usable for its remaining
   legitimate case (Steel per-step device fp64) without making it the default?
6. Any risk that "error" mode silently differs from the true pre-P1 behaviour, and
   how would I verify that claim rather than assert it?
