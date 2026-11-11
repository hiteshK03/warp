# Plan: de-contaminate the metal-df64 branch (1.18 headers on a 1.17 base)

## Discovery

The branch is based on metal-1.17 (matches the shipped libwarp.dylib 1.17.0), but
during P1a I copied some native headers from the `main` branch (1.18-era). Verified
by git:

- `builtin.h`: HEAD vs metal-1.17 shows `floordiv_signed` rewrite + `launch_bounds_t`
  `int shape[N]` -> `uint32_t shape[N]`. These are UPSTREAM 1.18 changes (present in
  upstream/main, absent in metal-1.17), NOT df64 work. HEAD vs main shows my ONLY
  builtin.h edits are the 3 df64 guard pairs (`|| defined(WP_METAL_DF64)`).
- `array.h`: HEAD carries 1.18 `array_atomic_*_slot` helpers and a
  `byte_offset_helper` signature change that are NOT part of P0a fixedarray.

So the branch mixes 1.18 header text with a 1.17 binary. That is a real risk
(ABI/behaviour drift) and it also pollutes any pre-P1 source differential.

## Why it matters for the open verification items

- Item 1 (baseline): a meaningful pre-P1 comparison must hold the non-df64 code
  fixed. Right now "pre-P1" (metal-1.17) and HEAD differ by upstream 1.18 changes
  too, so the differential conflates my work with upstream drift.
- Item 6 (source identity): impossible while unrelated 1.18 text is inlined.

## Proposed cleanup

Rebase my *intended* edits onto the metal-1.17 versions of the contaminated files:

1. `builtin.h`: start from metal-1.17's builtin.h, apply ONLY the 3 df64 guard
   pairs (adj_float64, DECLARE_FLOAT_OPS(float64), DECLARE_ADJ_APPROX_DIV(float64)).
   Drop floordiv_signed and the uint32_t shape change (they belong to 1.18).
2. `array.h`: start from metal-1.17's array.h, re-apply ONLY the P0a fixedarray
   block (the standalone Metal fixedarray_t + index/address/store/atomic/adj
   overloads). Drop the 1.18 slot-atomic helpers and byte_offset_helper change.
3. vec.h/mat.h/quat.h/svd.h: already identical to metal-1.17 except my df64 edits
   (main vs 1.17 = 0 lines), so they are clean — verify and leave.
4. Re-run: fixedarray suite (P0a), df64 opt-in suite, error-mode corpus.

## Questions for review

1. Is rebasing-onto-1.17 the right move, or should the whole branch move to a 1.18
   base (and a 1.18 libwarp)? What are the trade-offs given the shipped overlay is
   1.17.0.3?
2. The `launch_bounds_t` shape type (int vs uint32_t) is an ABI-ish struct shared
   with the host launcher. Does reverting it to `int` on a 1.17 binary matter, or
   is it purely device-side? How do I verify I'm not breaking launch metadata?
3. After cleanup, what is the correct "pre-P1" baseline for the source differential
   — metal-1.17 exactly, or metal-1.17 + P0a (since P0a is intended to stay)?
4. Any risk that dropping the 1.18 floordiv change breaks a kernel that the 1.17
   codegen expects? (codegen.py on this branch — is it 1.17 or 1.18?)
5. What is the minimal verification set to prove the cleanup changed nothing
   functional for P0a + df64, and removed only the contamination?
