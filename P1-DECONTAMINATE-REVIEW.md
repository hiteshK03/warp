# Review: plan to de-contaminate metal-df64

**Verdict:** Staying on a 1.17 base is the right call. As written, though, the plan can produce a false pass. It has three gaps that matter more than the hand edits:

- It does not check for a stale kernel cache.
- Its check for other contaminated files is incomplete.
- It assumes every extra hunk came from upstream 1.18.

## Gaps the plan does not cover

**G1. A warm kernel cache can make the verification meaningless.** Your own commit 24f3ba6f already warns about warm caches. Warp's module hash covers kernel source, options and version. It does not hash the contents of `warp/native/*.h`. So after you edit the headers, cached `.o`/metallib binaries that were built against the 1.18 headers get reused, and every suite passes without testing anything.

- Failing input: run `a // b` with `a=-7, b=2` on HEAD, then clean up and rerun with the same cache. You get the same result even though `floordiv` changed.
- Fix: run every verification with a new, empty `WARP_CACHE_PATH`/`kernel_cache_dir` per run, and assert it is empty at start. Do not call `clear_kernel_cache`; AGENTS.md forbids it.

**G2. The contamination sweep is incomplete.** You checked builtin.h, array.h, vec, mat, quat and svd, plus codegen.py and context.py. You did not check `exports.h`, which is generated from builtins.py and would reference 1.18 symbols if it was copied from main. You also did not check `warp/_src/builtins.py`, `types.py`, `tile.h`, `range.h`, the Metal-specific headers, or `warp/tests/`.

- Run `git diff --stat metal-1.17 HEAD -- warp/` and classify every file.
- Failing input: if builtins.py or exports.h from 1.18 routes integer `//` to `floordiv_signed`, then after cleanup any kernel with `x // y` on `int32` fails to compile with `use of undeclared identifier 'floordiv_signed'`.

**G3. "Not df64, therefore upstream 1.18" is not proven.** If `main` is the fork's main, some HEAD-vs-1.17 hunks may be Metal fixes that landed on main after metal-1.17. Dropping everything that is not df64 or P0a would silently revert them.

- Ask yourself why you copied from main during P1a. If something failed to compile without it, the cleanup brings that failure back.
- Check the origin of each hunk with `git log -S'<snippet>' upstream/main main`.

**G4. Don't re-apply by hand; restore mechanically.** Contamination happened in P1a, after P0a. So:

- array.h: `git checkout <last pre-P1a commit> -- warp/native/array.h`. That version is metal-1.17 plus P0a by construction.
- builtin.h: restore it from the same commit, then apply only the three guard hunks, taken from `git diff main HEAD -- warp/native/builtin.h`.
- Check `git log -p metal-1.17..HEAD -- warp/native/array.h` in case any P1 commit touched array.h on purpose.
- Make it a forward fix-up commit. Don't rewrite history, because e210cc65 and acd9ff0d cite measurements taken on the contaminated tree. Note in the commit body that those measurements need re-running.

## Answers to your questions

**Q1. Rebase onto 1.17 or move to 1.18?** Stay on 1.17. The shipped overlay is 1.17.0.3, and the Metal host code is compiled into the 1.17 dylib. Moving to 1.18 means porting the Metal backend's host side, rebuilding libwarp, and taking in 1.18 codegen and context changes. That mixes P1 with upstream drift again, which is the exact problem you are fixing. Do the 1.18 move later as its own project with its own baseline.

**Q2. Does reverting `launch_bounds_t` shape to `int` matter?**

- Layout: `int` and `uint32_t` are both 4 bytes with 4-byte alignment. Offsets of `shape[4]`, `ndim` and the `size_t size` field (expected at 24) do not change, so host packing is unaffected either way. Reverting to `int` matches what 1.17 context.py and the 1.17 dylib pack. Confirm by grepping the ctypes `launch_bounds_t` fields in context.py, and check whatever struct the Metal encoder fills in the dylib.
- Semantics: the only difference is device-side signedness in the tid decomposition. For shapes below 2³¹, results are bit-identical.
- Verify: launch with `dim=(2,3,5,7)` and have each thread write `((i*3+j)*5+k)*7+l` to `out[i,j,k,l]`; expect `out.flatten() == arange(210)`.
- Also launch 1-D with `dim=1000003`, a size that is not a multiple of the threadgroup size, into a buffer with a sentinel slot. The sentinel must be untouched; this tests the `size` bound guard.
- Also run `dim=0`, which must not crash or write anything.

**Q3. What is the pre-P1 baseline?** metal-1.17 plus P0a, cleaned, meaning the pre-P1a commit restored as in G4. P0a is a prior step that stays, so including it isolates P1.

Define "source identity" after preprocessing: with `WP_METAL_DF64` undefined, `clang -E` / `xcrun metal -E` of the cleaned headers must equal the baseline token for token. The raw bytes of the full translation unit can never match, because the guard text itself is a difference. Your falsified source-identity claim in e210cc65 may partly be this measurement error, compounded by the contamination.

**Q4. Can dropping the 1.18 floordiv break a kernel?** If codegen.py and builtins.py are truly 1.17, nothing emits `floordiv_signed`, and dropping it is safe for compilation. The behaviour does change:

- If 1.18 moved signed `//` to floor semantics, then `-7 // 2` gives -4 on HEAD and -3 after cleanup, which is the 1.17 behaviour.
- Possible latent bug in HEAD: if only `floordiv` was ported and `mod` was not, then HEAD currently breaks `a == (a//b)*b + a%b` for negative operands. With `a=-7, b=2`: `-4*2 + (-1) = -9 ≠ -7`.
- Test: run a table over `a ∈ [-7,7]`, `b ∈ [-3,3]\{0}` on pure metal-1.17 and on the cleaned tree. Results must be identical, on both CPU and Metal.

**Q5. Minimal verification set:**

1. **Structural.** Every hunk of `git diff metal-1.17 -- warp/` is tagged df64 or P0a, with zero left over. Preprocessed builtin.h without `WP_METAL_DF64` equals metal-1.17. array.h preprocessed for the CPU target equals metal-1.17, assuming the P0a block is Metal-guarded; if not, explain why.
2. **Symbol dependency.** Grep the P0a block for `byte_offset_helper` and `array_atomic_*_slot`. If P0a's fixedarray atomic or adjoint overloads call a slot helper, the cleaned tree fails on the first fixedarray atomic with `use of undeclared identifier 'array_atomic_add_slot'`. Also check that its `byte_offset_helper` calls match the 1.17 signature.
3. **Behaviour, with a fresh cache each run (G1), on CPU and Metal:**
   - the P0a fixedarray suite, which must cover `wp.atomic_add` on a fixedarray element in both forward and `Tape` backward;
   - the df64 opt-in suite, run with the opt-in both on and off;
   - the error-mode corpus;
   - the floordiv/mod table and the launch-bounds tests above;
   - the 1.17 versions of `-k TestArithmetic -k TestArray -k TestAtomic -k TestLaunch`.
4. **Equivalence.** With the opt-in off, the floordiv table, launch tests and corpus outputs on the cleaned tree must be bit-identical to a pure metal-1.17 checkout. That is the actual proof that only contamination was removed.