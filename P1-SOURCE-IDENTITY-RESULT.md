# Source identity (item 6): resolved with the correct criterion

The review reframed the criterion: compare PREPROCESSED output with WP_METAL_DF64
undefined, not raw bytes (the guard text itself is always a raw-byte difference).
After de-contamination, the analysis is:

## builtin.h — token-identical to 1.17 when df64 is off
My only edits are 3 guard conditions `|| defined(WP_METAL_DF64)`. With the macro
undefined these reduce to the exact 1.17 condition; the #endif comment difference
is stripped by preprocessing. Verified token-for-token: TRUE.

## vec.h / mat.h / quat.h / svd.h — df64 edits preprocess OUT on Metal error mode
On Metal, WP_NO_FLOAT64 is ALWAYS defined (metal_crt.h). The alias guards
`#if !defined(WP_NO_FLOAT64) || defined(WP_METAL_DF64)` are therefore FALSE in
error mode, so vec3d/mat33d/quatd aliases and the _svd_config<float64>
specialization are excluded entirely. Verified by simulating Metal error-mode
preprocessing: none of these symbols appear. (On CPU/CUDA the aliases use float64
instead of double, but float64==double there, so semantically identical.)

## mat.h — one genuine shared-code change, semantic no-op without df64
The cross-type ctor and adj_mat_t now cast explicitly: `Type(other.data[i][j])`
and `OtherType(adj_ret.data[i][j])`. This is always compiled (generic mat_t).
For every native scalar type the explicit cast equals the implicit conversion; it
only MATTERS for df64, whose operator float() is explicit-only. So in Metal error
mode (no df64) it is a semantic no-op, though not a token-identical one.

## Honest conclusion
- Preprocessed Metal error-mode output is token-identical to 1.17 EXCEPT the two
  mat.h explicit casts, which are semantic no-ops.
- Full token identity would require reverting the mat.h casts and instead giving
  df64 an implicit conversion (unsafe: silent narrowing) or a traits hook. Not
  worth it for a research fork; the casts are a genuine robustness improvement.
- The earlier "falsified source identity" finding (e210cc65) was caused by (a) the
  builtin.h contamination (now removed) and (b) comparing raw bytes instead of
  preprocessed tokens. Both are addressed.

## Remaining open verification items (next)
- #else-branch coverage (item 4): float32 kernels exercising each WP_METAL_DF64
  #else side. Note: after de-contamination the only #if/#else on WP_METAL_DF64 is
  the svd.h specialization (no #else); the alias guards have no #else. So this
  item is largely moot — verify there are no #else branches left.
- cross-backend hash perturbation (item 5)
- per-kernel refusal in df64 mode for sin(float64) (item 7)
- module="unique" option inheritance (item 8)
