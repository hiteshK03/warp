# Items 5 and 8: resolved

## Item 5 — option hash perturbation: FIXED (with a corrected understanding)

The metal_float64 option was folded into BOTH hash loops (hash_kernel and the
module-level options loop), perturbing CPU/CUDA hashes. Fix: skip the option in
both loops when its value is the default "error".

Verified by isolating the option's hash contribution (holding the header digest
constant): no-key == option=error (99530ca8…), option=df64 differs (d87ff0ca…).
So default modules get ZERO hash contribution from the option, on every backend.

IMPORTANT correction to my earlier framing: the WHOLE module hash still differs
from pure metal-1.17 (0/14 corpus match) because native_headers_digest() hashes
all native header bytes, and I added metal_float64.h + edited guards. That is
INTENDED upstream behavior (cache invalidation when headers change), not a defect.
The review's actual criterion — "default mode byte-identical to pre-P1" for the
option's contribution — is met. Full hash identity is impossible by design while
the headers differ, and should not be claimed.

## Item 8 — unique-module inheritance: NOT A BUG (matches upstream)

Measured: a module="unique" kernel does NOT inherit fast_math (an existing upstream
option) from the caller module either — both fast_math and metal_float64 stay at
their defaults. So metal_float64 non-inheritance is CONSISTENT upstream behavior,
not a bug. module="unique" isolates recompilation; options are set via
module_options={...}, which works (verified: unique + module_options df64 launches
0.33333334). No code change needed; documented.

The review's order-dependence concern is moot: since unique modules never inherit,
there is no copy-at-creation vs lazy resolution distinction to get wrong.

## Item 7 — still open (the substantial one)

The review showed my denylist plan is wrong (composite builtins hide
transcendentals; adjoints escape forward-only substitution; @wp.func gives silent
garbage). Correct approach: type-aware allowlist refusal propagated through
@wp.func + per-kernel compile-retry fault isolation. This is a larger change;
tackled next.
