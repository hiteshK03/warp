
## Implementation-review fixes (round 2)

The implementation review found two real correctness bugs + a name-stability risk,
all fixed and now covered by unittest (warp/tests/test_metal.py::TestMetalFaultIsolation):

- #1 Generic kernels: _metal_isolate_kernels now expands generics into overloads
  (ModuleHasher over live kernels yields concrete codegen roots) and builds one
  hasher per root. Previously get_mangled_name() on the unhashed generic template
  would raise. Test: generic f32/f64 overload pair, f32 survives isolation.
- #2 Leak: the isolated ModuleExec has handle=None, so the old __del__ guard
  skipped the per-kernel unload. Moved per-kernel unload OUTSIDE the handle guard;
  also unload partial libs if isolation fails partway.
- #3 Name stability: capture the mangled name AFTER the single-root hasher sets
  kernel.hash, and assert it is unchanged after codegen; refuse loudly if it drifts.
- #5 Capture the whole-module error BEFORE isolation (get_error_string is overwritten).
- #6 Parse metal_backward_unsupported markers in the isolated path too.
- #7 Membership test (name in metal_refused) instead of truthiness, so an empty
  error string is still a refusal.
- Tests are now real unittest assertions in warp/tests (CI), including a counted
  no-recompile-on-relaunch check (monkeypatch _metal_isolate_kernels, assert 0 calls).
- Repo hygiene: P1 notes moved to docs/p1-notes/; generated corpus/mode JSON removed
  from tracking and gitignored.

## Still open (honest)
- #4 Per-kernel two-tier backward (forward+backward, then forward-only with a
  "backward refused" mark) is NOT implemented; the module-wide forward-only retry
  still runs first. The mixed-backward test passes for the forward-failure case, but
  the broken-adjoint-only case (B forward works, B adjoint fails) is not isolated
  per-kernel. Behavior can depend on unrelated siblings (non-monotonic).
- #9 No per-kernel success caching: every failed load pays N isolated compiles.
- Injected interaction-failure test (each kernel compiles alone, module fails
  together) not added; the design handles it (per-kernel libs, no subset rebuild).
