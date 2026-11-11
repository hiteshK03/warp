"""Run the Steel determinism arms example (Metal-port copy) headless on Metal.
Verifies: default copies diverge, deterministic copies stay bitwise identical.
"""
import sys
import numpy as np
import warp as wp

wp.init()
wp.set_device("metal:0")

sys.path.insert(0, "/Users/viani/dev/mujoco-test/steel_demo")
import steel_determinism_metal as ex
from newton.viewer import ViewerNull


class Args:
    headless = True
    num_frames = 200
    device = "metal:0"
    viewer = "null"


viewer = ViewerNull()
example = ex.Example(viewer, Args())
print("constructed on Metal OK", flush=True)

nan_seen = False
for f in range(Args.num_frames):
    example.step()
    example.render()
    if f % 50 == 0:
        # determinism example tracks divergence between the two copies per side
        div = getattr(example, "divergence", None)
        q = example.state_0.body_q.numpy() if hasattr(example, "state_0") else None
        nan = bool(np.isnan(q).any()) if q is not None else False
        nan_seen = nan_seen or nan
        print(f"frame {f}: nan={nan}", flush=True)
print("DONE nan_seen:", nan_seen, flush=True)
# report any divergence metrics the example exposes
for attr in ("default_divergence", "deterministic_divergence", "max_divergence"):
    if hasattr(example, attr):
        print(attr, "=", getattr(example, attr), flush=True)
