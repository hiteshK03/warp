"""Run the Steel sleeping example (Metal-port copy) headless on Metal."""
import sys
import numpy as np
import warp as wp

wp.init()
wp.set_device("metal:0")

sys.path.insert(0, "/Users/viani/dev/mujoco-test/steel_demo")
import steel_sleeping_metal as ex
from newton.viewer import ViewerNull


class Args:
    headless = True
    num_frames = 400
    device = "metal:0"
    viewer = "null"


viewer = ViewerNull()
example = ex.Example(viewer, Args())
print("constructed on Metal OK", flush=True)

nan_seen = False
for f in range(Args.num_frames):
    example.step()
    example.render()
    if f % 100 == 0:
        q = example.state_0.body_q.numpy()
        awake = example.observables.body_awake.numpy()
        nan = bool(np.isnan(q).any())
        nan_seen = nan_seen or nan
        print(f"frame {f}: nan={nan} awake_bodies={int(awake.sum())}/{len(awake)}", flush=True)
print("DONE nan_seen:", nan_seen, flush=True)
