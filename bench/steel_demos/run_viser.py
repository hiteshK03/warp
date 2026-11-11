"""Steel sleeping demo with the VISER web viewer on Metal — interactive controls.

Serves http://localhost:8080. The viewer's Pause / Step / Reset buttons work:
- Pause toggles free-running vs paused.
- Step (enabled while paused) advances one frame.
- Reset rebuilds the scene from the initial state.
"""
import sys
import time
import numpy as np
import warp as wp

wp.init()
wp.set_device("metal:0")

sys.path.insert(0, "/Users/viani/dev/mujoco-test/steel_demo")
import steel_sleeping_metal as ex
from newton.viewer import ViewerViser


class Args:
    headless = False
    num_frames = 100000
    device = "metal:0"
    viewer = "viser"
    test = False
    benchmark = False
    render_fps = 60


def build(viewer, args):
    return ex.Example(viewer, args)


viewer = ViewerViser(port=8080, label="Steel Sleeping (Metal)", paused=False)
args = Args()
example = build(viewer, args)

# Wire the Reset button: rebuild the example from scratch.
_state = {"example": example}


def _reset():
    _state["example"] = build(viewer, args)


viewer.set_reset_callback(_reset)

print("Steel sleeping + viser ready on Metal", flush=True)
print("Open http://localhost:8080 — Pause/Step/Reset are live", flush=True)

while viewer.is_running():
    t0 = time.perf_counter()
    ex_ = _state["example"]
    if viewer.should_step():
        ex_.step()
    ex_.render()
    dt = time.perf_counter() - t0
    time.sleep(max(0.0, 1.0 / args.render_fps - dt))
