"""Steel sleeping demo with the VISER web viewer on Metal.
Serves an interactive 3D view at http://localhost:8080 — open in a browser.
Runs the simulation live; sleeping bodies turn slate blue, the kicked tower wakes.
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


viewer = ViewerViser(port=8080, label="Steel Sleeping (Metal)")
example = ex.Example(viewer, Args())
print("Steel sleeping + viser viewer ready on Metal", flush=True)
print("Open http://localhost:8080 in a browser", flush=True)

frame = 0
while True:
    example.step()
    example.render()
    frame += 1
    if frame % 120 == 0:
        awake = example.observables.body_awake.numpy()
        print(f"frame {frame}: awake {int(awake.sum())}/{len(awake)}", flush=True)
    # pace to ~60 fps real-time so the viewer is watchable
    time.sleep(1.0 / 60.0)
