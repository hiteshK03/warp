"""End-to-end record-cache test:
Run 1 (cold): whole-module fails -> isolation -> record written.
Run 2 (warm, same cache): record hit -> bundle load, skip whole-module compile.
Verify: good kernels launch, bad refused, record file exists, second run is faster.
"""
import os
import sys
import time
import numpy as np
import warp as wp
from warp._src import context as ctx
import q5_kernels as K

wp.init()
DEV = "metal:0"
wp.set_device(DEV)
m = K.m
a = wp.array([1.0], dtype=float, device=DEV)
o = wp.zeros(1, dtype=float, device=DEV)

run = sys.argv[1] if len(sys.argv) > 1 else "1"
t0 = time.perf_counter()
wp.launch(K.goods[0], dim=1, inputs=[a, o, 2.0], device=DEV)
wp.synchronize_device(DEV)
dt = time.perf_counter() - t0
print(f"run {run}: good_0 launch = {dt*1000:.1f}ms, value={float(o.numpy()[0])} (expect 2.0)")

# bad kernel must be refused
try:
    wp.launch(K.bad, dim=1, inputs=[o], device=DEV)
    print("run {run}: bad LAUNCHED (UNEXPECTED)")
except RuntimeError as e:
    print(f"run {run}: bad refused (correct)")

# record file present?
mh = m.get_module_hash(256)
path = ctx._metal_iso_record_path(mh)
print(f"run {run}: record exists={os.path.exists(path)}")
if os.path.exists(path):
    import json
    rec = json.load(open(path))
    print(f"  refused={list(rec['refused'])} backward_refused={rec['backward_refused']} bundle_failed={rec['bundle_failed']}")

# all good kernels launch correctly
for j, g in enumerate(K.goods):
    oo = wp.zeros(1, dtype=float, device=DEV)
    wp.launch(g, dim=1, inputs=[a, oo, 2.0], device=DEV)
    wp.synchronize_device(DEV)
    expect = 1.0 * 2.0 + j
    got = float(oo.numpy()[0])
    assert abs(got - expect) < 1e-5, f"good_{j}: {got} != {expect}"
print(f"run {run}: all 6 good kernels correct")
