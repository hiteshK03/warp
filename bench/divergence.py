"""Per-step CPU-vs-Metal divergence growth on the patched quadruped manifold path.
If the fixedarray manifold code were wrong, divergence would be large at step 1.
Chaotic float drift instead starts tiny and grows."""
import subprocess, sys
import numpy as np

steps_list = [1, 2, 5, 10, 20, 50]
src = open("parity_diverge.py").read() if False else None

template = '''
import numpy as np, warp as wp
wp.init(); wp.set_device("{dev}")
import newton, newton.solvers, newton.examples
b = newton.ModelBuilder(); s = newton.ModelBuilder()
s.add_urdf(open(newton.examples.get_asset("quadruped.urdf")).read(),
           joint_damping_mode="drive", xform=wp.transform(p=wp.vec3(0,0,0.6)))
s.add_ground_plane(); b.add_world(s); m = b.finalize()
solver = newton.solvers.SolverFeatherstone(m)
pipe = newton.CollisionPipeline(m); c = pipe.contacts()
s0, s1 = m.state(), m.state()
for i in range({n}):
    pipe.collide(s0, c); solver.step(s0, s1, None, c, 0.002); s0, s1 = s1, s0
np.save("/tmp/div_{dev2}_{n}.npy", s0.body_q.numpy().astype(np.float64))
'''

import tempfile, os
for n in steps_list:
    for dev, tag in [("cpu", "cpu"), ("metal:0", "met")]:
        code = template.format(dev=dev, dev2=tag, n=n)
        p = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=".")
        p.write(code); p.close()
        subprocess.run([".venv-metal/bin/python", p.name], capture_output=True)
        os.unlink(p.name)
    a = np.load(f"/tmp/div_cpu_{n}.npy"); b = np.load(f"/tmp/div_met_{n}.npy")
    d = np.abs(a - b)
    nan = np.isnan(a).any() or np.isnan(b).any()
    print(f"steps={n:3d}  max|cpu-metal|={d.max():.3e}  mean={d.mean():.3e}  nan={nan}")
