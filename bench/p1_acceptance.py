"""P1 acceptance: Newton finalize() with FULL inertia validation on metal:0.
No skip flags, no monkeypatching — the fp64 kernels must run via df64.
Then step with SolverSteel if available, else Featherstone.
"""
import sys
import numpy as np
import warp as wp

device = sys.argv[1] if len(sys.argv) > 1 else "metal:0"
wp.init()
wp.set_device(device)

import newton, newton.solvers, newton.examples

print(f"[{device}] newton {newton.__version__}")

b = newton.ModelBuilder()
s = newton.ModelBuilder()
s.add_urdf(open(newton.examples.get_asset("quadruped.urdf")).read(),
           joint_damping_mode="drive", xform=wp.transform(p=wp.vec3(0, 0, 0.6)))
s.add_ground_plane()
b.add_world(s)

# FULL validation: runs validate_and_correct_inertia_kernel (fp64 eig3)
# and compute_body_inverses_kernel (fp64 inverse). No skip flags.
model = b.finalize()
print(f"[{device}] finalize() with full fp64 inertia validation: OK")
inv = model.body_inv_inertia.numpy()
print(f"[{device}] body_inv_inertia[1] diag = {np.diag(inv[1])}")
assert not np.isnan(inv).any(), "NaN in inverses"

solvers = ["SolverSteel", "SolverFeatherstone"]
for name in solvers:
    if not hasattr(newton.solvers, name):
        print(f"[{device}] {name}: not present")
        continue
    cls = getattr(newton.solvers, name)
    try:
        mb = newton.ModelBuilder()
        sb = newton.ModelBuilder()
        if hasattr(cls, "register_custom_attributes"):
            cls.register_custom_attributes(sb)
        sb.add_mjcf(open(newton.examples.get_asset("nv_ant.xml")).read())
        sb.add_ground_plane()
        mb.add_world(sb)
        m = mb.finalize()
        solver = cls(m)
        pipe = newton.CollisionPipeline(m)
        c = pipe.contacts()
        s0, s1 = m.state(), m.state()
        for _ in range(20):
            pipe.collide(s0, c)
            solver.step(s0, s1, None, c, 0.005)
            s0, s1 = s1, s0
        wp.synchronize_device(device)
        q = s0.body_q.numpy()
        print(f"[{device}] {name}: 20 steps OK, nan={bool(np.isnan(q).any())}")
    except Exception as e:
        print(f"[{device}] {name}: FAILED {type(e).__name__}: {str(e)[:300]}")
