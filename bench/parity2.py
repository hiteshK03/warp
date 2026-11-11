import sys
import numpy as np
import warp as wp
device = sys.argv[1]
wp.init(); wp.set_device(device)
import newton, newton.solvers, newton.examples

b = newton.ModelBuilder()
single = newton.ModelBuilder()
single.add_urdf(open(newton.examples.get_asset("quadruped.urdf")).read(),
                joint_damping_mode="drive", xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.6)))
single.add_ground_plane()
b.add_world(single)
model = b.finalize()

solver = newton.solvers.SolverFeatherstone(model)
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
s0, s1 = model.state(), model.state()

bq = s0.body_q.numpy()
print(f"[{device}] t=0 base z: {bq[0][2]:.4f} nan: {np.isnan(bq).any()}")
dt = 0.002
for i in range(1, 3):
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, None, contacts, dt)
    s0, s1 = s1, s0
    if i in (1, 2):
        bq = s0.body_q.numpy().astype(np.float64)
        print(f"[{device}] step {i:3d}: base z={bq[0][2]:.4f} nan={np.isnan(bq).any()}")
np.save(f"/tmp/parity2_{device.replace(':','_')}.npy", s0.body_q.numpy().astype(np.float64))
