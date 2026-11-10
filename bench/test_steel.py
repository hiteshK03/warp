import time, sys
import warp as wp
wp.init()
dev = sys.argv[1] if len(sys.argv) > 1 else "metal:0"
wp.set_device(dev)
import newton, newton.solvers, newton.examples
print("newton:", newton.__version__, "| device:", wp.get_device())
print("has SolverSteel:", hasattr(newton.solvers, "SolverSteel"))

ANT = open(newton.examples.get_asset("nv_ant.xml")).read()
dt = 0.01
nw = int(sys.argv[2]) if len(sys.argv) > 2 else 64

single = newton.ModelBuilder()
if hasattr(newton.solvers.SolverSteel, "register_custom_attributes"):
    newton.solvers.SolverSteel.register_custom_attributes(single)
single.add_mjcf(ANT)
single.add_ground_plane()
b = newton.ModelBuilder()
for _ in range(nw): b.add_world(single)
model = b.finalize(skip_validation_inertia=True)

solver = newton.solvers.SolverSteel(model)
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
s0, s1 = model.state(), model.state()
def step():
    global s0, s1
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, None, contacts, dt)
    s0, s1 = s1, s0

for _ in range(10): step()
wp.synchronize_device(dev)
t0 = time.perf_counter(); n = 0
while time.perf_counter() - t0 < 2.0:
    for _ in range(10): step()
    wp.synchronize_device(dev); n += 10
sps = n / (time.perf_counter() - t0)
print(f"[steel {dev} w={nw}] batch={sps:.1f}/s env={sps*nw:,.0f}/s")
