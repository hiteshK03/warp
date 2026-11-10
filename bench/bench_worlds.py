"""Newton batched-worlds RL-style benchmark: N copies of the MuJoCo 'ant' robot,
SolverXPBD (pure Warp), metal:0 vs cpu. Measures sim steps/sec of whole batch."""
import sys, time
import warp as wp

device = sys.argv[1]
world_counts = [int(x) for x in sys.argv[2:]] or [1, 16, 64, 256, 1024]

wp.init()
wp.set_device(device)
import newton, newton.solvers, newton.examples

ANT_XML = open(newton.examples.get_asset("nv_ant.xml")).read()

def build(nworlds):
    b = newton.ModelBuilder()
    single = newton.ModelBuilder()
    single.add_mjcf(ANT_XML)
    single.add_ground_plane()
    for w in range(nworlds):
        b.add_world(single)
    model = b.finalize()
    return model

dt = 0.01
warmup_steps = 20
bench_seconds = 3.0

results = []
for nw in world_counts:
    model = build(nw)
    solver = newton.solvers.SolverXPBD(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()

    def step():
        global s0, s1
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0

    for _ in range(warmup_steps):
        step()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    nsteps = 0
    while time.perf_counter() - t0 < bench_seconds:
        for _ in range(10):
            step()
        wp.synchronize_device(device)
        nsteps += 10
    elapsed = time.perf_counter() - t0
    sps = nsteps / elapsed                    # batch steps/sec
    env_sps = sps * nw                        # env steps/sec (RL metric)
    results.append((nw, sps, env_sps))
    print(f"[{device:8s}] worlds={nw:5d}  batch_steps/s={sps:9.1f}  env_steps/s={env_sps:12.0f}", flush=True)

import json
json.dump(results, open(f"results_{device.replace(':','_')}.json", "w"))
