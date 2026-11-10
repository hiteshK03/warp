"""Extended Newton benchmark sweep: robots x solvers x world counts x device.
Appends JSONL rows to results_sweep.jsonl. Per-config try/except: failures recorded, sweep continues."""
import sys, time, json, traceback
import numpy as np
import warp as wp

device = sys.argv[1]
robots = sys.argv[2].split(",")
solvers = sys.argv[3].split(",")
world_counts = [int(x) for x in sys.argv[4].split(",")]

wp.init()
wp.set_device(device)
import newton, newton.solvers, newton.examples

ASSETS = {
    "cartpole": ("cartpole.urdf", "add_urdf"),
    "ant": ("nv_ant.xml", "add_mjcf"),
    "humanoid": ("nv_humanoid.xml", "add_mjcf"),
    "quadruped": ("quadruped.urdf", "add_urdf"),
}
SOLVER_CLS = {
    "xpbd": newton.solvers.SolverXPBD,
    "featherstone": newton.solvers.SolverFeatherstone,
    "semiimplicit": newton.solvers.SolverSemiImplicit,
    "mujoco": newton.solvers.SolverMuJoCo,
    "kamino": newton.solvers.SolverKamino,
}

dt = 0.01
bench_seconds = 2.0
warmup_steps = 10
out = open("results_sweep.jsonl", "a")

NEEDS_REGISTER = {"kamino", "mujoco"}

def make_solver(solver_name, model, single_builder):
    if solver_name == "kamino":
        cfg = newton.solvers.SolverKamino.Config.from_model(
            model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        cfg.dvi.bilateral_solver_type = "LLTBRCM"
        cfg.dvi.max_alternating_iterations = 8
        cfg.dvi.bilateral_solve_interval = 8
        return newton.solvers.SolverKamino(model, config=cfg)
    return SOLVER_CLS[solver_name](model)

def run_one(robot, solver_name, nw):
    fname, method = ASSETS[robot]
    single = newton.ModelBuilder()
    if solver_name == "kamino":
        newton.solvers.SolverKamino.register_custom_attributes(single)
    elif solver_name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(single)
    getattr(single, method)(open(newton.examples.get_asset(fname)).read(),
                            **({"joint_damping_mode": "drive"} if method == "add_urdf" else {}))
    single.add_ground_plane()
    b = newton.ModelBuilder()
    for _ in range(nw):
        b.add_world(single)
    t_build0 = time.perf_counter()
    model = b.finalize()
    t_build = time.perf_counter() - t_build0
    nbody = model.body_count

    solver = make_solver(solver_name, model, single)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()

    def step():
        nonlocal s0, s1
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
    sps = nsteps / elapsed
    return {"device": device, "robot": robot, "solver": solver_name, "worlds": nw,
            "nbody_per_world": nbody // max(nw,1), "build_s": round(t_build,2),
            "batch_steps_s": round(sps,1), "env_steps_s": round(sps * nw), "error": None}

for robot in robots:
    for solver_name in solvers:
        for nw in world_counts:
            rec = {"device": device, "robot": robot, "solver": solver_name, "worlds": nw}
            try:
                rec = run_one(robot, solver_name, nw)
                print(f"[OK] {device:8s} {robot:10s} {solver_name:13s} w={nw:5d}  "
                      f"batch={rec['batch_steps_s']:9.1f}/s  env={rec['env_steps_s']:12,d}/s  build={rec['build_s']}s", flush=True)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"[FAIL] {device:8s} {robot:10s} {solver_name:13s} w={nw:5d}  {rec['error'][:120]}", flush=True)
            out.write(json.dumps(rec) + "\n"); out.flush()
out.close()
