"""Hot/cold block using real MPM p2g2p; no trained model or renderer needed."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = str(ROOT / ".thermal-cache")
wp.init()

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.engine_utils import save_data_at_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if wp.is_cuda_available() else "cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--output", default=str(ROOT / "output" / "thermal_demo"))
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0:
        parser.error("steps and dt must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    n_grid = 32
    axes = [np.arange(8, 24) / n_grid, np.arange(13, 19) / n_grid, np.arange(13, 19) / n_grid]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    initial_temperature = np.where(positions[:, 0] < 0.5, 80., 20.).astype(np.float32)
    n = len(positions)
    solver = MPM_Simulator_WARP(n, n_grid=n_grid, device=args.device)
    state = solver.mpm_state
    state.particle_x = wp.from_numpy(positions, dtype=wp.vec3, device=args.device)
    state.particle_temperature = wp.from_numpy(initial_temperature, dtype=float, device=args.device)
    state.particle_vol.fill_((1 / n_grid) ** 3)
    state.particle_selection.fill_(1)  # Freeze mechanics, retain heat exchange.
    identity = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    state.particle_F = wp.from_numpy(identity, dtype=wp.mat33, device=args.device)
    state.particle_F_trial = wp.from_numpy(identity, dtype=wp.mat33, device=args.device)
    solver.set_parameters_dict({"E": 1000., "nu": 0.2, "density": 200.,
                                "thermal": {"diffusivity": 0.01}}, device=args.device)
    solver.finalize_mu_lam(device=args.device)
    mass = state.particle_mass.numpy().astype(np.float64)
    initial_energy = float(mass @ initial_temperature)
    save_data_at_frame(solver, str(output), 0, save_to_ply=True, save_to_h5=True)
    history = [initial_temperature.copy()]
    for step in range(args.steps):
        solver.p2g2p(step, args.dt, device=args.device)
        history.append(state.particle_temperature.numpy())
    temperature = history[-1]
    save_data_at_frame(solver, str(output), args.steps, save_to_ply=True, save_to_h5=True)
    np.savez_compressed(output / "history.npz", position=positions,
                        temperature=np.stack(history), dt=np.array(args.dt))
    report = {"device": args.device, "particles": n, "steps": args.steps,
              "time": solver.time, "initial_hot_mean": 80., "initial_cold_mean": 20.,
              "final_hot_mean": float(temperature[initial_temperature == 80].mean()),
              "final_cold_mean": float(temperature[initial_temperature == 20].mean()),
              "temperature_min": float(temperature.min()), "temperature_max": float(temperature.max()),
              "relative_heat_error": abs(float(mass @ temperature) - initial_energy) / initial_energy,
              "max_position_change": float(np.abs(state.particle_x.numpy() - positions).max()),
              "material": solver.mpm_model.material}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
