"""Numerical tests of the actual Warp kernels; no renderer is required.

Run: python -m unittest discover -s tests -v
Set THERMAL_TEST_DEVICE=cpu or cuda:0 to select a backend.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import warp as wp

wp.config.kernel_cache_dir = str(ROOT / ".thermal-cache")
wp.init()

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from thermal_utils import ThermalSolver, validate_thermal_config, thermal_g2p
from mpm_solver_warp.engine_utils import save_data_at_frame

DEVICE = os.environ.get("THERMAL_TEST_DEVICE", "cuda:0" if wp.is_cuda_available() else "cpu")


def particles(positions, temperatures, gradients=None, thermal=None):
    positions = np.asarray(positions, dtype=np.float32)
    n = len(positions)
    s = MPM_Simulator_WARP(n, n_grid=16, device=DEVICE)
    s.mpm_state.particle_x = wp.from_numpy(positions, dtype=wp.vec3, device=DEVICE)
    s.mpm_state.particle_temperature = wp.from_numpy(np.asarray(temperatures, dtype=np.float32), dtype=float, device=DEVICE)
    if gradients is not None:
        s.mpm_state.particle_temperature_gradient = wp.from_numpy(np.asarray(gradients, dtype=np.float32), dtype=wp.vec3, device=DEVICE)
    s.mpm_state.particle_vol.fill_(0.001)
    s.mpm_state.particle_F_trial = wp.from_numpy(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=DEVICE)
    params = {"density": 1.0, "E": 10.0, "nu": 0.2}
    if thermal is not None:
        params["thermal"] = thermal
    s.set_parameters_dict(params, device=DEVICE)
    s.finalize_mu_lam(device=DEVICE)
    return s


def grid_solver(mass, temperature, alpha=0.1, boundaries=None):
    n = len(mass)
    t = ThermalSolver(n, DEVICE, {"diffusivity": alpha, "boundary_conditions": boundaries or []})
    t.grid.mass = wp.from_numpy(np.asarray(mass, dtype=np.float32), dtype=float, device=DEVICE)
    t.grid.temperature = wp.from_numpy(np.asarray(temperature, dtype=np.float32), dtype=float, device=DEVICE)
    model = MPM_Simulator_WARP(1, n_grid=n, device=DEVICE).mpm_model
    return t, model


class ThermalTests(unittest.TestCase):
    def test_constant_temperature_stays_constant(self):
        rng = np.random.default_rng(0)
        s = particles(rng.uniform(0.3, 0.7, (40, 3)), np.full(40, 42.0), thermal={"diffusivity": 0.05})
        for step in range(8):
            s.p2g2p(step, 0.001, device=DEVICE)
        np.testing.assert_allclose(s.mpm_state.particle_temperature.numpy(), 42.0, atol=3e-4)
        np.testing.assert_allclose(s.mpm_state.particle_temperature_gradient.numpy(), 0.0, atol=3e-3)

    def test_affine_transfer_preserves_linear_field(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.25, 0.75, (80, 3)).astype(np.float32)
        grad = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        temp = 20.0 + x @ grad
        s = particles(x, temp, np.tile(grad, (len(x), 1)), thermal={"diffusivity": 0.0})
        t = s.thermal_solver
        t.p2g(s.mpm_state, s.mpm_model, s.n_particles)
        t.g2p(s.mpm_state, s.mpm_model, s.n_particles)
        np.testing.assert_allclose(s.mpm_state.particle_temperature.numpy(), temp, atol=2e-5)
        np.testing.assert_allclose(s.mpm_state.particle_temperature_gradient.numpy(), np.tile(grad, (len(x), 1)), atol=4e-4)

    def test_d3q7_impulse_and_diffusivity_scaling(self):
        mass = np.ones((8, 8, 8), np.float32)
        initial = np.zeros_like(mass)
        initial[4, 4, 4] = 100.0
        for alpha in (0.01, 0.02):
            t, m = grid_solver(mass, initial, alpha)
            dt = 0.01
            r = alpha * dt / m.dx ** 2
            t.advance(m, 0.0, dt)
            result = t.grid.temperature.numpy()
            self.assertAlmostEqual(float(result[4, 4, 4]), 100 * (1 - 6 * r), places=4)
            self.assertAlmostEqual(float(result[5, 4, 4]), 100 * r, places=4)
            self.assertAlmostEqual(float(result.sum()), 100.0, places=4)

    def test_unequal_mass_conservation_and_insulation(self):
        rng = np.random.default_rng(2)
        mass = rng.uniform(0.1, 2.0, (8, 8, 8)).astype(np.float32)
        mass[4, :, :] = 0.0  # Insulating gap separates two objects.
        temp = np.full_like(mass, 20.0)
        temp[:4] = 100.0
        t, m = grid_solver(mass, temp)
        before = np.sum(mass.astype(np.float64) * temp)
        t.advance(m, 0.0, 0.1)  # Also exercises automatic substeps.
        result = t.grid.temperature.numpy()
        after = np.sum(mass.astype(np.float64) * result)
        self.assertLess(abs(after - before) / before, 2e-6)
        np.testing.assert_allclose(result[:4], 100.0, atol=1e-4)
        np.testing.assert_allclose(result[5:], 20.0, atol=1e-4)

    def test_hot_to_cold_and_grid_energy_conservation(self):
        rng = np.random.default_rng(3)
        mass = rng.uniform(0.1, 2.0, (8, 8, 8)).astype(np.float32)
        temp = np.full_like(mass, 20.0)
        temp[4, 4, 4] = 100.0
        t, m = grid_solver(mass, temp)
        before = np.sum(mass.astype(np.float64) * temp)
        t.advance(m, 0.0, 0.05)
        result = t.grid.temperature.numpy()
        self.assertLess(result[4, 4, 4], 100)
        self.assertGreater(result[5, 4, 4], 20)
        self.assertGreaterEqual(result.min(), 20 - 1e-4)
        self.assertLessEqual(result.max(), 100)
        self.assertLess(abs(np.sum(mass.astype(np.float64) * result) - before) / before, 2e-6)

    def test_neumann_cosine_matches_heat_equation(self):
        n = 20
        mass = np.ones((n, n, n), np.float32)
        profile = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        initial = np.broadcast_to(20 + 5 * profile[:, None, None], mass.shape).copy()
        t, m = grid_solver(mass, initial, alpha=0.1)
        duration = 0.1
        t.advance(m, 0, duration)
        expected = 20 + 5 * profile * np.exp(-0.1 * np.pi ** 2 * duration)
        np.testing.assert_allclose(t.grid.temperature.numpy()[:, 10, 10], expected, atol=0.005)

    def test_heat_source_schedule_and_fixed_boundary(self):
        mass = np.ones((8, 8, 8), np.float32)
        temp = np.full_like(mass, 20.0)
        sources = [{"type": "heat_source", "point": [0.5]*3, "size": [1.0]*3,
                    "rate": 10.0, "start_time": 0.025, "end_time": 0.075},
                   {"type": "fixed_temperature", "point": [0, 0, 0], "size": [0.01]*3,
                    "temperature": 80.0}]
        t, m = grid_solver(mass, temp, alpha=0, boundaries=sources)
        t.advance(m, 0.0, 0.1)
        result = t.grid.temperature.numpy()
        self.assertAlmostEqual(float(result[3, 3, 3]), 20.5, places=4)
        self.assertAlmostEqual(float(result[0, 0, 0]), 80, places=4)

    def test_temperature_does_not_change_mechanics_or_material(self):
        x = [[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]]
        a = particles(x, [20.0, 20.0])
        b = particles(x, [20.0, 10000.0], thermal={"diffusivity": 0.1})
        for step in range(5):
            a.p2g2p(step, 0.001, device=DEVICE)
            b.p2g2p(step, 0.001, device=DEVICE)
        np.testing.assert_array_equal(a.mpm_state.particle_x.numpy(), b.mpm_state.particle_x.numpy())
        np.testing.assert_array_equal(a.mpm_state.particle_F_trial.numpy(), b.mpm_state.particle_F_trial.numpy())
        self.assertEqual(b.mpm_model.material, a.mpm_model.material)
        self.assertGreater(b.mpm_state.particle_temperature.numpy()[0], 20)

    def test_particle_heat_conservation_over_transfers(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0.25, 0.75, (80, 3))
        temp = rng.uniform(20, 100, len(x))
        s = particles(x, temp, thermal={"diffusivity": 0.1})
        mass = rng.uniform(0.1, 2, len(x)).astype(np.float32)
        s.mpm_state.particle_mass = wp.from_numpy(mass, dtype=float, device=DEVICE)
        before = np.sum(mass.astype(np.float64) * s.mpm_state.particle_temperature.numpy())
        for _ in range(10):
            s.thermal_solver.p2g(s.mpm_state, s.mpm_model, s.n_particles)
            s.thermal_solver.advance(s.mpm_model, 0, 0.001)
            s.thermal_solver.g2p(s.mpm_state, s.mpm_model, s.n_particles)
        after = np.sum(mass.astype(np.float64) * s.mpm_state.particle_temperature.numpy())
        self.assertLess(abs(after - before) / before, 2e-6)

    def test_thermal_export_contains_temperature(self):
        from plyfile import PlyData
        import h5py
        s = particles([[0.5]*3], [73.0])
        directory = ROOT / ".thermal-test-output" / "export"
        directory.mkdir(parents=True, exist_ok=True)
        save_data_at_frame(s, str(directory), 0, save_to_ply=True, save_to_h5=True)
        ply = PlyData.read(directory / "sim_0000000000.ply")
        self.assertEqual(float(ply['vertex']['temperature'][0]), 73.0)
        with h5py.File(directory / "sim_0000000000.h5") as data:
            np.testing.assert_array_equal(data['temperature'][:], [73.0])
            self.assertEqual(data['temperature_gradient'].shape, (3, 1))

    def test_invalid_parameters_fail(self):
        for config in ({"diffusivity": -1}, {"diffusivity": float('nan')},
                       {"enabled": "false"}, {"initial_regions": [{"point": [0]*3, "size": [-1]*3, "temperature": 20}]}):
            with self.assertRaises(ValueError):
                validate_thermal_config(config)


if __name__ == "__main__":
    unittest.main()
