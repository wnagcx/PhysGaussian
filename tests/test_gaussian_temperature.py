"""Temperature persistence and Gaussian indexing without a compiled renderer.

Only the unused KNN extension import is stubbed. All Gaussian temperature,
PLY persistence, pruning, cloning and coordinate transforms run real code.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "gaussian-splatting"
sys.path.insert(0, str(GS))
# GaussianModel only uses this extension when creating points from a PCD.
try:
    import simple_knn._C
except ImportError:
    module = types.ModuleType("simple_knn._C")
    def unused_knn(*args):
        raise AssertionError("KNN is not exercised by these persistence tests")
    module.distCUDA2 = unused_knn
    sys.modules["simple_knn._C"] = module

spec = importlib.util.spec_from_file_location("thermal_gaussian_model", GS / "scene" / "gaussian_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GaussianModel = module.GaussianModel

from utils.thermal_render import temperature_to_rgb, synchronize_gaussian_temperature, save_temperature_frame


def gaussian_model():
    model = GaussianModel(0)
    for attr, value in {"_xyz": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                        "_features_dc": torch.zeros((3, 1, 3)),
                        "_features_rest": torch.zeros((3, 0, 3)),
                        "_opacity": [[1], [1], [1]],
                        "_scaling": torch.full((3, 3), -5.0),
                        "_rotation": [[1, 0, 0, 0]] * 3}.items():
        setattr(model, attr, torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32)))
    model.set_temperature([20.0, 40.0, 80.0], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    groups = [{"params": [getattr(model, attr)], "name": name} for attr, name in
              [("_xyz", "xyz"), ("_features_dc", "f_dc"), ("_features_rest", "f_rest"),
               ("_opacity", "opacity"), ("_scaling", "scaling"), ("_rotation", "rotation")]]
    model.optimizer = torch.optim.Adam(groups)
    model.xyz_gradient_accum = torch.zeros((3, 1))
    model.denom = torch.zeros((3, 1))
    model.max_radii2D = torch.zeros(3)
    model.percent_dense = 1.0
    return model


class GaussianTemperatureTests(unittest.TestCase):
    def test_ply_round_trip_and_old_model_defaults(self):
        model = gaussian_model()
        directory = ROOT / ".thermal-test-output" / "gaussians"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "temperature.ply"
        model.save_ply(str(path))
        restored = GaussianModel(0)
        restored.load_ply(str(path), device="cpu")
        torch.testing.assert_close(restored.get_temperature, model.get_temperature)
        torch.testing.assert_close(restored.get_temperature_gradient, model.get_temperature_gradient)
        ply = PlyData.read(path)
        old_fields = [name for name in ply['vertex'].data.dtype.names if not name.startswith('temperature')]
        old_dtype = [(name, 'f4') for name in old_fields]
        old_data = np.empty(3, dtype=old_dtype)
        for name in old_fields:
            old_data[name] = ply['vertex'][name]
        legacy = directory / 'legacy.ply'
        PlyData([PlyElement.describe(old_data, 'vertex')]).write(legacy)
        restored.load_ply(str(legacy), device="cpu")
        torch.testing.assert_close(restored.get_temperature, torch.full((3, 1), 20.0))
        torch.testing.assert_close(restored.get_temperature_gradient, torch.zeros((3, 3)))

    def test_pruning_and_cloning_preserve_attributes(self):
        model = gaussian_model()
        model.prune_points(torch.tensor([False, True, False]))
        torch.testing.assert_close(model.get_temperature[:, 0], torch.tensor([20.0, 80.0]))
        model.densify_and_clone(torch.ones((2, 1)), 0.5, 1.0)
        torch.testing.assert_close(model.get_temperature[:, 0], torch.tensor([20.0, 80.0, 20.0, 80.0]))
        torch.testing.assert_close(model.get_temperature_gradient[2:], model.get_temperature_gradient[:2])

    def test_mapping_and_gradient_coordinate_conversion(self):
        model = gaussian_model()
        class Solver:
            def export_particle_temperature_to_torch(self):
                return torch.tensor([100.0, 200.0, 999.0])  # Last particle is filling.
            def export_particle_temperature_gradient_to_torch(self):
                return torch.tensor([[1, 0, 0], [0, 1, 0], [9, 9, 9]], dtype=torch.float32)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        synchronize_gaussian_temperature(model, Solver(), torch.tensor([2, 0]), [rotation], 2.0)
        torch.testing.assert_close(model.get_temperature[:, 0], torch.tensor([200., 40., 100.]))
        torch.testing.assert_close(model.get_temperature_gradient[[2, 0]], torch.tensor([[0., -2., 0.], [2., 0., 0.]]))

    def test_display_does_not_modify_temperature(self):
        temperature = torch.tensor([-50.0, 20.0, 100.0, 200.0])
        before = temperature.clone()
        rgb = temperature_to_rgb(temperature)
        self.assertEqual(rgb.shape, (4, 3))
        self.assertTrue(torch.all((rgb >= 0) & (rgb <= 1)))
        torch.testing.assert_close(temperature, before)


if __name__ == "__main__":
    unittest.main()
