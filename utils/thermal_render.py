"""Temperature visualization and full-Gaussian snapshot helpers."""
import os
import numpy as np
import torch


def temperature_to_rgb(temperature, minimum=20.0, maximum=100.0):
    if maximum <= minimum:
        raise ValueError("temperature display maximum must exceed minimum")
    t = ((temperature.reshape(-1) - minimum) / (maximum - minimum)).clamp(0, 1)
    # Blue -> cyan -> yellow -> red. Clamp for display only, never for physics.
    return torch.stack(((1.5 - (4 * t - 3).abs()).clamp(0, 1),
                        (1.5 - (4 * t - 2).abs()).clamp(0, 1),
                        (1.5 - (4 * t - 1).abs()).clamp(0, 1)), dim=1).contiguous()


def synchronize_gaussian_temperature(gaussians, solver, gaussian_indices,
                                      rotation_matrices, scale_origin):
    count = gaussian_indices.numel()
    temperature = solver.export_particle_temperature_to_torch()[:count]
    gradient = solver.export_particle_temperature_gradient_to_torch()[:count] * scale_origin
    for rotation in reversed(rotation_matrices):
        gradient = gradient @ rotation
    gaussians._temperature[gaussian_indices, 0] = temperature
    gaussians._temperature_gradient[gaussian_indices] = gradient


def save_temperature_frame(directory, frame, gaussians, solver, gaussian_indices):
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, f"{frame:04d}.npz"),
                        time=np.array(solver.time),
                        gaussian_temperature=gaussians.get_temperature.detach().cpu().numpy(),
                        gaussian_temperature_gradient=gaussians.get_temperature_gradient.detach().cpu().numpy(),
                        simulated_gaussian_indices=gaussian_indices.detach().cpu().numpy(),
                        particle_temperature=solver.mpm_state.particle_temperature.numpy(),
                        particle_temperature_gradient=solver.mpm_state.particle_temperature_gradient.numpy(),
                        particle_position=solver.mpm_state.particle_x.numpy())
