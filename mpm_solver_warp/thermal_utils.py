"""Affine thermal P2G/G2P and conservative D3Q7 diffusion LBM.

The BGK relaxation time is one. Directional weights are r=alpha*dt/dx**2
and the rest weight is 1-6*r, so small MPM time steps do not introduce a
fixed amount of diffusion. Partial link reflection accounts for unequal
nodal masses. Empty cells and domain faces are insulating. Temperature
advection is supplied by the moving MPM particles, not applied twice here.
"""

import math
import warp as wp
from warp_utils import MPMStateStruct, MPMModelStruct


@wp.struct
class ThermalGrid:
    mass: wp.array(dtype=float, ndim=3)
    energy: wp.array(dtype=float, ndim=3)
    temperature: wp.array(dtype=float, ndim=3)
    populations: wp.array(dtype=float, ndim=4)


@wp.func
def thermal_weights(fx: wp.vec3):
    a = wp.vec3(1.5) - fx
    b = fx - wp.vec3(1.0)
    c = fx - wp.vec3(0.5)
    return wp.mat33(0.5 * wp.cw_mul(a, a),
                    wp.vec3(0.75) - wp.cw_mul(b, b),
                    0.5 * wp.cw_mul(c, c))


@wp.func
def inside_grid(x: int, y: int, z: int, n: int):
    return x >= 0 and y >= 0 and z >= 0 and x < n and y < n and z < n


@wp.kernel
def thermal_p2g(state: MPMStateStruct, model: MPMModelStruct, grid: ThermalGrid):
    p = wp.tid()
    gp = state.particle_x[p] * model.inv_dx
    bx = wp.int(wp.floor(gp[0] - 0.5))
    by = wp.int(wp.floor(gp[1] - 0.5))
    bz = wp.int(wp.floor(gp[2] - 0.5))
    fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
    w = thermal_weights(fx)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                x = bx + a
                y = by + b
                z = bz + c
                if inside_grid(x, y, z, model.n_grid):
                    weight = w[0, a] * w[1, b] * w[2, c]
                    dpos = (wp.vec3(wp.float(a), wp.float(b), wp.float(c)) - fx) * model.dx
                    mass = weight * state.particle_mass[p]
                    temperature = state.particle_temperature[p] + wp.dot(
                        dpos, state.particle_temperature_gradient[p])
                    wp.atomic_add(grid.mass, x, y, z, mass)
                    wp.atomic_add(grid.energy, x, y, z, mass * temperature)


@wp.kernel
def normalize_thermal_grid(grid: ThermalGrid):
    x, y, z = wp.tid()
    grid.temperature[x, y, z] = 0.0
    if grid.mass[x, y, z] > 1.0e-15:
        grid.temperature[x, y, z] = grid.energy[x, y, z] / grid.mass[x, y, z]


@wp.kernel
def set_particle_temperature_region(state: MPMStateStruct, point: wp.vec3,
                                    size: wp.vec3, temperature: float):
    p = wp.tid()
    d = state.particle_x[p] - point
    if wp.abs(d[0]) <= size[0] and wp.abs(d[1]) <= size[1] and wp.abs(d[2]) <= size[2]:
        state.particle_temperature[p] = temperature
        state.particle_temperature_gradient[p] = wp.vec3(0.0)


@wp.kernel
def thermal_boundary(grid: ThermalGrid, dx: float, point: wp.vec3,
                     size: wp.vec3, value: float, fixed: int):
    x, y, z = wp.tid()
    d = wp.vec3(wp.float(x), wp.float(y), wp.float(z)) * dx - point
    if grid.mass[x, y, z] > 1.0e-15:
        if wp.abs(d[0]) <= size[0] and wp.abs(d[1]) <= size[1] and wp.abs(d[2]) <= size[2]:
            if fixed == 1:
                grid.temperature[x, y, z] = value
            else:
                grid.temperature[x, y, z] = grid.temperature[x, y, z] + value


@wp.func
def lattice_direction(q: int):
    d = wp.vec3i(0, 0, 0)
    if q == 1:
        d = wp.vec3i(1, 0, 0)
    elif q == 2:
        d = wp.vec3i(-1, 0, 0)
    elif q == 3:
        d = wp.vec3i(0, 1, 0)
    elif q == 4:
        d = wp.vec3i(0, -1, 0)
    elif q == 5:
        d = wp.vec3i(0, 0, 1)
    elif q == 6:
        d = wp.vec3i(0, 0, -1)
    return d


@wp.kernel
def lbm_collide(grid: ThermalGrid, r: float):
    x, y, z = wp.tid()
    energy = grid.mass[x, y, z] * grid.temperature[x, y, z]
    # BGK with tau=1: f* = feq. Populations represent mass * temperature
    # (constant, shared specific heat), rather than unweighted temperature.
    grid.populations[x, y, z, 0] = (1.0 - 6.0 * r) * energy
    for q in range(1, 7):
        grid.populations[x, y, z, q] = r * energy


@wp.kernel
def lbm_stream(grid: ThermalGrid, n: int):
    x, y, z = wp.tid()
    mass = grid.mass[x, y, z]
    if mass > 1.0e-15:
        energy = grid.populations[x, y, z, 0]
        for q in range(1, 7):
            d = lattice_direction(q)
            sx = x - d[0]
            sy = y - d[1]
            sz = z - d[2]
            # Full reflection at missing neighbors; partial reflection at
            # unequal capacities gives symmetric min(mi,mj) link fluxes.
            incoming = grid.populations[x, y, z, q]
            if inside_grid(sx, sy, sz, n):
                neighbor_mass = grid.mass[sx, sy, sz]
                if neighbor_mass > 1.0e-15:
                    link_mass = wp.min(mass, neighbor_mass)
                    incoming = (grid.populations[sx, sy, sz, q] * link_mass / neighbor_mass
                                + incoming * (1.0 - link_mass / mass))
            energy = energy + incoming
        grid.temperature[x, y, z] = energy / mass


@wp.kernel
def thermal_g2p(state: MPMStateStruct, model: MPMModelStruct, grid: ThermalGrid):
    p = wp.tid()
    gp = state.particle_x[p] * model.inv_dx
    bx = wp.int(wp.floor(gp[0] - 0.5))
    by = wp.int(wp.floor(gp[1] - 0.5))
    bz = wp.int(wp.floor(gp[2] - 0.5))
    fx = gp - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
    w = thermal_weights(fx)
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    temperature = float(0.0)
    gradient = wp.vec3(0.0)
    weight_sum = float(0.0)
    gradient_sum = wp.vec3(0.0)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                x = bx + a
                y = by + b
                z = bz + c
                if inside_grid(x, y, z, model.n_grid):
                    if grid.mass[x, y, z] > 1.0e-15:
                        weight = w[0, a] * w[1, b] * w[2, c]
                        dweight = wp.vec3(dw[0, a] * w[1, b] * w[2, c],
                                          w[0, a] * dw[1, b] * w[2, c],
                                          w[0, a] * w[1, b] * dw[2, c]) * model.inv_dx
                        value = grid.temperature[x, y, z]
                        temperature = temperature + weight * value
                        gradient = gradient + value * dweight
                        weight_sum = weight_sum + weight
                        gradient_sum = gradient_sum + dweight
    if weight_sum > 1.0e-10:
        temperature = temperature / weight_sum
        state.particle_temperature[p] = temperature
        # Quotient rule only matters for truncated stencils at domain faces.
        state.particle_temperature_gradient[p] = (gradient - temperature * gradient_sum) / weight_sum


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def validate_thermal_config(config):
    """Validate before allocating or changing any simulation state."""
    if not isinstance(config, dict):
        raise ValueError("thermal must be a JSON object")
    result = dict(config)
    if not isinstance(result.get("enabled", True), bool):
        raise ValueError("thermal.enabled must be a boolean")
    result.setdefault("enabled", True)
    result["diffusivity"] = _finite(result.get("diffusivity", 0.01), "diffusivity")
    if result["diffusivity"] < 0:
        raise ValueError("diffusivity must be nonnegative")
    if "initial_temperature" in result:
        result["initial_temperature"] = _finite(result["initial_temperature"], "initial_temperature")
    for key in ("initial_regions", "boundary_conditions"):
        regions = result.get(key, [])
        if not isinstance(regions, list):
            raise ValueError("thermal." + key + " must be a list")
        checked = []
        for region in regions:
            item = dict(region)
            for field in ("point", "size"):
                values = item.get(field, [])
                if len(values) != 3:
                    raise ValueError(field + " must have three components")
                item[field] = [_finite(v, field) for v in values]
            if any(v < 0 for v in item["size"]):
                raise ValueError("thermal region size is a nonnegative half extent")
            if key == "initial_regions":
                item["temperature"] = _finite(item["temperature"], "temperature")
            else:
                kind = item.get("type", "fixed_temperature")
                if kind not in ("fixed_temperature", "heat_source"):
                    raise ValueError("unknown thermal boundary type: " + str(kind))
                item["type"] = kind
                field = "temperature" if kind == "fixed_temperature" else "rate"
                item[field] = _finite(item[field], field)
                item["start_time"] = _finite(item.get("start_time", 0.0), "start_time")
                item["end_time"] = _finite(item.get("end_time", 1.0e30), "end_time")
                if item["end_time"] <= item["start_time"]:
                    raise ValueError("thermal boundary end_time must exceed start_time")
            checked.append(item)
        result[key] = checked
    return result


class ThermalSolver:
    def __init__(self, n_grid, device, config):
        self.config = validate_thermal_config(config)
        self.device = device
        self.n_grid = n_grid
        self.shape = (n_grid, n_grid, n_grid)
        self.grid = ThermalGrid()
        self.grid.mass = wp.zeros(self.shape, dtype=float, device=device)
        self.grid.energy = wp.zeros(self.shape, dtype=float, device=device)
        self.grid.temperature = wp.zeros(self.shape, dtype=float, device=device)
        self.grid.populations = wp.zeros(self.shape + (7,), dtype=float, device=device)

    def p2g(self, state, model, n_particles):
        self.grid.mass.zero_()
        self.grid.energy.zero_()
        wp.launch(thermal_p2g, dim=n_particles, inputs=[state, model, self.grid], device=self.device)
        wp.launch(normalize_thermal_grid, dim=self.shape, inputs=[self.grid], device=self.device)

    def _boundaries(self, time, dt, dx, sources):
        # Add sources before pinning fixed-temperature regions; list order is
        # the priority when multiple fixed-temperature regions overlap.
        for fixed in (False, True):
            for region in self.config["boundary_conditions"]:
                if (region["type"] == "fixed_temperature") == fixed:
                    if region["start_time"] <= time < region["end_time"] and (fixed or sources):
                        value = region["temperature"] if fixed else region["rate"] * dt
                        wp.launch(thermal_boundary, dim=self.shape,
                                  inputs=[self.grid, dx, wp.vec3(region["point"]),
                                          wp.vec3(region["size"]), value, int(fixed)], device=self.device)

    def advance(self, model, time, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("thermal time step must be finite and positive")
        alpha = self.config["diffusivity"]
        # Split at scheduled boundary changes and maintain positive D3Q7
        # weights (r <= .15 < 1/6). No clipping of physical temperatures.
        end = time + dt
        events = {time, end}
        for region in self.config["boundary_conditions"]:
            for key in ("start_time", "end_time"):
                if time < region[key] < end:
                    events.add(region[key])
        events = sorted(events)
        for begin, finish in zip(events[:-1], events[1:]):
            count = max(1, math.ceil(alpha * (finish - begin) / (0.15 * model.dx ** 2)))
            h = (finish - begin) / count
            r = alpha * h / model.dx ** 2
            for k in range(count):
                current = begin + (k + 0.5) * h
                self._boundaries(current, h, model.dx, sources=True)
                if alpha > 0:
                    wp.launch(lbm_collide, dim=self.shape, inputs=[self.grid, r], device=self.device)
                    wp.launch(lbm_stream, dim=self.shape, inputs=[self.grid, self.n_grid], device=self.device)
                self._boundaries(current, h, model.dx, sources=False)

    def g2p(self, state, model, n_particles):
        wp.launch(thermal_g2p, dim=n_particles, inputs=[state, model, self.grid], device=self.device)
