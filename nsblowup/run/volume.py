"""Shared volumetric renderer and sampling helpers (water look, speed-colored).

Emission-absorption ray marching through a periodic speed field on the unit
box.  Color and opacity both encode |u| via a water-like colormap; slow fluid
is transparent, fast fluid glows.  A 2x band-limited spectral upsampling step
(cell-centred origin-registered, see `resample_origin_phase`; the coarse
Nyquist planes are removed) keeps the per-frame cost low.

Used by the movie (`nsblowup.run.video`) and by the manufactured-solution
check (`nsblowup.verify`).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nsblowup.core.spectral import resample_origin_phase

WATER_COLORS = [
    (0.00, (0.010, 0.030, 0.105)),
    (0.25, (0.020, 0.150, 0.380)),
    (0.50, (0.050, 0.450, 0.780)),
    (0.75, (0.350, 0.850, 0.950)),
    (1.00, (0.950, 1.000, 1.000)),
]
WATER = LinearSegmentedColormap.from_list("water", WATER_COLORS)
LUT = WATER(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
BACKGROUND = np.array([0.012, 0.030, 0.060], dtype=np.float32)


def camera_basis(azim_deg: float, elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    return d, right, up


def trilinear(field: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """Sample a (n, n, n) periodic field at pos[..., 3] in [-1/2, 1/2).

    Works in the precision of the inputs: pass float64 fields for physics
    (tracer advection) and float32 snapshots for pure visualisation.  No
    silent downcast is applied here.
    """
    g = (pos + 0.5) * n - 0.5
    i0 = np.floor(g).astype(np.int64)
    fr = g - i0
    i0 %= n
    i1 = (i0 + 1) % n
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fx, fy, fz = fr[..., 0], fr[..., 1], fr[..., 2]
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    c00 = c000 + (c100 - c000) * fx
    c01 = c001 + (c101 - c001) * fx
    c10 = c010 + (c110 - c010) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


def sample_vector(field: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(pos, dtype=np.float64)
    for axis in range(3):
        out[..., axis] = trilinear(field[axis], pos, n)
    return out


def upsample2(field: np.ndarray) -> np.ndarray:
    """Spectral 2x upsampling of a cell-centred periodic field.

    The samples live at x_j = (j + 1/2)/n - 1/2, so the coarse coefficients
    are first rotated by the shared cell-centred origin phase for an
    n -> 2n refinement (`resample_origin_phase`, the same helper `restrict_hat`
    uses): copying coefficients without it translates the volume by 1/(4n)
    per axis relative to the physical coordinates - renderer-only, but it
    visibly separates the bright core from the tracers once the core is small.

    The coarse Nyquist planes (|m| = n/2) are removed first: a general scalar
    like |u| is not band-limited, and the Nyquist coefficient of an even-length
    real DFT is not unambiguously interpolable.  What remains is the
    band-limited trigonometric interpolant of the resolved modes, sampled on
    the 2n grid.
    """
    n = field.shape[0]
    n2 = 2 * n
    half = n // 2
    spec = np.fft.fftn(field)
    spec[half, :, :] = 0.0
    spec[:, half, :] = 0.0
    spec[:, :, half] = 0.0
    spec = spec * resample_origin_phase(n, n2)
    padded = np.zeros((n2, n2, n2), dtype=np.complex128)
    for bx, sx in ((slice(0, half), slice(0, half)), (slice(n2 - half, n2), slice(n - half, n))):
        for by, sy in ((slice(0, half), slice(0, half)), (slice(n2 - half, n2), slice(n - half, n))):
            for bz, sz in ((slice(0, half), slice(0, half)), (slice(n2 - half, n2), slice(n - half, n))):
                padded[bx, by, bz] = spec[sx, sy, sz]
    return np.real(np.fft.ifftn(padded)) * 8.0


_RAY_CACHE: dict = {}


def ray_geometry(image: int, samples: int, zoom: float, azim_deg: float, elev_deg: float):
    """Orthographic ray setup for the unit cube; cached per camera angle."""
    key = (image, samples, round(zoom, 4), elev_deg, round(azim_deg, 4))
    cached = _RAY_CACHE.get(key)
    if cached is not None:
        return cached

    d, right, up = camera_basis(azim_deg, elev_deg)
    us = ((np.arange(image) + 0.5) / image * 2.0 - 1.0) * zoom
    vs = (1.0 - (np.arange(image) + 0.5) / image * 2.0) * zoom
    U, V = np.meshgrid(us, vs)

    origins = (U[..., None] * right + V[..., None] * up - 4.0 * d).astype(np.float32)
    t_lo = np.full((image, image), -np.inf, dtype=np.float32)
    t_hi = np.full((image, image), np.inf, dtype=np.float32)
    for axis in range(3):
        comp = origins[..., axis]
        inv = 1.0 / d[axis]
        a = (-0.5 - comp) * inv
        b = (0.5 - comp) * inv
        t_lo = np.maximum(t_lo, np.minimum(a, b))
        t_hi = np.minimum(t_hi, np.maximum(a, b))
    depth = np.where(t_hi > t_lo, t_hi - t_lo, 0.0).astype(np.float32)
    geometry = (d, origins, t_lo, t_hi, depth)
    if len(_RAY_CACHE) > 64:            # bounded: the display zoom varies per frame
        _RAY_CACHE.pop(next(iter(_RAY_CACHE)))
    _RAY_CACHE[key] = geometry
    return geometry


def render_volume(speed: np.ndarray, vmax: float, image: int, samples: int, zoom: float,
                  azim_deg: float, elev_deg: float, exposure: float = 3.0,
                  opacity: float = 5.0, density_power: float = 1.5) -> np.ndarray:
    """Ray-march a periodic speed field; returns an (image, image, 3) RGB image."""
    d, origins, t_lo, t_hi, depth = ray_geometry(image, samples, zoom, azim_deg, elev_deg)
    K = samples
    scaled = np.clip(speed * (1.0 / max(vmax, 1e-9)), 0.0, 1.0)
    fine = upsample2(scaled.astype(np.float64))
    n2 = fine.shape[0]

    out = np.empty((image, image, 3), dtype=np.float32)
    block = max(8, int(3.5e6 / (image * K)))
    steps = (np.arange(K, dtype=np.float32) + 0.5) / K
    for row0 in range(0, image, block):
        row1 = min(image, row0 + block)
        s = t_lo[row0:row1, :, None] + depth[row0:row1, :, None] * steps
        pos = origins[row0:row1, :, None, :] + s[..., None] * d
        idx = np.floor((pos + 0.5) * n2).astype(np.int32)
        np.clip(idx, 0, n2 - 1, out=idx)
        val = fine[idx[..., 0], idx[..., 1], idx[..., 2]]
        np.clip(val, 0.0, 1.0, out=val)

        sigma = opacity * val ** density_power
        step = sigma * (depth[row0:row1, :, None] / K)
        trans = np.exp(-np.cumsum(step, axis=2))
        pre = trans / np.maximum(np.exp(-step), 1e-30)
        alpha = 1.0 - np.exp(-step)
        color = LUT[np.clip((val * 255.0).astype(np.int32), 0, 255)]

        radiance = np.sum((pre * alpha)[..., None] * color, axis=2)
        rgb = 1.0 - np.exp(-exposure * radiance)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
        rgb = np.maximum(rgb, BACKGROUND * (1.0 - trans[..., -1:]).astype(np.float32) * 0.35)
        out[row0:row1] = rgb.astype(np.float32)
    return out
