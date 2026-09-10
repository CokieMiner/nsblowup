"""Manufactured-solution check of the solver and the forcing pipeline.

A smooth manufactured field u* (Gaussian envelopes) is band-limited with the
same 2/3 mask the solver applies; its exact discrete residual is used as the
forcing, and the solver is integrated from rest with it.  Because u* is known,
the integration error is directly measurable, so this checks the solver, the
time stepping and the residual pipeline without depending on the paper's
construction: the tracking error measures the integrator, not a mismatch of
discrete spaces.

Run:  python cli.py verify [--n 32] [--t-end 0.9] [--frames 12] [--dealias pad32]

The tracking window is t >= 5 * smooth_start = 0.5 and the check needs at
least 4 frames inside it; with the defaults (t_end 0.9, 12 frames) there are
6, and the check fails explicitly if a run has too few.
"""

from __future__ import annotations

import argparse
import time as _time
from dataclasses import dataclass, replace

import numpy as np

from nsblowup.core.config import DNSConfig, ReferenceConfig
from nsblowup.core.dual import Dual, smooth_step
from nsblowup.core.spectral import (
    curl_hat,
    physical,
    project,
    restrict_hat,
    retained_mask,
    rotational_product,
    wavenumbers,
)
from nsblowup.run.solver import simulate
from nsblowup.run.volume import upsample2


@dataclass(frozen=True)
class VerifyConfig:
    """Manufactured case and check tolerances."""
    n: int = 32
    viscosity: float = 0.02
    tau_min: float = 0.10          # collapse clock at the end of the run
    smooth_start: float = 0.10     # ramp: u*(., 0) = 0
    core_radius: float = 0.22
    axial_radius: float = 0.16
    swirl_scale: float = 0.35
    frames: int = 12
    max_dt: float = 4.0e-3
    cfl: float = 0.7
    dealias: str = "strict23"
    tail_tolerance: float = 5.0e-3
    tracking_tolerance: float = 0.05   # 5 % on the post-ramp frames

    @property
    def t_end(self) -> float:
        return 1.0 - self.tau_min


@dataclass(frozen=True)
class _Grid:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    rr: np.ndarray


_GRID_CACHE: dict[int, _Grid] = {}


def _grid(cfg: VerifyConfig) -> _Grid:
    cached = _GRID_CACHE.get(cfg.n)
    if cached is None:
        x = (np.arange(cfg.n) + 0.5) / cfg.n - 0.5
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        cached = _Grid(X, Y, Z, np.sqrt(X * X + Y * Y))
        _GRID_CACHE.clear()
        _GRID_CACHE[cfg.n] = cached
    return cached


# -- manufactured construction: A = (-y P, x P, c), u = curl A -------------

def _profiles(cfg: VerifyConfig, t: float, grid) -> tuple[Dual, Dual]:
    """Gaussian poloidal (P) and swirl (c) potentials, with their time slopes."""
    ramp = smooth_step(Dual(t / cfg.smooth_start, 1.0 / cfg.smooth_start))
    rho = Dual(-(grid.rr / cfg.core_radius) ** 2, 0.0).exp()
    zeta = Dual(-(grid.Z / cfg.axial_radius) ** 2, 0.0).exp()
    growth = (Dual(1.0 - t, -1.0)) ** (-0.5)            # mild collapse growth
    poloidal = ramp * growth * rho * zeta
    swirl = cfg.swirl_scale * ramp * growth * rho * zeta
    return poloidal, swirl


def constructed_u_hat(cfg: VerifyConfig, t: float) -> np.ndarray:
    """u*(., t): divergence-free and band-limited by the solver's 2/3 mask."""
    k = wavenumbers(cfg.n)
    grid = _grid(cfg)
    poloidal, swirl = _profiles(cfg, t, grid)
    potential = np.stack((poloidal.v * (-grid.Y), poloidal.v * grid.X, swirl.v))
    return retained_mask(cfg.n) * curl_hat(np.fft.fftn(potential, axes=(1, 2, 3)), k)


def force_hat(cfg: VerifyConfig, t: float) -> np.ndarray:
    """Exact discrete residual of the band-limited u*, used as the forcing."""
    k = wavenumbers(cfg.n)
    _, _, _, k2, _ = k
    mask = retained_mask(cfg.n)
    grid = _grid(cfg)
    poloidal, swirl = _profiles(cfg, t, grid)
    potential_hat = np.fft.fftn(np.stack((poloidal.v * (-grid.Y), poloidal.v * grid.X,
                                          swirl.v)), axes=(1, 2, 3))
    slope_hat = np.fft.fftn(np.stack((poloidal.d * (-grid.Y), poloidal.d * grid.X,
                                      swirl.d)), axes=(1, 2, 3))
    # the solver masks its state every step, so the manufactured solution must
    # live in that same discrete space
    u_hat = mask * curl_hat(potential_hat, k)
    du_hat = mask * curl_hat(slope_hat, k)
    adv_hat, _ = rotational_product(u_hat, k, cfg.dealias)
    return project(du_hat + adv_hat + cfg.viscosity * k2 * u_hat, k)


# -- run and check ---------------------------------------------------------

def run(cfg: VerifyConfig) -> dict:
    """Integrate from rest with the manufactured forcing and report errors."""
    dns = DNSConfig(n=cfg.n, viscosity=cfg.viscosity, t_end=cfg.t_end,
                    frames=cfg.frames, max_dt=cfg.max_dt, cfl=cfg.cfl,
                    dealias=cfg.dealias,
                    tail_tolerance=cfg.tail_tolerance, particles=0,
                    image=240, samples=32)
    ref = ReferenceConfig(n=cfg.n, viscosity=cfg.viscosity, q_floor=cfg.tau_min)
    start = _time.perf_counter()
    res = simulate(dns, ref,
                   force_fn=lambda t: force_hat(cfg, t),
                   field_fn=lambda t: physical(constructed_u_hat(cfg, t)))
    elapsed = _time.perf_counter() - start

    taus = np.clip(1.0 - res.times, cfg.tau_min, None)
    window = res.times >= 5.0 * cfg.smooth_start
    valid = window & np.isfinite(res.tracking)
    tracking = res.tracking[valid]
    if np.count_nonzero(valid) >= 4:
        speed_fit = float(np.polyfit(np.log(taus[valid]),
                                     np.log(res.max_speed[valid]), 1)[0])
    else:
        speed_fit = float("nan")
    return {
        "result": res,
        "elapsed": elapsed,
        "fit_points": int(np.count_nonzero(valid)),
        "max_tracking": float(np.nanmax(tracking)) if tracking.size else float("nan"),
        "final_div": res.div_error,
        "energy_start": float(res.energy[0]),
        "energy_peak": float(np.nanmax(res.energy)),
        "speed_slope": speed_fit,
        "steps": res.steps,
        "restriction_error": max(restriction_registration_error(cfg.n, 2),
                                 restriction_registration_error(cfg.n, 3)),
        "restriction_analytic_error": max(restriction_analytic_error(cfg.n, 2),
                                          restriction_analytic_error(cfg.n, 3)),
        "upsample_error": upsample_registration_error(cfg.n),
        "leray_error": projection_check(cfg.n),
    }


def restriction_registration_error(n: int, factor: int = 3) -> float:
    """Max relative error of the fine-to-coarse spectral restriction.

    Builds a random coarse-band field, evaluates the SAME physical function on
    a finer cell-centred grid (x_j = (j+1/2)/N - 1/2), restricts it back and
    compares coefficients.  Without the origin-phase correction in
    `restrict_hat` the two grids' cell centres differ by
    delta = 1/(2 n_f) - 1/(2 n_c) and the error is O(k delta) ~ 1e-1; with it
    the comparison is at machine precision.
    """
    n_c = n
    n_f = factor * n
    cut = (n_c - 1) // 3
    rng = np.random.default_rng(2026)
    coarse = rng.standard_normal((3, n_c, n_c, n_c))
    _, _, _, _, (mx, my, mz) = wavenumbers(n_c)
    band = (np.abs(mx) <= cut) & (np.abs(my) <= cut) & (np.abs(mz) <= cut)
    hat_c = np.fft.fftn(coarse, axes=(1, 2, 3)) * band
    # centred layouts: index n/2 is frequency zero for even n
    centred = np.fft.fftshift(hat_c, axes=(1, 2, 3))
    pad = np.zeros((3, n_f, n_f, n_f), dtype=complex)
    sl_c = slice(n_c // 2 - cut, n_c // 2 + cut + 1)
    sl_f = slice(n_f // 2 - cut, n_f // 2 + cut + 1)
    # embedding coarse numpy coefficients into the fine array carries the
    # (n_f/n_c)^3 normalisation of the DFT (same factor as `_embed`)
    pad[:, sl_f, sl_f, sl_f] = centred[:, sl_c, sl_c, sl_c] * (n_f / n_c) ** 3
    freqs = np.fft.fftshift(np.fft.fftfreq(n_f, d=1.0 / n_f))
    MX, MY, MZ = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    origin = 1.0 / (2.0 * n_f) - 1.0 / (2.0 * n_c)      # fine sample offset
    pad *= np.exp(2j * np.pi * origin * (MX + MY + MZ))  # fine samples of f
    hat_f = np.fft.ifftshift(pad, axes=(1, 2, 3))
    restricted = restrict_hat(hat_f, n_c, n_f)
    # coefficient comparison: `_truncate` rescales by (n_c/n_f)^3 so that both
    # arrays are the numpy coefficients of the same physical field
    scale = max(float(np.max(np.abs(hat_c))), 1e-30)
    return float(np.max(np.abs(restricted - hat_c)) / scale)


def restriction_analytic_error(n: int, factor: int = 3) -> float:
    """Independent check of `restrict_hat` with analytic cos modes.

    A low-mode trigonometric polynomial is sampled DIRECTLY on the fine and the
    coarse physical grids, both are FFT'd, and the restricted fine coefficients
    are compared with the coarse ones.  Nothing in the expected result knows
    the origin-phase formula `restrict_hat` applies, so this is not circular
    with `restriction_registration_error`.
    """
    n_c = n
    n_f = factor * n
    cut = (n_c - 1) // 3
    rng = np.random.default_rng(7)
    modes = []
    while len(modes) < 3:
        m = rng.integers(-cut, cut + 1, size=3)
        if np.any(m):
            modes.append(m)
    xc = (np.arange(n_c) + 0.5) / n_c - 0.5
    xf = (np.arange(n_f) + 0.5) / n_f - 0.5
    XC, YC, ZC = np.meshgrid(xc, xc, xc, indexing="ij")
    XF, YF, ZF = np.meshgrid(xf, xf, xf, indexing="ij")
    coarse = np.zeros((3, n_c, n_c, n_c))
    fine = np.zeros((3, n_f, n_f, n_f))
    for k, m in enumerate(modes):
        amp = float(10.0 ** rng.uniform(-1.0, 0.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        coarse[k] = amp * np.cos(2 * np.pi * (m[0] * XC + m[1] * YC + m[2] * ZC)
                                 + phase)
        fine[k] = amp * np.cos(2 * np.pi * (m[0] * XF + m[1] * YF + m[2] * ZF)
                               + phase)
    hat_c = np.fft.fftn(coarse, axes=(1, 2, 3))
    restricted = restrict_hat(np.fft.fftn(fine, axes=(1, 2, 3)), n_c, n_f)
    scale = max(float(np.max(np.abs(hat_c))), 1e-30)
    return float(np.max(np.abs(restricted - hat_c)) / scale)


def projection_check(n: int) -> float:
    """Leray projector: divergence-free output and idempotence, both relative."""
    rng = np.random.default_rng(11)
    field = rng.standard_normal((3, n, n, n))
    k = wavenumbers(n)
    kx, ky, kz, _, _ = k
    projected = project(np.fft.fftn(field, axes=(1, 2, 3)), k)
    div = kx * projected[0] + ky * projected[1] + kz * projected[2]
    scale = max(float(np.max(np.abs(projected))), 1e-30)
    kmax = max(float(np.max(np.abs(kx) + np.abs(ky) + np.abs(kz))), 1e-30)
    divergence = float(np.max(np.abs(div)) / (scale * kmax))
    idempotence = float(np.max(np.abs(project(projected, k) - projected)) / scale)
    return max(divergence, idempotence)


def upsample_registration_error(n: int) -> float:
    """Cell-centred registration of the renderer's 2x upsampling (`upsample2`).

    A band-limited trigonometric polynomial (modes well inside the retained
    band, so no Nyquist content) is sampled DIRECTLY on the coarse and on the
    doubled cell-centred grids; `upsample2` of the coarse samples is compared
    with the analytic fine samples.  Without the origin phase from
    `resample_origin_phase` the upsampled volume is translated by 1/(4n) per
    axis and the error is O(1); with it the comparison is at machine
    precision.
    """
    n2 = 2 * n
    cut = (n - 1) // 3
    rng = np.random.default_rng(99)
    modes = []
    while len(modes) < 3:
        m = rng.integers(-cut, cut + 1, size=3)
        if np.any(m):
            modes.append(m)
    xc = (np.arange(n) + 0.5) / n - 0.5
    xf = (np.arange(n2) + 0.5) / n2 - 0.5
    XC, YC, ZC = np.meshgrid(xc, xc, xc, indexing="ij")
    XF, YF, ZF = np.meshgrid(xf, xf, xf, indexing="ij")
    coarse = np.zeros((n, n, n))
    fine = np.zeros((n2, n2, n2))
    for m in modes:
        amp = float(10.0 ** rng.uniform(-1.0, 0.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        coarse += amp * np.cos(2 * np.pi * (m[0] * XC + m[1] * YC + m[2] * ZC)
                               + phase)
        fine += amp * np.cos(2 * np.pi * (m[0] * XF + m[1] * YF + m[2] * ZF)
                             + phase)
    upsampled = upsample2(coarse)
    scale = max(float(np.max(np.abs(fine))), 1e-30)
    return float(np.max(np.abs(upsampled - fine)) / scale)


def report(cfg: VerifyConfig, summary: dict) -> bool:
    """Print the checks; returns True when they pass."""
    passed = True
    print(f"  grid {cfg.n}^3 · dealias={cfg.dealias} · steps {summary['steps']} "
          f"· {summary['elapsed']:.1f} s")
    print(f"  divergence (final)      {summary['final_div']:.2e}")
    print(f"  tracking error (max)    {summary['max_tracking'] * 100:.2f} %"
          f"   (tolerance {cfg.tracking_tolerance * 100:.0f} %)")
    print(f"  kinetic energy          start {summary['energy_start']:.2e} · "
          f"peak {summary['energy_peak']:.4f}")
    print(f"  fitted max|u| slope     {summary['speed_slope']:.3f}  "
          f"(manufactured growth: expected -0.500)")
    print(f"  restriction registration {summary['restriction_error']:.2e}  "
          f"(fine->coarse origin phase, threshold 1e-10)")
    print(f"  restriction (analytic)  {summary['restriction_analytic_error']:.2e}  "
          f"(cos modes sampled on both grids, threshold 1e-10)")
    print(f"  upsample registration   {summary['upsample_error']:.2e}  "
          f"(2x renderer resampling, cell-centred phase, threshold 1e-10)")
    print(f"  Leray projection        {summary['leray_error']:.2e}  "
          f"(div P u and idempotence, threshold 1e-12)")
    if summary["restriction_error"] > 1e-10:
        print("  FAIL: fine->coarse restriction is not origin-registered")
        passed = False
    if summary["restriction_analytic_error"] > 1e-10:
        print("  FAIL: analytic restriction check failed")
        passed = False
    if summary["upsample_error"] > 1e-10:
        print("  FAIL: 2x renderer upsampling is not origin-registered")
        passed = False
    if summary["leray_error"] > 1e-12:
        print("  FAIL: Leray projector is not clean")
        passed = False
    if summary["fit_points"] < 4:
        print("  FAIL: fewer than 4 post-ramp frames (raise --t-end)")
        passed = False
    elif not np.isfinite(summary["max_tracking"]):
        print("  FAIL: tracking is not finite on the post-ramp frames")
        passed = False
    elif summary["max_tracking"] > cfg.tracking_tolerance:
        print("  FAIL: solver does not follow the manufactured solution")
        passed = False
    if not np.isfinite(summary["final_div"]) or summary["final_div"] > 1e-12:
        print("  FAIL: divergence not at machine precision")
        passed = False
    return passed


def parse_args(argv: list[str] | None = None) -> VerifyConfig:
    defaults = VerifyConfig()
    parser = argparse.ArgumentParser(prog="nsblowup verify",
                                     description="manufactured-solution check")
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--frames", type=int, default=defaults.frames)
    parser.add_argument("--viscosity", type=float, default=defaults.viscosity)
    parser.add_argument("--dealias", choices=("strict23", "pad32"),
                        default=defaults.dealias)
    args = parser.parse_args(argv)
    cfg = replace(defaults, n=args.n, frames=args.frames,
                  viscosity=args.viscosity, dealias=args.dealias)
    if args.t_end is not None:
        cfg = replace(cfg, tau_min=1.0 - args.t_end)
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    print("Manufactured-solution check: u(x,0)=0 forced by the residual of u*")
    summary = run(cfg)
    ok = report(cfg, summary)
    print("  PASS" if ok else "  FAILED")
    return 0 if ok else 1
