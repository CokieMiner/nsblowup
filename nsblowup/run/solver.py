"""Solver for the forced Navier-Stokes problem:

    d_t u + (u . grad)u - nu Lap u + grad p = f,    div u = 0,   u(x, 0) = 0

on the unit periodic box: Fourier pseudo-spectral, IMEX-RK2 with exact
integrating factor for the viscous term and trapezoidal forcing average.
Nonlinearity is integrated in rotational form (curl u) x u.

The forcing enters as a callable force_fn(t) (tabulated or analytical);
the solver time loop maintains no coupling to the reference field.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np

from nsblowup.core.config import DNSConfig, ReferenceConfig
from nsblowup.core.spectral import (
    physical,
    project,
    retained_mask,
    rotational_product,
    spectral_tail_fraction,
    wavenumbers,
)
from nsblowup.run.volume import sample_vector


@dataclass
class Result:
    """Everything the solver records, plus the scalars used by the movie."""
    times: np.ndarray
    speeds: np.ndarray              # (frames, n, n, n) snapshot dtype (movie only)
    swirl: np.ndarray | None        # (frames, n, n, n) u_theta, optional
    beyond_resolution: np.ndarray   # bool per frame: after the spectral-tail stop
    analytic_reference: np.ndarray  # bool per frame: field from the construction
    tracers: np.ndarray             # (frames, particles, 3) float32
    max_speed: np.ndarray
    max_swirl: np.ndarray
    max_radial: np.ndarray
    length_r: np.ndarray
    length_z: np.ndarray
    energy: np.ndarray
    tail: np.ndarray
    reynolds_swirl: np.ndarray
    tracking: np.ndarray            # DNS vs construction (diagnostics only)
    stop_index: int
    steps: int
    runtime: float
    div_error: float
    vmax: float
    tail_arm: float                 # monitor armed after this time
    tail_prearm_max: float          # peak tail over ALL internal steps before
                                    # arming (recorded, never hidden: a large
                                    # value means the early transient was NOT
                                    # resolvable either)


def simulate(cfg: DNSConfig, ref: ReferenceConfig, force_fn=None, field_fn=None,
             initial_hat=None) -> Result:
    """Integrate the forced Navier-Stokes system.

    force_fn(t) -> f_hat(t)     the only forcing the solver ever sees
    field_fn(t) -> u_ref(x, t)  optional, diagnostics only; None disables the
                                tracking measurement and the reference frames
    initial_hat                 optional (3, n, n, n) initial state, masked and
                                projected; None starts from rest.  The
                                free-evolution experiment releases the solver
                                from the constructed field u*(t0) this way.
    """
    if force_fn is None:
        raise SystemExit("simulate() needs a forcing callable (force_fn)")
    n = cfg.n
    k = wavenumbers(n)
    kx, ky, kz, k2, _ = k
    mask = retained_mask(n, cfg.dealias)
    dx = 1.0 / n

    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rr = np.sqrt(X * X + Y * Y)
    inv_r = np.where(rr > 1e-12, 1.0 / rr, 0.0)

    u_hat = (mask * project(initial_hat.astype(np.complex128), k)
             if initial_hat is not None
             else np.zeros((3, n, n, n), dtype=np.complex128))   # u(x, 0) = 0
    # released runs open at the release time so the clock stays absolute
    times = (np.linspace(cfg.release_time, cfg.t_end, cfg.frames)
             if initial_hat is not None
             else np.linspace(0.0, cfg.t_end, cfg.frames))
    # snapshot storage only: solver, force and measurements stay float64/128
    speeds = np.empty((cfg.frames, n, n, n), dtype=np.dtype(cfg.store_dtype))
    swirl = (np.empty((cfg.frames, n, n, n), dtype=np.dtype(cfg.store_dtype))
             if cfg.store_swirl else None)
    scalars = {name: np.full(cfg.frames, np.nan) for name in
               ("max_speed", "max_swirl", "max_radial", "length_r", "length_z",
                "energy", "tail", "reynolds_swirl", "tracking")}

    # passive tracers: uniform random in the periodic box, so the collapse and
    # its support boundary are sampled everywhere rather than in one sheet
    rng = np.random.default_rng(cfg.seed)
    particles = int(cfg.particles)
    tracers = rng.random((particles, 3)) - 0.5
    tracer_history = np.empty((cfg.frames, particles, 3), dtype=np.float32)

    force_cache: dict[float, np.ndarray] = {}

    def force_at(tq: float) -> np.ndarray:
        key = round(tq, 12)
        cached = force_cache.get(key)
        if cached is None:
            cached = force_fn(tq)
            force_cache.clear()
            force_cache[key] = cached
        return cached

    t = float(times[0])
    steps = 0
    stop_index = cfg.frames
    tail_prearm_max = 0.0
    start = _time.perf_counter()
    next_report = start + 3.0

    for frame, target in enumerate(times):
        while t < target - 1e-12:
            u_phys = physical(u_hat)
            umax = float(np.sqrt(np.max(np.sum(u_phys * u_phys, axis=0))))
            dt = min(cfg.max_dt, cfg.cfl * dx / (np.sqrt(3.0) * max(umax, 1e-9)),
                     target - t)

            adv1, _ = rotational_product(u_hat, k, cfg.dealias)
            f_now = force_at(t)
            g1 = -project(adv1, k) + f_now

            m12 = np.exp(-0.5 * cfg.viscosity * k2 * dt)
            m1 = np.exp(-cfg.viscosity * k2 * dt)
            u_mid = mask * project(m12 * (u_hat + 0.5 * dt * g1), k)

            u_mid_phys = physical(u_mid)
            adv2, _ = rotational_product(u_mid, k, cfg.dealias)
            f_next = force_at(t + dt)
            g2 = -project(adv2, k) + 0.5 * (f_now + f_next)
            u_hat = mask * project(m1 * u_hat + dt * m12 * g2, k)

            v1 = sample_vector(u_phys, tracers, n)
            mid_pos = (tracers + 0.5 * dt * v1 + 0.5) % 1.0 - 0.5
            v2 = sample_vector(u_mid_phys, mid_pos, n)
            tracers = (tracers + dt * v2 + 0.5) % 1.0 - 0.5

            t = min(t + dt, target)
            steps += 1
            # true pre-arm peak: EVERY internal step counts, not only the
            # stored snapshot frames (the ramp transient lives between them)
            if t < cfg.tail_arm - 1e-12:
                tail_prearm_max = max(tail_prearm_max,
                                      spectral_tail_fraction(u_hat, n))
            now = _time.perf_counter()
            if now > next_report:
                print(f"  step {steps:5d}  t={t:6.4f}  tau={1.0 - t:6.4f}  "
                      f"max|u|={umax:7.4f}  tail={spectral_tail_fraction(u_hat, n):.2e}  "
                      f"{steps / (now - start):5.1f} steps/s")
                next_report = now + 8.0

        # ---- snapshot ----------------------------------------------------
        u_phys = physical(u_hat)
        speed = np.sqrt(np.sum(u_phys * u_phys, axis=0))
        speeds[frame] = speed.astype(np.dtype(cfg.store_dtype))
        tracer_history[frame] = tracers

        ux, uy, uz = u_phys
        u_theta = (-Y * ux + X * uy) * inv_r
        u_rad = (X * ux + Y * uy) * inv_r
        if swirl is not None:
            swirl[frame] = u_theta.astype(np.dtype(cfg.store_dtype))
        # widths use a |u|^4 weighting so the moments follow the core instead of
        # the broad exterior; descriptive only, see the report for profiles
        weight = speed ** 4
        total = float(np.sum(weight)) + 1e-30
        scalars["max_speed"][frame] = float(speed.max())
        scalars["max_swirl"][frame] = float(np.max(np.abs(u_theta)))
        scalars["max_radial"][frame] = float(np.max(np.abs(u_rad)))
        scalars["length_r"][frame] = float(np.sqrt(np.sum(weight * rr * rr) / total))
        scalars["length_z"][frame] = float(np.sqrt(np.sum(weight * Z * Z) / total))
        scalars["energy"][frame] = 0.5 * float(np.mean(speed * speed))
        tail = spectral_tail_fraction(u_hat, n)
        scalars["tail"][frame] = tail
        scalars["reynolds_swirl"][frame] = (
            scalars["max_swirl"][frame] * scalars["length_r"][frame] / cfg.viscosity)

        # diagnostics only, after the timestep: how close did the blind
        # integration come to the construction?  During the start-up ramp
        # u_ref ~ 0 and a relative error is meaningless, so those frames stay NaN.
        if field_fn is not None and t >= 5.0 * ref.smooth_start:
            u_ref = field_fn(t)
            num = float(np.sqrt(np.mean((u_phys - u_ref) ** 2)))
            den = max(float(np.sqrt(np.mean(u_ref ** 2))), 1e-12)
            scalars["tracking"][frame] = num / den

        if tail > cfg.tail_tolerance and frame > 0 and t >= cfg.tail_arm - 1e-12:
            stop_index = frame
            print(f"  DNS stopped at t={t:.4f}: spectral tail {tail:.2e} > "
                  f"{cfg.tail_tolerance:.2e} (beyond numerical resolution)")
            break

    runtime = _time.perf_counter() - start

    # frames beyond the resolution limit: stored for the archive, but flagged
    # by role - frozen after the stop, or filled from the construction.  Tracer
    # history is frozen at the last valid state so the renderer cannot read
    # stale memory.
    beyond_resolution = np.zeros(cfg.frames, dtype=bool)
    analytic_reference = np.zeros(cfg.frames, dtype=bool)
    if stop_index < cfg.frames:
        beyond_resolution[stop_index:] = True
        last = max(stop_index - 1, 0)          # stop_index itself is unresolved
        last_valid = tracer_history[last].copy()
        for frame in range(stop_index, cfg.frames):
            tracer_history[frame] = last_valid
        if field_fn is None:
            frozen = speeds[last].copy()
            for frame in range(stop_index, cfg.frames):
                speeds[frame] = frozen
        else:
            analytic_reference[stop_index:] = True
            for frame in range(stop_index, cfg.frames):
                ref_speed = np.sqrt(np.sum(field_fn(times[frame]) ** 2, axis=0))
                speeds[frame] = ref_speed.astype(np.dtype(cfg.store_dtype))
        if swirl is not None:
            swirl[stop_index:] = swirl[last]

    div_hat = 1j * (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2])
    div_l2 = float(np.sqrt(np.mean(np.real(np.fft.ifftn(div_hat)) ** 2)))

    visible = speeds[:max(stop_index, 1)]      # hidden reference frames must not
    return Result(                             # set the colour scale of the movie
        times=times, speeds=speeds, swirl=swirl,
        beyond_resolution=beyond_resolution,
        analytic_reference=analytic_reference,
        tracers=tracer_history, max_speed=scalars["max_speed"],
        max_swirl=scalars["max_swirl"], max_radial=scalars["max_radial"],
        length_r=scalars["length_r"], length_z=scalars["length_z"],
        energy=scalars["energy"], tail=scalars["tail"],
        reynolds_swirl=scalars["reynolds_swirl"], tracking=scalars["tracking"],
        stop_index=stop_index, steps=steps, runtime=runtime,
        div_error=div_l2, vmax=float(visible.max()),
        tail_arm=float(cfg.tail_arm),
        tail_prearm_max=tail_prearm_max,
    )
