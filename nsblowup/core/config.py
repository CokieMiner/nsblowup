"""Configuration dataclasses for the reference construction and DNS solver.

Supports two forcing grid modes:
* `force_grid = None` (grid-relative): The construction is evaluated on an
  n x reference_factor grid, restricted to the DNS grid, and the nonlinearity
  is formed on the DNS grid: f_N = P_N[d_t P_N u + N_N(P_N u) - nu Lap P_N u].
* `force_grid = N*` (absolute): The force is formed entirely on the reference
  grid N* and restricted afterwards: f_N = P_N f_N*. This provides a single,
  resolution-independent forcing for multi-grid studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

H_LIMIT = 0.01        # h must satisfy 0 < h < 1/100


@dataclass(frozen=True)
class ReferenceConfig:
    # similarity / profile parameters
    h: float = 0.005              # must satisfy 0 < h < 1/100
    viscosity: float = 0.02
    q_floor: float = 0.10         # smallest axial tau reached: forced runs
                                  # stop at t_end = 1 - q_floor (no smoothing)
    amplitude: float = 1.0        # overall velocity scale
    similarity_scale: float = 0.30  # L: L-coordinate scale of (X, eta)
    core_radius: float = 0.25     # Gaussian core scale at q = 1 (sigma); collapses like sqrt(q)
    j0: float = 0.25              # upward bias: U(eta) = 4 eta + j0
    # oscillatory annulus (two families), X = (r/L)^2/(2q)
    ring_x_center: float = 0.6    # annulus center in the similarity variable X
    ring_x_width: float = 0.45
    ring_k1: float = 10.0
    ring_k2: float = 14.0
    ring_phase1: float = 0.0
    ring_phase2: float = 0.0
    # four coefficients of the wave basis (family A in-phase/quadrature and
    # family B in-phase/quadrature).  Fitted with --calibrate; the objective is
    # the raw finite-tau annulus residual, not the asymptotic cancellation
    # (see nsblowup.reference.quadratic).
    wave_coeffs: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # compact C-infinity windows; note the band-limited discrete field is
    # globally supported (Fourier truncation spreads tails over the box)
    support_radius: float = 0.42
    support_z: float = 0.42
    window_inner: float = 0.28
    smooth_start: float = 0.06    # C-infinity ramp: u = 0 for t <= 0
    # derivative settings (scale aware, convergence checked)
    deriv_rel_step: float = 2.0e-3
    deriv_min_step: float = 1.0e-5
    # numerics
    n: int = 48
    dealias: str = "strict23"
    reference_factor: int = 2
    force_grid: int | None = None
    allow_large: bool = False

    def validate(self) -> None:
        finite = (self.h, self.viscosity, self.q_floor, self.amplitude,
                  self.similarity_scale, self.core_radius, self.j0,
                  self.ring_x_center, self.ring_x_width, self.ring_k1,
                  self.ring_k2, self.ring_phase1, self.ring_phase2,
                  self.support_radius, self.support_z, self.window_inner,
                  self.smooth_start, self.deriv_rel_step, self.deriv_min_step)
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("ReferenceConfig floats must all be finite")
        if self.n % 2 or self.n < 8:
            raise ValueError("n must be an even integer >= 8")
        if self.force_grid is not None:
            if self.force_grid % 2 or self.force_grid < 8:
                raise ValueError("force_grid must be an even integer >= 8")
            if self.force_grid < self.n:
                raise ValueError(
                    f"force_grid ({self.force_grid}) must be >= the DNS grid n "
                    f"({self.n}): the restriction is fine-to-coarse only")
        if not (0.0 < self.h < H_LIMIT):
            raise ValueError(f"h must satisfy 0 < h < 1/100, got {self.h}")
        if self.viscosity < 0.0:
            raise ValueError(f"viscosity must be non-negative, got {self.viscosity}")
        if not 0.0 < self.q_floor < 1.0:
            raise ValueError(f"q_floor must lie in (0, 1), got {self.q_floor}")
        if self.similarity_scale <= 0.0:
            raise ValueError("similarity_scale (L) must be positive")
        if self.core_radius <= 0.0:
            raise ValueError("core_radius must be positive")
        if self.ring_x_width <= 0.0:
            raise ValueError("ring_x_width must be positive (it divides the "
                             "wave-packet envelope)")
        if self.support_radius <= 0.0 or self.support_z <= 0.0:
            raise ValueError("support radii must be positive")
        if not 0.0 < self.window_inner < min(self.support_radius, self.support_z):
            raise ValueError("window_inner must lie in (0, min(support_radius, "
                             "support_z))")
        if self.smooth_start <= 0.0:
            raise ValueError("smooth_start must be positive (it divides the ramp)")
        if self.reference_factor < 1:
            raise ValueError("reference_factor must be >= 1")
        if self.dealias not in ("strict23", "pad32"):
            raise ValueError(f"unknown dealias mode {self.dealias!r}")
        if self.deriv_rel_step <= 0.0 or self.deriv_min_step <= 0.0:
            raise ValueError("derivative step parameters must be positive")
        if not all(math.isfinite(value) for value in self.wave_coeffs):
            raise ValueError("wave_coeffs must be finite")

    def reference_grid_size(self) -> int:
        """Side of the grid the construction (and, with force_grid, the force) uses.

        With `force_grid` set this is an ABSOLUTE size, independent of the DNS
        grid, which is what makes the forcing one fixed object.
        """
        if self.force_grid:
            n_ref = int(self.force_grid)
        else:
            n_ref = self.n * max(int(self.reference_factor), 1)
        return n_ref + (n_ref % 2)

    def memory_estimate_gb(self, copies: int = 8) -> float:
        """Peak footprint of the construction machinery, in GB.

        The dominant term is the reference grid: a 192^3 construction holds
        several complex128 (3, n_ref, n_ref, n_ref) arrays (~340 MB each) at
        once.
        """
        n_ref = self.reference_grid_size()
        per_field = 3.0 * n_ref ** 3 * 16.0         # complex128 vector field
        polar = 512.0 * 512.0 * 8.0 * 8.0           # dual-number polar evaluation
        return (copies * per_field + polar) / 1e9

    def assert_affordable(self, cap_gb: float = 1.6) -> None:
        """Refuse configurations above `cap_gb` unless `allow_large` is set.

        Oversized requests raise SystemExit with an explicit opt-in instead of
        pushing the machine into swap.
        """
        estimate = self.memory_estimate_gb()
        if estimate > cap_gb and not self.allow_large:
            mode = (f"force_grid={self.force_grid}" if self.force_grid
                    else f"n={self.n} x reference_factor={self.reference_factor}")
            raise SystemExit(
                f"reference grid {self.reference_grid_size()}^3 ({mode}) needs "
                f"~{estimate:.1f} GB, above the {cap_gb:.1f} GB safety cap.\n"
                "Options: use n<=48 (or --reference-factor 1), lower --force-grid, "
                "or pass --allow-large if the machine has the memory.")

    @property
    def t_end(self) -> float:
        return 1.0 - self.q_floor


@dataclass(frozen=True)
class DNSConfig:
    """Solver and movie settings.  `n` is the DNS grid; the forcing grid is a
    property of the reference construction (`ReferenceConfig.force_grid`)."""
    n: int = 64
    viscosity: float = 0.02
    t_end: float = 0.90
    free_evolution: bool = False    # no forcing; released from u*(release_time)
    release_time: float = 0.0
    frames: int = 144        # movie length is frames / fps (see fps, hold_frames)
    fps: int = 24            # playback rate: duration = (frames + hold_frames) / fps
    hold_frames: int = 24    # 1 s freeze on the final state
    cfl: float = 0.70
    max_dt: float = 0.004
    dealias: str = "strict23"
    tail_tolerance: float = 2.0e-3
    tail_arm: float = 0.0           # t after which the tail stop is enforced
    store_dtype: str = "float32"    # snapshot storage only; never used for physics
    store_swirl: bool = False       # also store u_theta snapshots (profile collapse)
    follow_collapse: bool = False   # optional display zoom ~ sqrt(tau)
    # tracers: uniform random in the periodic box
    particles: int = 700
    trail: int = 9
    seed: int = 7
    # rendering
    image: int = 480
    samples: int = 84
    zoom: float = 0.46
    azim: float = 35.0
    elev: float = 24.0
    exposure: float = 3.0
    opacity: float = 5.0
    density_power: float = 1.5
    output: str = "openai_dns.mp4"


QUICK = dict(n=32, t_end=0.80, frames=48, max_dt=0.010, image=300, samples=64,
             particles=400, trail=5)
