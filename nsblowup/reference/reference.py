"""Reference construction and analytical forcing.

The field is built in similarity variables:

    tau = q (1 - eta^2),  z = L q^D eta,  X = (r/L)^2 / (2 q),
    A = 1/2 + h,          D = 1/2 - h,    0 < h < 1/100,

where q solves q = tau + (z/L)^2 q^{2h} (a non-uniform collapse clock).
u = curl A is divergence-free by construction, and the forcing is the
discrete residual of that field:

    f = P[d_t u + (u.grad)u - nu Lap u],

evaluated using forward-mode automatic differentiation for d_t u.

Runs stop at t_end = 1 - q_floor, with q_floor setting the minimum axial tau.
The construction serves as a finite-scale surrogate to study precursor
kinematics and verify numerical integration under the prescribed scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from nsblowup.core.config import ReferenceConfig
from nsblowup.core.dual import Dual, smooth_step
from nsblowup.core.spectral import (
    convective_product,
    curl_hat,
    physical,
    project,
    restrict_hat,
    retained_mask,
    rotational_product,
    spectral_tail_fraction,
    wavenumbers,
)


def residual_rms(cfg: ReferenceConfig, t: float,
                 wave_coeffs: tuple[float, float, float, float]) -> float:
    """Box RMS of the discrete residual for the given wave coefficients."""
    oracle = ForceOracle(cfg, wave_amps=wave_coeffs)
    field = physical(oracle.force_hat(t))
    return float(np.sqrt(np.mean(field ** 2)))


def annulus_weight(cfg: ReferenceConfig, t: float, grid: "SpectralGrid") -> np.ndarray:
    """Smooth weight selecting the annular region in similarity variables.

    Uses the solved clock directly: q_floor is not a floor applied to q, it is
    the smallest axial tau the runs reach (t_end = 1 - q_floor), and solve_q
    keeps q >= tau on that whole domain.
    """
    tau = max(1.0 - t, 0.0)
    scale = cfg.similarity_scale
    q = solve_q(grid.Z / scale, tau, cfg.h)
    X = (grid.rr / scale) ** 2 / (2.0 * q)
    eta = grid.Z / (scale * q ** (0.5 - cfg.h))
    return (np.exp(-((X - cfg.ring_x_center) / (1.5 * cfg.ring_x_width)) ** 2)
            * np.exp(-(eta / 2.5) ** 2))


# --------------------------------------------------------------------------
# smooth building blocks
# --------------------------------------------------------------------------
def _phi(x: np.ndarray) -> np.ndarray:
    """phi(x) = exp(-1/x) for x > 0, 0 otherwise (C-infinity, flat at 0+)."""
    arr = np.asarray(x, dtype=float)
    out = np.zeros_like(arr)
    positive = arr > 0.0
    out[positive] = np.exp(-1.0 / arr[positive])
    return out


def flat01(t: np.ndarray | float) -> np.ndarray:
    """C-infinity step: 0 for t <= 0, 1 for t >= 1, flat to all orders.

    Uses the standard construction S(x) = phi(x) / (phi(x) + phi(1-x)).  The
    bump exp(-1/(t(1-t))) alone returns to 0 at both ends, so a window built
    from it is a near-unit plateau with a cliff at t = 1; the normalised form
    below is a genuine step, smooth at both ends.
    """
    arr = np.asarray(t, dtype=float)
    left = _phi(arr)
    right = _phi(1.0 - arr)
    return left / (left + right + 1e-300)


def flat_window(x: np.ndarray, inner_fraction: float) -> np.ndarray:
    """1 for |x| <= inner, C-infinity decay to exactly 0 at |x| = 1."""
    ax = np.abs(x)
    return 1.0 - flat01((ax - inner_fraction) / (1.0 - inner_fraction))


# --------------------------------------------------------------------------
# similarity variables
# --------------------------------------------------------------------------
def solve_q(z: np.ndarray, tau: float, h: float, iterations: int = 40) -> np.ndarray:
    """Solve q = tau + z^2 q^{2h} (Newton; q is the local collapse clock)."""
    q = tau + z * z
    for _ in range(iterations):
        q2h = q ** (2.0 * h)
        residual = q - z * z * q2h - tau
        slope = 1.0 - 2.0 * h * z * z * q ** (2.0 * h - 1.0)
        q = q - residual / slope
    return np.maximum(q, tau)


def solve_q_dual(z_scaled: np.ndarray, tau: Dual, h: float) -> Dual:
    """Implicit dual solution of q = tau + z^2 q^{2h}.

    The derivative dq/dt is evaluated by implicit differentiation.
    q_floor sets the domain boundary in tau without smoothing q.
    """
    value = solve_q(z_scaled, tau.v, h)
    dq_dtau = 1.0 / (1.0 - 2.0 * h * z_scaled ** 2 * value ** (2.0 * h - 1.0))
    return Dual(value, dq_dtau * tau.d)


def potential_dual(cfg: ReferenceConfig, t: float, grid: "SpectralGrid",
                   wave_amps: tuple[float, float, float, float] | None = None):
    """Vector potential A = (-y P, x P, c) and its exact time derivative.

    u = curl(A) gives, in cylindrical components,

        u_r = -r d_z P,      u_theta = -d_r c,      u_z = 2 P + r d_r P,

    so the sampled field is divergence-free by construction (no Leray
    projection, no deformation of the profiles).

    The potentials carry their own compact envelopes (suppressed with the
    collapsing scales) before the safety window is applied: windowing a
    potential whose value is much larger than its own variation injects
    spurious velocities ~ c|grad W| at the window edge, which would dominate
    the force outside the declared support.
    """
    if wave_amps is None:
        wave_amps = cfg.wave_coeffs
    scale = cfg.similarity_scale
    h = cfg.h
    tau = Dual(1.0 - t, -1.0)                      # d tau/dt = -1
    q = solve_q_dual(grid.Z / scale, tau, h)
    X = Dual((grid.rr / scale) ** 2, 0.0) / (2.0 * q)
    eta = Dual(grid.Z / scale, 0.0) / (q ** (0.5 - h))
    ramp = smooth_step(Dual(t / cfg.smooth_start, 1.0 / cfg.smooth_start))
    q_pow = q ** (-(0.5 + h))

    # compact envelopes: radius ~ sqrt(q) so the core genuinely collapses
    gauss_r = (Dual(-(grid.rr / cfg.core_radius) ** 2, 0.0) / q).exp()
    gauss_z = (Dual(-(grid.Z / cfg.core_radius) ** 2, 0.0)
               / (q ** (1.0 - 2.0 * h))).exp()
    core_r = cfg.core_radius * q ** 0.5

    # swirl potential: u_theta = -d_r c gives
    #   u_theta ~ amplitude q^{-1/2-h} (r/(R0 sqrt(q))) exp(-(r/(R0 sqrt(q)))^2)
    # growing like the paper's q^{-1/2-h}: c carries q^{-h} on its own.
    swirl = (cfg.amplitude * ramp * (cfg.core_radius / 2.0) * q ** (-h)
             * gauss_r * gauss_z * (1.0 + X) ** (-h))

    # poloidal potential: u_z = 2P + r d_r P ~ amplitude q^{-1/2-h}
    # (axial outflow on both sides of z = 0 with the upward bias j0)
    poloidal = (cfg.amplitude * ramp * q_pow * 0.25
                * (4.0 * eta + cfg.j0) * gauss_r * gauss_z)

    # non-axisymmetric annular wave packets (m = 1 via the smooth factors x/L,
    # y/L; zero angular mean, compact by construction).
    #
    # Scaling is derived from the DESIRED VELOCITY scaling after the curl, not
    # by assigning the velocity exponent to the potential:
    #
    #   poloidal family:  u_z = 2P + r d_r P  ~ P k_X   (oscillatory part)
    #   swirl family:     u_theta = -d_r c    ~ c k_X
    #   with k_X ~ q^{-h/2} (see below) and the paper's u_wave ~ q^{-1/2-h/2}:
    #       P ~ c ~ q^{-1/2}.
    #
    # The similarity wavenumber must itself grow as q^{-h/2}: with k_X fixed,
    # a fixed X-wavenumber gives lambda_r ~ q^{1/2}, whereas the paper's wave
    # packets have lambda_wave ~ q^{1/2+h/2}; h is small but sets the
    # asymptotic exponents under study.
    a1, a2, a3, a4 = wave_amps
    if a1 != 0.0 or a2 != 0.0 or a3 != 0.0 or a4 != 0.0:
        shear = q ** (-h / 2.0)
        k1 = cfg.ring_k1 * shear
        k2 = cfg.ring_k2 * shear
        wave_scale = q ** -0.5                   # derived above
        annulus = (((-((X - cfg.ring_x_center) / cfg.ring_x_width) ** 2).exp())
                   * ((-(eta / 2.5) ** 2).exp()))
        # X and eta are already dual numbers.  Wrapping them in another
        # Dual(...) nests duals (v becomes a Dual), numpy stores them as object
        # dtype and every downstream FFT fails.
        phase1 = k1 * X + cfg.ring_phase1
        phase2 = k2 * X + cfg.ring_phase2
        # The two families need different prefactors because the curl acts
        # differently on them.  The (x/L), (y/L) angular factors are O(sqrt q)
        # on the annulus (r ~ L sqrt q), and u_z = 2P + r d_r P while
        # u_theta = -d_r c, so the poloidal potential needs an extra q^{-1/2}
        # to give u_z ~ q^{-1/2-h/2}; the swirl family gets there directly.
        wave_scale_poloidal = wave_scale / q ** 0.5
        # The Dual must stay on the left of mixed products: numpy would
        # otherwise convert the dual operand into an object-dtype array.
        poloidal = poloidal + wave_scale_poloidal * annulus * (
            phase1.cos() * (a1 * (grid.X / scale))
            + phase2.sin() * (a2 * (grid.Y / scale)))
        swirl = swirl + wave_scale * annulus * (
            phase1.sin() * (a3 * (grid.X / scale))
            + phase2.cos() * (a4 * (grid.Y / scale)))

    window = (flat_window(grid.rr / cfg.support_radius,
                          cfg.window_inner / cfg.support_radius)
              * flat_window(grid.Z / cfg.support_z,
                            cfg.window_inner / cfg.support_z))
    # Dual on the left again: ax = -y P W, ay = +x P W, az = c W
    ax = poloidal * (-grid.Y) * window
    ay = poloidal * grid.X * window
    az = swirl * window
    return ax, ay, az


def potential_value_and_derivative(cfg: ReferenceConfig, t: float, grid: "SpectralGrid",
                                   wave_amps: tuple[float, float, float, float] | None = None
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """A and d_t A from ONE dual-number evaluation.

    Requesting value and derivative separately evaluates the whole
    construction twice; the fused pass halves the cost of every force call and
    stays bit-identical to it (same expression tree, same order).
    """
    ax, ay, az = potential_dual(cfg, t, grid, wave_amps)
    values = np.stack((ax.v, ay.v, az.v))
    derivs = np.stack((ax.d, ay.d, az.d))
    return values, derivs


# --------------------------------------------------------------------------
# spectral side (must match the DNS solver operators)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralGrid:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    rr: np.ndarray


def make_grid(n: int) -> SpectralGrid:
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return SpectralGrid(X, Y, Z, np.sqrt(X * X + Y * Y))


class ForceOracle:
    """Frozen-force oracle.

    Public interface for the solver: `force_hat(t)`, and nothing else.  All
    other methods are diagnostics-only.
    """

    def __init__(self, cfg: ReferenceConfig,
                 wave_amps: tuple[float, float, float, float] | None = None,
                 reference_factor: int | None = None):
        """Two modes, and they are NOT equivalent for a resolution study.

        reference_factor (default): the construction is evaluated on an
        n x factor grid, but u and d_t u are then restricted to the DNS grid
        and the nonlinearity is formed there.  The object handed to the solver
        is therefore f_N = P_N[d_t P_N u + N_N(P_N u) - nu Lap P_N u], which
        still depends on N through N_N and P_N: convenient, but not a fixed
        forcing.

        force_grid = N*: the construction AND the whole force are computed on
        the absolute grid N*, and only the finished force is restricted:

            f_N* = P_N*[d_t u + N(u) - nu Lap u],   f_N = P_N f_N*.

        Every DNS resolution then consumes restrictions of one and the same
        forcing - reproducible by caching f_N*(t_j) to disk with a hash.
        """
        cfg.validate()
        self.cfg = cfg
        self.wave_amps = wave_amps if wave_amps is not None else cfg.wave_coeffs
        self.k = wavenumbers(cfg.n)
        self.mask = retained_mask(cfg.n, cfg.dealias)
        self.grid = make_grid(cfg.n)
        cfg.assert_affordable()                  # before any big allocation
        self.reference_factor = max(int(reference_factor if reference_factor is not None
                                        else cfg.reference_factor), 1)
        if cfg.force_grid:
            # absolute grid: the whole force lives there, independent of n
            n_ref = cfg.reference_grid_size()
            self.force_on_reference = True
        else:
            n_ref = cfg.n * self.reference_factor
            if n_ref % 2:
                n_ref += 1
            self.force_on_reference = False
        self.n_ref = n_ref
        self.grid_ref = make_grid(n_ref)
        self.k_ref = wavenumbers(n_ref)
        self.mask_ref = retained_mask(n_ref, cfg.dealias)
        self._cache: dict[float, np.ndarray] = {}

    def _construction_hat(self, t: float, derivative: bool = False) -> np.ndarray:
        """curl of the potential on the reference grid, restricted to the DNS grid."""
        values, derivs = potential_value_and_derivative(self.cfg, t, self.grid_ref,
                                                        self.wave_amps)
        potentials = derivs if derivative else values
        hat = np.fft.fftn(potentials, axes=(1, 2, 3))
        field = curl_hat(hat, self.k_ref) * self.mask_ref
        return restrict_hat(field, self.cfg.n, self.n_ref)

    def construction_field_and_derivative(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(curl A, d_t curl A) on the reference grid, band-limited, UNRESTRICTED.

        One dual evaluation gives both halves.  The restriction is applied by
        the callers: for the fixed-N* force it must happen only AFTER the
        nonlinearity has been formed on the reference grid.
        """
        values, derivs = potential_value_and_derivative(self.cfg, t, self.grid_ref,
                                                        self.wave_amps)
        out: list[np.ndarray] = []
        for array in (values, derivs):
            hat = np.fft.fftn(array, axes=(1, 2, 3))
            out.append(curl_hat(hat, self.k_ref) * self.mask_ref)
        return out[0], out[1]

    def _construction_hats(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(curl A, d_t curl A) restricted to the DNS grid, from ONE dual pass."""
        u_hat, du_hat = self.construction_field_and_derivative(t)
        return (restrict_hat(u_hat, self.cfg.n, self.n_ref),
                restrict_hat(du_hat, self.cfg.n, self.n_ref))

    # -- force (public) ----------------------------------------------------
    def full_force_on_reference(self, t: float) -> np.ndarray:
        """f_N* = P_N*[d_t u + (u.grad)u + nu k^2 u] on the absolute grid.

        This is the one object a resolution study should share: with a fixed
        `force_grid` the DNS sees restrictions P_N f_N* of exactly this field.
        """
        cfg = self.cfg
        u_hat, du_hat = self.construction_field_and_derivative(t)
        _, _, _, k2_ref, _ = self.k_ref
        # rotational form: P[omega x u] = P[(u.grad)u] for divergence-free u
        adv_hat, _ = rotational_product(u_hat, self.k_ref, cfg.dealias)
        force = project(du_hat + adv_hat + cfg.viscosity * k2_ref * u_hat, self.k_ref)
        return force * self.mask_ref

    def force_hat(self, t: float) -> np.ndarray:
        """The only reference method the DNS is allowed to call.

        With a fixed `force_grid` this is P_N f_N* (one forcing, all grids).
        Otherwise it is the grid-relative f_N with the nonlinearity formed on
        the DNS grid (see the class docstring).
        """
        cfg = self.cfg
        if self.force_on_reference:
            # f_N = P_N f_N*: the restriction alone can fold content at the
            # coarse Nyquist plane, so the DNS retained-band mask is applied
            # afterwards, exactly as in the grid-relative path.
            restricted = restrict_hat(self.full_force_on_reference(t), cfg.n, self.n_ref)
            return restricted * self.mask
        _, _, _, k2, _ = self.k
        u_hat, du_dt = self._construction_hats(t)   # one dual pass, both halves
        u_hat = u_hat * self.mask
        du_dt = du_dt * self.mask
        adv_hat, _ = rotational_product(u_hat, self.k, cfg.dealias)
        force = project(du_dt + adv_hat + cfg.viscosity * k2 * u_hat, self.k)
        return force * self.mask

    def time_derivative_hat(self, t: float) -> np.ndarray:
        """d_t u* obtained by forward-mode AD through the construction."""
        return self._construction_hat(t, derivative=True) * self.mask

    def time_derivative_fd_hat(self, t: float) -> np.ndarray:
        """Central-difference d_t u* (independent cross-check of the AD result)."""
        cfg = self.cfg
        tau = max(1.0 - t, cfg.q_floor)
        eps = max(cfg.deriv_rel_step * tau, cfg.deriv_min_step)
        return (self.field_hat(t + eps) - self.field_hat(t - eps)) / (2.0 * eps)

    # -- diagnostics only --------------------------------------------------
    def field_hat(self, t: float) -> np.ndarray:
        """Constructed u* in spectral space: exactly divergence-free, no projection.

        Band-limiting multiplies every component by the same mask, so it cannot
        create divergence: div(masked u) = i k . (mask * u_hat) = 0 pointwise,
        because u_hat = i k x A_hat for the potential above.
        """
        key = round(t, 12)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._construction_hat(t) * self.mask
            self._cache.clear()
            self._cache[key] = cached
        return cached

    def pressure_hat(self, t: float) -> np.ndarray:
        """p_hat = i (k . conv_hat) / k^2 from Lap p = -div((u.grad)u)."""
        u_hat = self.field_hat(t)
        conv_hat, _ = convective_product(u_hat, self.k)
        kx, ky, kz, k2, _ = self.k
        dot = kx * conv_hat[0] + ky * conv_hat[1] + kz * conv_hat[2]
        p_hat = np.zeros_like(dot)
        nz = k2 != 0.0
        p_hat[nz] = 1j * dot[nz] / k2[nz]
        return p_hat

    def momentum_terms(self, t: float) -> dict[str, float]:
        """Momentum-balance diagnostics, formed on the construction grid.

        Absolute force_grid mode forms the force on N* (f_N* = P_N*[...]), and
        the terms are computed THERE - reconstructing them on the DNS grid
        would measure a different object.  Grid-relative mode forms the
        nonlinearity after restriction, so the DNS grid is the right place.

        Every term is filtered to the SAME retained band as the force before
        its norm is taken (multiply by the retained mask): the discrete PDE -
        and the masked force - never see modes above the mask, so unfiltered
        norms of the advection or pressure terms are not objects of the same
        discretisation.  The identity f = d_t u + (u.grad)u - nu Lap u + grad p
        then holds exactly term by term on the masked grid.

        'force / largest-term ratio' measures the max-norm ratio, which can be
        shifted by spatial separation of component peaks. The regional cancellation
        metric reports ||sum T_i||_2 / sum ||T_i||_2 over the annular region
        (0 = complete cancellation, 1 = none), accompanied by pointwise percentiles.

        'f vs oracle' measures the restriction-after-nonlinearity error:
        in grid-relative mode it checks reconstruction consistency; in absolute
        mode it compares the DNS-grid reconstruction against P_N f_N*.
        """
        cfg = self.cfg
        if self.force_on_reference:
            kx, ky, kz, k2, _ = self.k_ref
            k, mask, grid = self.k_ref, self.mask_ref, self.grid_ref
            u_hat, du_dt = self.construction_field_and_derivative(t)
            conv_hat, _ = convective_product(u_hat, self.k_ref)
            visc_hat = cfg.viscosity * k2 * u_hat
            dot = kx * conv_hat[0] + ky * conv_hat[1] + kz * conv_hat[2]
            gradp_hat = np.empty_like(conv_hat)
            for axis, kk in enumerate((kx, ky, kz)):
                gradp_hat[axis] = -kk * dot / np.where(k2 != 0.0, k2, 1.0)
            force_hat = self.mask_ref * project(du_dt + conv_hat + visc_hat,
                                                self.k_ref)
            # DNS-grid reconstruction of the same terms (what grid-relative
            # mode would hand the solver)
            _, _, _, k2_n, _ = self.k
            u_n, du_n = self._construction_hats(t)
            u_n = u_n * self.mask
            du_n = du_n * self.mask
            adv_n, _ = rotational_product(u_n, self.k, cfg.dealias)
            force_n = self.mask * project(du_n + adv_n + cfg.viscosity * k2_n * u_n,
                                          self.k)
            restricted = restrict_hat(force_hat, cfg.n, self.n_ref) * self.mask
            mismatch = (float(np.linalg.norm(physical(restricted - force_n)))
                        / max(float(np.linalg.norm(physical(restricted))), 1e-30))
        else:
            kx, ky, kz, k2, _ = self.k
            k, mask, grid = self.k, self.mask, self.grid
            du_dt = self.time_derivative_hat(t)      # AD, not finite differences
            u_hat = self.field_hat(t)
            conv_hat, _ = convective_product(u_hat, self.k)
            visc_hat = cfg.viscosity * k2 * u_hat
            dot = kx * conv_hat[0] + ky * conv_hat[1] + kz * conv_hat[2]
            gradp_hat = np.empty_like(conv_hat)
            for axis, kk in enumerate((kx, ky, kz)):
                gradp_hat[axis] = -kk * dot / np.where(k2 != 0.0, k2, 1.0)
            force_hat = self.mask * project(du_dt + conv_hat + visc_hat, self.k)
            oracle_force = self.force_hat(t)
            mismatch = (float(np.linalg.norm(physical(oracle_force - force_hat)))
                        / max(float(np.linalg.norm(physical(oracle_force))), 1e-30))

        term_hats = {"d_t u": du_dt, "(u.grad)u": conv_hat,
                     "-nu Lap u": visc_hat, "grad p": gradp_hat}
        # band-filter every term: the same mask the force carries
        term_vecs = {name: physical(mask * hat) for name, hat in term_hats.items()}

        def max_norm(vec):
            return float(np.max(np.sqrt(np.sum(vec * vec, axis=0))))

        terms = {name: max_norm(vec) for name, vec in term_vecs.items()}
        total = sum(term_vecs.values())              # == physical(mask * force_hat)
        terms["f (sum)"] = max_norm(total)
        terms["f vs oracle"] = mismatch
        biggest = max(terms[name] for name in term_hats)
        # Max-norm ratio and regional annular L2 cancellation
        terms["force / largest-term ratio"] = terms["f (sum)"] / max(biggest, 1e-30)

        # Regional annular cancellation metric
        weight = annulus_weight(cfg, t, grid)
        weight = weight / max(float(np.sum(weight)), 1e-30)

        def l2(vec):
            return float(np.sqrt(np.sum(weight * np.sum(vec * vec, axis=0))))

        terms["cancellation"] = l2(total) / max(
            sum(l2(vec) for vec in term_vecs.values()), 1e-30)
        magnitude = sum(np.abs(vec) for vec in term_vecs.values())
        pointwise = np.abs(total).sum(axis=0) / np.maximum(
            magnitude.sum(axis=0), 1e-30)
        region = weight > 5.0e-2 * float(weight.max())
        values = pointwise[region] if np.any(region) else pointwise.ravel()
        for percentile in (10, 50, 90):
            terms[f"cancellation p{percentile}"] = float(
                np.percentile(values, percentile))
        return terms
