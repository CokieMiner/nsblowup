"""Wave-sector analysis built on one basis definition.

All wave diagnostics use the four basis fields:

    w_i = curl A(a = e_i) - curl A(a = 0),      i = 1..4

with unit vectors e_i in `WAVE_BASIS`; `FAMILY_A = (0, 2)` and
`FAMILY_B = (1, 3)` are the pairings used.

Key diagnostics:
* `stress_matrices`: Annulus-averaged Reynolds-stress moments of the wave
  velocities (units u^2), characterizing basis structure.
* `full_quadratic_stress_balance`: Discrete quadratic stress divergence (units u/t),
  computed as M P N_h(w) with the same discrete rotational operator and
  dealiasing as the force. Includes both m = 0 and m = 2 azimuthal harmonics.

Carrier wavenumber and resolution diagnostics are evaluated on the DNS grid
to monitor retained-band coverage. The amplitude optimization model lives in
`nsblowup.reference.quadratic`.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from nsblowup.core.config import ReferenceConfig
from nsblowup.core.spectral import retained_mask, rotational_product
from nsblowup.reference.reference import (ForceOracle, SpectralGrid, annulus_weight,
                                          make_grid, physical, project, wavenumbers)

# --------------------------------------------------------------------------
# the one basis definition
# --------------------------------------------------------------------------
WAVE_BASIS: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
FAMILY_A = (0, 2)          # in-phase/quadrature of the cos(theta) packets
FAMILY_B = (1, 3)          # the sin(theta) packets
CONSTRUCTION_SOURCES = ("nsblowup/core/config.py", "nsblowup/core/dual.py",
                        "nsblowup/core/spectral.py", "nsblowup/core/frozen.py",
                        "nsblowup/reference/reference.py",
                        "nsblowup/reference/waves.py",
                        "nsblowup/reference/quadratic.py",
                        "nsblowup/reference/tables.py")
# files that change what a DNS RUN does; the construction alone does not.
# run/volume.py lives HERE, not in the construction: the renderer only samples
# the field for tracers and pixels (sample_vector/upsample2), so editing it
# must not invalidate the calibration or the frozen tables.
RUN_SOURCES = CONSTRUCTION_SOURCES + ("nsblowup/run/solver.py",
                                      "nsblowup/run/cli.py",
                                      "nsblowup/run/volume.py")
# backwards-compatible alias: the construction identity used by calibration
MODEL_SOURCES = CONSTRUCTION_SOURCES


def _hash_sources(names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"code_hash source missing: {name}")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def code_hash() -> str:
    """Construction identity: construction, operators and tables code.

    Stamps calibration metadata and every number the reference prints.  The
    renderer/sampling layer (run/volume.py) is not part of this
    identity - see RUN_SOURCES.
    """
    return _hash_sources(CONSTRUCTION_SOURCES)


def run_code_hash() -> str:
    """Run identity: construction sources PLUS solver, front end and renderer.

    The archive records both, because changing IMEX-RK2, the run plumbing or
    the sampling/rendering layer changes what a run means while leaving the
    construction identical.
    """
    return _hash_sources(RUN_SOURCES)


def metadata(cfg: ReferenceConfig, **extra) -> dict:
    """Configuration metadata: grids, conventions, window - plus code hash."""
    info = {
        "code_hash": code_hash(),
        "n": cfg.n,
        "reference_factor": cfg.reference_factor,
        "reference_grid": cfg.reference_grid_size(),
        "force_grid": getattr(cfg, "force_grid", None),
        "dealias": cfg.dealias,
        "h": cfg.h,
        "q_floor": cfg.q_floor,
        "viscosity": cfg.viscosity,
    }
    info.update(extra)
    return info


# --------------------------------------------------------------------------
# basis velocity fields
# --------------------------------------------------------------------------
def basis_fields(cfg: ReferenceConfig, t: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (u_base, [w_1, ..., w_4]) as physical-space vector fields.

    The construction is linear in the wave coefficients, so w_i is the exact
    curl of the potential with the i-th unit coefficient minus the base field.
    """
    base = physical(ForceOracle(cfg, wave_amps=(0.0, 0.0, 0.0, 0.0)).field_hat(t))
    fields = [physical(ForceOracle(cfg, wave_amps=unit).field_hat(t)) - base
              for unit in WAVE_BASIS]
    return base, fields


def mechanism_fields(cfg: ReferenceConfig, t: float
                     ) -> tuple[SpectralGrid, np.ndarray, list[np.ndarray]]:
    """(grid, u_base, [w_i]) on the grid where the force's nonlinearity forms.

    Absolute force_grid mode builds the force on N*: the stress and momentum
    diagnostics must be formed there too.  Grid-relative mode forms the
    nonlinearity on the DNS grid (u is restricted first), so its diagnostics
    live there.  Carrier diagnostics intentionally use `basis_fields`, which
    always restricts to the DNS grid.
    """
    oracle = ForceOracle(cfg)
    if oracle.force_on_reference:
        grid = oracle.grid_ref
        # the BASE field is u*(a = 0), NOT u*(a = cfg.wave_coeffs): subtracting
        # the configured field would make w_i = u(e_i) - u(a_cfg) and silently
        # corrupt every stress diagnostic whenever wave_coeffs != 0
        zero = (0.0, 0.0, 0.0, 0.0)
        base_hat = ForceOracle(cfg, wave_amps=zero).construction_field_and_derivative(t)[0]
        fields = [physical(ForceOracle(cfg, wave_amps=unit)
                           .construction_field_and_derivative(t)[0] - base_hat)
                  for unit in WAVE_BASIS]
        return grid, physical(base_hat), fields
    _, fields = basis_fields(cfg, t)
    base = physical(ForceOracle(cfg, wave_amps=(0.0, 0.0, 0.0, 0.0)).field_hat(t))
    return make_grid(cfg.n), base, fields


def cylindrical_components(field: np.ndarray, grid: SpectralGrid
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_r = np.where(grid.rr > 1e-12, 1.0 / grid.rr, 0.0)
    er = np.stack((grid.X * inv_r, grid.Y * inv_r, np.zeros_like(grid.X)))
    et = np.stack((-grid.Y * inv_r, grid.X * inv_r, np.zeros_like(grid.X)))
    ez = np.stack((np.zeros_like(grid.X), np.zeros_like(grid.X), np.ones_like(grid.X)))
    return (np.sum(field * er, axis=0), np.sum(field * et, axis=0),
            np.sum(field * ez, axis=0))


# --------------------------------------------------------------------------
# Reynolds-stress moments of the basis (units u^2; characterisation only)
# --------------------------------------------------------------------------
def stress_matrices(cfg: ReferenceConfig, tau: float) -> dict[str, np.ndarray]:
    """Annulus-weighted stress moments C_theta, C_z of the wave basis.

        (C_theta)_{ij} = 1/2 < w_{i,r} w_{j,theta} + w_{j,r} w_{i,theta} >
        (C_z)_{ij}     = 1/2 < w_{i,r} w_{j,z}     + w_{j,r} w_{i,z}     >

    with <...> the annulus_weight-normalised mean at fixed tau, formed on the
    construction grid (N* in absolute force_grid mode, the DNS grid
    otherwise).  These are u^2 objects: they describe which quadratic products
    the basis can deliver and are not comparable to the u/t residual.
    """
    t = 1.0 - tau
    grid, _, fields = mechanism_fields(cfg, t)
    weight = annulus_weight(cfg, t, grid)
    weight = weight / max(float(np.sum(weight)), 1e-30)

    comps = [cylindrical_components(field, grid) for field in fields]
    c_theta = np.zeros((4, 4))
    c_z = np.zeros((4, 4))
    for i in range(4):
        for j in range(i, 4):
            wr_i, wt_i, wz_i = comps[i]
            wr_j, wt_j, wz_j = comps[j]
            c_theta[i, j] = c_theta[j, i] = float(
                np.sum(weight * 0.5 * (wr_i * wt_j + wr_j * wt_i)))
            c_z[i, j] = c_z[j, i] = float(
                np.sum(weight * 0.5 * (wr_i * wz_j + wr_j * wz_i)))
    return {"C_theta": c_theta, "C_z": c_z, "weight": "annulus_weight", "tau": tau}


# --------------------------------------------------------------------------
# mean-field balance: R_bar_B + div<w (x) w>   (units u/t)
# --------------------------------------------------------------------------
def full_quadratic_stress_balance(cfg: ReferenceConfig, tau: float,
                                  amps: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
                                  ) -> dict[str, object]:
    """Discrete quadratic stress balance R + lambda D, Cartesian.

        D = M P N_h(w),  N_h(w) = (curl w) x w in cfg.dealias, M the retained
                         mask, P the Leray projector
        R = residual of the base flow (masked, same band)

    For divergence-free w, P N_h(w) = P div (w (x) w), evaluated through the
    same discrete rotational operator and dealiasing as the forcing.

    Includes both m = 0 and m = 2 azimuthal harmonics. Formed on the construction
    grid (N* in absolute force_grid mode, the DNS grid otherwise) with the
    annular weight applied for norms.

    lambda is the signed shape alignment; physical_lambda = max(lambda, 0)
    enforces non-negative scaling. Ratios report remaining norm ||R + lambda D|| / ||R||.
    """
    t = 1.0 - tau
    grid, _, fields = mechanism_fields(cfg, t)
    wave = sum(a * w for a, w in zip(amps, fields))
    oracle = ForceOracle(cfg, wave_amps=(0.0, 0.0, 0.0, 0.0))
    residual = physical(oracle.full_force_on_reference(t)
                        if oracle.force_on_reference else oracle.force_hat(t))

    n_grid = grid.X.shape[0]
    k = wavenumbers(n_grid)
    mask = retained_mask(n_grid, cfg.dealias)
    wave_hat = np.fft.fftn(wave, axes=(1, 2, 3))
    adv_hat, _ = rotational_product(wave_hat, k, cfg.dealias)
    div_t = physical(mask * project(adv_hat, k))

    weight = annulus_weight(cfg, t, grid)
    weight = weight / max(float(np.sum(weight)), 1e-30)

    def norm(field: np.ndarray) -> float:
        return float(np.sqrt(np.sum(weight * np.sum(field * field, axis=0))))

    target_energy = norm(residual)
    stress_energy = norm(div_t)
    inner = float(np.sum(weight * np.sum(residual * div_t, axis=0)))
    lam = -inner / max(stress_energy ** 2, 1e-30)
    lam_phys = max(lam, 0.0)
    combined_signed = norm(residual + lam * div_t)
    combined_phys = norm(residual + lam_phys * div_t)
    return {
        "tau": tau,
        "R_bar": residual,
        "div_T": div_t,
        "target_energy": target_energy,
        "stress_div_energy": stress_energy,
        "signed_alignment_lambda": lam,
        "physical_lambda": lam_phys,
        "combined_energy_signed": combined_signed,
        "combined_energy": combined_phys,        # physically realisable one
        "remaining_norm_ratio_signed": combined_signed / max(target_energy, 1e-30),
        "remaining_norm_ratio_phys": combined_phys / max(target_energy, 1e-30),
        "weight": "annulus_weight on the construction grid",
        "harmonics": "full quadratic stress (m=0 and m=2); no m=0 projection",
    }


def residual_objective(cfg: ReferenceConfig, t: float,
                       amps: tuple[float, float, float, float]) -> dict[str, float]:
    """Box RMS and annulus-weighted L2 norm of the residual.

    The annulus-weighted L2 with the SAME normalised weight is the objective
    the quadratic calibration minimises (`nsblowup.reference.quadratic`); the
    box RMS is the cruder reference used in earlier tables.
    """
    oracle = ForceOracle(cfg, wave_amps=amps)
    field = physical(oracle.force_hat(t))
    box = float(np.sqrt(np.mean(field ** 2)))
    weight = annulus_weight(cfg, t, make_grid(cfg.n))
    weight = weight / max(float(np.sum(weight)), 1e-30)
    annulus = float(np.sqrt(np.sum(weight * np.sum(field * field, axis=0))))
    return {"box_rms": box, "annulus_norm": annulus}


# --------------------------------------------------------------------------
# carrier wavenumber, fitted only where the packets are resolved
# --------------------------------------------------------------------------
CARRIER_VECTORS: dict[str, tuple[float, float, float, float]] = {
    "w1": (1.0, 0.0, 0.0, 0.0),
    "w2": (0.0, 1.0, 0.0, 0.0),
    "w3": (0.0, 0.0, 1.0, 0.0),
    "w4": (0.0, 0.0, 0.0, 1.0),
    "family A": (1.0, 0.0, 1.0, 0.0),
    "family B": (0.0, 1.0, 0.0, 1.0),
    "all four": (1.0, 1.0, 1.0, 1.0),
}


def carrier_records(cfg: ReferenceConfig, taus=(0.2, 0.3, 0.45, 0.6, 0.75),
                    retained_fraction: float = 0.8, energy_gate: float = 0.99,
                    vectors: dict[str, tuple[float, float, float, float]] | None = None
                    ) -> list[dict]:
    """Carrier wavenumber per tau for EACH wave vector, with a worst-case gate.

    A single FFT peak of one component of the summed field can be moved by
    cross-family cancellation, so records are produced separately for w1..w4,
    for families A and B, and for the full sum.  A tau is FITTABLE only if the
    WORST of those fields keeps at least `energy_gate` of its energy inside
    `retained_fraction` of the retained band; beyond that a measured slope is
    truncation rather than physics.
    """
    vectors = CARRIER_VECTORS if vectors is None else vectors
    cut = (cfg.n - 1) // 3
    m = np.fft.fftfreq(cfg.n, d=1.0 / cfg.n)
    mx, my, mz = np.meshgrid(m, m, m, indexing="ij")
    shell = (np.maximum(np.maximum(np.abs(mx), np.abs(my)), np.abs(mz))
             <= retained_fraction * cut)
    iz = cfg.n // 2
    records = []
    for tau in taus:
        t = 1.0 - tau
        q = max(float(tau), cfg.q_floor)   # z = 0: q = tau, floored by domain
        _, fields = basis_fields(cfg, t)
        carriers: dict[str, float] = {}
        inside: dict[str, float] = {}
        for name, amps in vectors.items():
            wave = sum(a * w for a, w in zip(amps, fields))
            # The packages are m=1: the cos-oriented families (w1, w3) vanish on
            # the y-axis line and the sin-oriented ones (w2, w4) on the x-axis
            # line, so measure on whichever of the two carries the field.
            line_x = wave[:, :, iz, iz]
            line_y = wave[:, iz, :, iz]
            comps = line_x if np.sum(line_x ** 2) >= np.sum(line_y ** 2) else line_y
            line = comps[int(np.argmax(np.sum(comps ** 2, axis=1)))]
            spectrum = np.abs(np.fft.rfft(line))
            carriers[name] = float(np.argmax(spectrum[1:]) + 1)
            full = np.fft.fftn(wave, axes=(1, 2, 3))
            energy = np.sum(np.abs(full) ** 2, axis=0)
            inside[name] = (float(np.sum(energy[shell]))
                            / max(float(np.sum(energy)), 1e-30))
        worst = min(inside.values())
        records.append({"tau": tau, "q": q, "carriers": carriers,
                        "energy_inside": inside, "worst_inside": worst,
                        "fittable": worst >= energy_gate,
                        "n": cfg.n, "cut": cut,
                        "retained_fraction": retained_fraction,
                        "energy_gate": energy_gate,
                        "metric": "FFT peak of the loudest component on the brighter "
                                  "axis line (y=z=0 or x=z=0), wave-only velocity"})
    return records
