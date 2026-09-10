"""Diagnostics of the reference construction.

Prints the momentum balance, the wave-sector model fit and the numerical
cross-checks.  Everything here is read-only: no part of this module changes the
forcing that the solver consumes.
"""

from __future__ import annotations

import numpy as np

from nsblowup.core.config import H_LIMIT, ReferenceConfig
from nsblowup.core.spectral import convective_product, physical, project, rotational_product
from nsblowup.reference import quadratic, waves
from nsblowup.reference.tables import (parent_force_data_fingerprint,
                                       restricted_force_fingerprint)
from nsblowup.reference.reference import ForceOracle
from nsblowup.run.volume import sample_vector


def derivative_ad_vs_fd(cfg: ReferenceConfig, t: float
                        ) -> tuple[float, float, float, float]:
    """Compare the dual-number derivative of u* against a central difference.

    Returns (ad_rms, fd_rms, fd_eps, relative_difference); the finite
    difference is only a cross-check of the automatic derivative.
    """
    oracle = ForceOracle(cfg)
    ad = physical(oracle.time_derivative_hat(t))
    fd = physical(oracle.time_derivative_fd_hat(t))
    ad_rms = float(np.sqrt(np.mean(ad ** 2)))
    fd_rms = float(np.sqrt(np.mean(fd ** 2)))
    tau = max(1.0 - t, cfg.q_floor)
    eps = max(cfg.deriv_rel_step * tau, cfg.deriv_min_step)
    rel = float(np.sqrt(np.mean((ad - fd) ** 2)) / max(ad_rms, 1e-30))
    return ad_rms, fd_rms, eps, rel


def advection_fd_check(cfg: ReferenceConfig, t: float) -> tuple[float, float, float]:
    """Compare the spectral (u.grad)u against a 4th-order finite difference.

    Measured as interior L2 norms (r < 0.85 support): the max norm is set by
    the thin localization layer, which a finite-difference stencil on this grid
    cannot represent.  Also returns the defect of the rotational identity

        mask * P[omega x u] = mask * P[(u.grad)u]

    evaluated on the retained band: the two raw products differ outside that
    band (their out-of-band content folds differently), and the retained band
    is the space the solver integrates in, so the check is made there.
    """
    oracle = ForceOracle(cfg)
    u_hat = oracle.field_hat(t)
    conv_hat, u = convective_product(u_hat, oracle.k)
    rot_hat, _ = rotational_product(u_hat, oracle.k, cfg.dealias)
    spectral = physical(conv_hat)

    n = cfg.n
    dx = 1.0 / n
    derivs = []
    for axis in range(3):
        acc = np.zeros_like(u)
        for offset, weight in zip((-2, -1, 1, 2), (1.0, -8.0, 8.0, -1.0)):
            acc += weight * np.roll(u, -offset, axis=axis + 1)
        derivs.append(acc / (12.0 * dx))
    adv_fd = np.zeros_like(u)                     # (u.grad)u_i = sum_j u_j d_j u_i
    for j in range(3):
        adv_fd += u[j] * derivs[j]

    interior = (oracle.grid.rr <= 0.85 * cfg.support_radius).astype(float)

    def interior_l2(field):
        return float(np.sqrt(np.sum(interior * np.sum(field * field, axis=0))
                             / max(np.sum(interior), 1e-30)))

    conv_banded = physical(oracle.mask * project(conv_hat, oracle.k))
    rot_banded = physical(oracle.mask * project(rot_hat, oracle.k))
    identity = (interior_l2(conv_banded - rot_banded)
                / max(interior_l2(conv_banded), 1e-30))
    return interior_l2(spectral), interior_l2(adv_fd), identity


def swirl_collapse_records(cfg: ReferenceConfig,
                           taus=(0.15, 0.25, 0.4, 0.6, 0.75),
                           reference_tau: float = 0.45, points: int = 160) -> list[dict]:
    """Self-similarity check of the swirl profile in similarity variables.

    Samples the azimuthal velocity on the z = 0 half-line, rescales
    q^{1/2+h} u_theta and evaluates it at fixed X = (r/L)^2/(2q) by choosing
    r = sqrt(2X) L sqrt(q) per time.  On a self-similar field the curves at
    different tau coincide; the defect is the relative RMS difference from the
    curve at `reference_tau`.  DNS-side profiles need `--store-swirl`.
    """
    scale = cfg.similarity_scale
    # the clock is used as solved (no soft floor): q_floor is the smallest tau
    # reached, and frames beyond the resolution stop keep the floor clamp
    q_values = {tau: float(max(tau, cfg.q_floor)) for tau in taus}
    reference_tau = min(taus, key=lambda tau: abs(tau - reference_tau))
    x_hi = 0.9 * (0.42 ** 2) / (2.0 * scale ** 2 * max(q_values.values()))
    x_grid = np.linspace(0.02, x_hi, points)
    curves = {}
    for tau, q in q_values.items():
        radius = np.sqrt(2.0 * x_grid) * scale * np.sqrt(q)
        positions = np.stack((radius, np.zeros_like(radius), np.zeros_like(radius)),
                             axis=1)
        sample = sample_vector(physical(ForceOracle(cfg).field_hat(1.0 - tau)),
                               positions, cfg.n)
        curves[tau] = q ** (0.5 + cfg.h) * sample[:, 1]     # u_theta at y = 0
    reference = curves[reference_tau]
    norm = max(float(np.sqrt(np.mean(reference ** 2))), 1e-30)
    return [{"tau": tau, "q": q_values[tau],
             "defect": float(np.sqrt(np.mean((curves[tau] - reference) ** 2)) / norm)}
            for tau in taus]


def report(cfg: ReferenceConfig, calibrate_waves: bool
           ) -> tuple[tuple[float, ...], float, float] | None:
    """Print the full construction report.

    Returns the four fitted wave coefficients (and the residual energies before
    and after) when `calibrate_waves` is set, otherwise None.
    """
    print("Reference construction (finite-scale surrogate)")
    print(f"  similarity: A=1/2+h, D=1/2-h, h={cfg.h} (limit {H_LIMIT}), "
          f"q_floor={cfg.q_floor}, t_end={cfg.t_end:.3f}")
    print(f"  grid {cfg.n}^3 · dealias={cfg.dealias} · nu={cfg.viscosity} · "
          f"j0={cfg.j0} · support r<= {cfg.support_radius}, |z| <= {cfg.support_z}")
    print("  pre-spectral construction: C-infinity compact window and smooth")
    print("  start (u(.,0)=0); the band-limited discrete field is globally supported")

    print("\n  momentum balance (max norms of terms filtered to the retained")
    print("  band the force carries, formed on the construction grid: N* in")
    print("  --force-grid mode, the DNS grid otherwise; 'f/max term' is NOT a")
    print("  cancellation measure - the largest terms can peak at different")
    print("  cells; the cancellation line below is the real one; 'check'")
    print("  compares the DNS-grid reconstruction against the force - roundoff")
    print("  in grid-relative mode, restriction-after-nonlinearity in N* mode)")
    print(f"  {'tau':>6} {'d_t u':>10} {'(u.g)u':>10} {'-nu Lap u':>10} "
          f"{'grad p':>10} {'f':>10} {'f/max term':>10} {'check':>8}")
    for tau in (0.10, 0.20, 0.40, 0.70):
        terms = ForceOracle(cfg).momentum_terms(1.0 - tau)
        print(f"  {tau:6.2f} {terms['d_t u']:10.3f} {terms['(u.grad)u']:10.3f} "
              f"{terms['-nu Lap u']:10.3f} {terms['grad p']:10.3f} "
              f"{terms['f (sum)']:10.3f} "
              f"{terms['force / largest-term ratio']:10.4f} "
              f"{terms['f vs oracle']:8.1e}")
        print(f"  {'':6} cancellation (annulus region) "
              f"{terms['cancellation']:.4f} · pointwise p10/p50/p90 "
              f"{terms['cancellation p10']:.3f}/"
              f"{terms['cancellation p50']:.3f}/"
              f"{terms['cancellation p90']:.3f}")

    print("\n  annular wave sector")
    zero = (0.0, 0.0, 0.0, 0.0)
    model_taus = (0.4, 0.6)
    fitted: tuple[tuple[float, ...], float, float] | None = None
    if calibrate_waves:
        model = quadratic.quadratic_model(cfg, taus=model_taus)
        best = quadratic.optimize_amplitudes(model)
        check = quadratic.verify_optimizer(cfg, model, best)
        regress = quadratic.verify_model_random(cfg, model)
        fitted = (tuple(best["amps"]), best["energy_zero"], best["energy"])
        coeffs = tuple(best["amps"])
        print(f"    basis: w_i = curl A(a = e_i), i = 1..{len(waves.WAVE_BASIS)} "
              f"(families A={waves.FAMILY_A}, B={waves.FAMILY_B}); taus={model_taus}")
        print(f"    |gradient| = {np.linalg.norm(best['gradient']):.4e} · "
              f"Hessian at a=0 positive definite: "
              f"{bool(np.all(best['hessian_eigenvalues'] > 0.0))}")
        print(f"    local (Newton) quadratic prediction of the achievable reduction: "
              f"{best['local_quadratic_predicted_reduction']:.3e}")
        print(f"    log-scale free-phase scan: {best['scan_energy']:.8f} vs "
              f"{best['energy_zero']:.8f} at a=0 (winner: {best['source']})")
        print(f"    optimum a* = {np.array2string(best['amps'], precision=3)}")
        if all(abs(value) < 1e-12 for value in best["amps"]):
            print("    note: in the current discrete system the optimizer prefers ZERO;")
            print("    the basis packets do not pass the retained-band gate at this n,")
            print("    so this does not test the paper's cancellation mechanism.")
        print(f"    model checked against true force evaluations: model "
              f"{check['model_energy']:.8f} vs true {check['true_energy']:.8f} "
              f"(rel. error {check['relative_model_error']:.1e})")
        print(f"    model vs {regress['trials']} random amplitude vectors "
              f"(10^-5..1): worst relative error "
              f"{regress['worst_relative_error']:.1e}")
        print("    note: the Newton value is local; the scan is a coarse sample;")
        print("    the objective is quartic, so neither is a global certificate;")
        print("    it is the finite-tau norm, not the asymptotic cancellation.")
        print(f"  {'tau':>6} {'base box':>10} {'waves box':>10} {'box ratio':>10} "
              f"{'base ann':>10} {'waves ann':>10} {'ann ratio':>10}")
        print("  (box = RMS; ann = the annulus-weighted L2 objective the fit")
        print("  minimises; ratios are reductions base/waves)")
        for tau in (0.15, 0.25, 0.4, 0.6):
            base = waves.residual_objective(cfg, 1.0 - tau, zero)
            with_waves = waves.residual_objective(cfg, 1.0 - tau, coeffs)
            print(f"  {tau:6.2f} {base['box_rms']:10.5f} "
                  f"{with_waves['box_rms']:10.5f} "
                  f"{base['box_rms'] / max(with_waves['box_rms'], 1e-30):9.3f}x "
                  f"{base['annulus_norm']:10.5f} "
                  f"{with_waves['annulus_norm']:10.5f} "
                  f"{base['annulus_norm'] / max(with_waves['annulus_norm'], 1e-30):9.3f}x")
    else:
        print("    (run --calibrate for the quadratic model of this sector, or")
        print("     --save-calibration to write wave_calibration.json)")

    print("\n  wave basis stress moments (annulus-weighted, units u^2: basis")
    print("  characterisation only, not comparable to a residual; formed on the")
    print("  construction grid)")
    mat = waves.stress_matrices(cfg, 0.6)
    for name in ("C_theta", "C_z"):
        print(f"    {name}:")
        for row in mat[name]:
            print("     " + "".join(f"{value:12.5f}" for value in row))

    print("\n  full quadratic stress balance R + lambda D  (Cartesian, units u/t;")
    print("  D = M P N_h(w) - same nonlinear operator and dealiasing as the")
    print("  force, masked; includes the m=0 AND m=2 azimuthal harmonics -")
    print("  NOT the azimuthal mean; formed on the construction grid; 'ratio'")
    print("  = ||R + lambda D|| / ||R||, a REMAINING norm: 0.996 = 0.4%")
    print("  reduction)")
    print(f"  {'tau':>6} {'||R||':>10} {'||D||':>10} {'lam_signed':>12} "
          f"{'lam_phys':>10} {'||R+ls D||':>11} {'ratio_s':>8} "
          f"{'||R+lp D||':>11} {'ratio_p':>8}")
    for tau in (0.4, 0.6):
        bal = waves.full_quadratic_stress_balance(cfg, tau)
        print(f"  {tau:6.2f} {bal['target_energy']:10.4f} "
              f"{bal['stress_div_energy']:10.4f} "
              f"{bal['signed_alignment_lambda']:+12.3e} "
              f"{bal['physical_lambda']:10.4f} "
              f"{bal['combined_energy_signed']:11.4f} "
              f"{bal['remaining_norm_ratio_signed']:8.4f} "
              f"{bal['combined_energy']:11.4f} "
              f"{bal['remaining_norm_ratio_phys']:8.4f}")

    print("\n  carrier wavenumber per wave vector vs retained band (worst-case gate)")
    for rec in waves.carrier_records(cfg):
        summary = " ".join(f"{name}={value:.0f}"
                           for name, value in rec["carriers"].items())
        gate = "FITTABLE" if rec["fittable"] else "unresolved (do not use for carrier-scaling fits)"
        print(f"  tau={rec['tau']:.2f} q={rec['q']:.2f}  [{summary}]  "
              f"worst energy inside {rec['retained_fraction']:.2f}x cut: "
              f"{rec['worst_inside']:.3f}  {gate}")

    print("\n  swirl-profile self-similarity (analytic field, z = 0 half-line)")
    print("  q^{1/2+h} u_theta at fixed X = (r/L)^2/(2q); defect = relative RMS"
          " vs tau=0.45 (DNS-side profiles need --store-swirl)")
    for rec in swirl_collapse_records(cfg):
        print(f"  tau={rec['tau']:.2f} q={rec['q']:.3f}  defect {rec['defect']:.3e}")

    meta = waves.metadata(cfg, taus=list(model_taus),
                          metric="annulus-weighted L2 of the residual",
                          carrier_metric="gated FFT peak, wave-only velocity")
    print("  metadata: " + "  ".join(f"{key}={value}" for key, value in meta.items()))

    parent = parent_force_data_fingerprint(cfg)
    print(f"\n  force fingerprints at tau=0.3, 0.6: restricted P_N f_N* "
          f"{restricted_force_fingerprint(cfg)}"
          + (f" · parent f_N* canonical data hash {parent}"
             if cfg.force_grid else ""))

    print("\n  d_t u_ref: automatic derivative vs central difference")
    for t in (0.3, 0.7):
        ad_rms, fd_rms, eps, rel = derivative_ad_vs_fd(cfg, t)
        print(f"  t={t}: AD rms={ad_rms:.6f}   FD rms={fd_rms:.6f} (eps={eps:.2e})   "
              f"relative difference {rel:.2e}")

    print("\n  advection cross-check: (u.grad)u spectral vs 4th-order differences,")
    print("  and the rotational identity mask*P[omega x u] = mask*P[(u.grad)u]")
    for t in (0.3, 0.7):
        spectral, fd, identity = advection_fd_check(cfg, t)
        print(f"  t={t}: advective spectral {spectral:10.4f}   FD {fd:10.4f}   "
              f"rel.diff {abs(fd - spectral) / max(spectral, 1e-30):.2%}   "
              f"banded identity {identity:.2e}")
    return fitted
