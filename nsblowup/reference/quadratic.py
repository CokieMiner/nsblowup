"""Amplitude model of the annular residual: exact quadratic fit and optimizer.

For fixed tau the discrete residual is quadratic in the four wave amplitudes
(d_t and the viscous term are linear, (u.grad)u is quadratic), so 15 force
evaluations per tau determine the weighted residual energy of EVERY amplitude
vector in R^4.  The optimum is then found on that algebraic model - local
Newton analysis at a = 0 plus a log-amplitude scan over free phases - and
verified against true force evaluations.  The basis itself lives in
`nsblowup.reference.waves`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nsblowup.core.config import ReferenceConfig
from nsblowup.reference.reference import (ForceOracle, annulus_weight,
                                          make_grid, physical)
from nsblowup.reference.waves import WAVE_BASIS, metadata


# --------------------------------------------------------------------------
# exact quadratic model of the residual in the amplitudes
# --------------------------------------------------------------------------
@dataclass
class QuadraticModel:
    """Exact expansion R(a) = R0 + sum_i a_i c_i + sum_ij a_i a_j G_ij per tau.

    `energies[k]` is the Gram matrix of the annulus-weighted model fields for
    tau_k in the monomial order of `monomials()`, so the weighted residual
    energy of ANY amplitude vector is m(a)^T Gram m(a) - no force evaluations.
    """
    taus: tuple[float, ...]
    weights: tuple[np.ndarray, ...]
    energies: np.ndarray            # (ntau, 15, 15)
    metadata: dict

    def energy(self, amps) -> float:
        m = monomials(amps)
        return float(sum(m @ gram @ m for gram in self.energies))


def monomials(amps) -> np.ndarray:
    """Monomial vector in the exact order of the model fields:

        [1, a_i (4), a_i^2 (4), a_i a_j for i < j (6)]
    """
    a = np.asarray(amps, dtype=float)
    return np.concatenate(([1.0], a, a * a, [a[0] * a[1], a[0] * a[2],
                                             a[0] * a[3], a[1] * a[2],
                                             a[1] * a[3], a[2] * a[3]]))


def _weights_for(cfg: ReferenceConfig, taus) -> list[np.ndarray]:
    grid = make_grid(cfg.n)
    out = []
    for tau in taus:
        weight = annulus_weight(cfg, 1.0 - tau, grid)
        out.append(weight / max(float(np.sum(weight)), 1e-30))
    return out


def quadratic_model(cfg: ReferenceConfig, taus=(0.25, 0.4, 0.6)) -> QuadraticModel:
    """Extract the exact amplitude dependence of the annular residual energy.

    Evaluations: R(0), R(e_i), R(2 e_i), R(e_i + e_j) per tau - 15 force calls
    per tau.  With a_i the coefficients of `WAVE_BASIS`, the residual field is
    exactly quadratic in a, so these calls determine the objective for every
    amplitude in R^4 (up to floating point).
    """
    weights = _weights_for(cfg, taus)
    energies = []
    for tau, weight in zip(taus, weights):
        t = 1.0 - tau
        r0 = physical(ForceOracle(cfg, wave_amps=(0.0, 0.0, 0.0, 0.0)).force_hat(t))
        r1 = [physical(ForceOracle(cfg, wave_amps=unit).force_hat(t)) for unit in WAVE_BASIS]
        r2 = [physical(ForceOracle(cfg, wave_amps=tuple(2.0 * v for v in unit)).force_hat(t))
              for unit in WAVE_BASIS]
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        r12 = []
        for i, j in pairs:
            amps = list(WAVE_BASIS[i])
            amps = tuple(amps[k] + WAVE_BASIS[j][k] for k in range(4))
            r12.append(physical(ForceOracle(cfg, wave_amps=amps).force_hat(t)))

        # fields of the model: R0, c_i, G_ii, 2 G_ij (see class docstring)
        fields = [r0]
        g_ii = []
        for i in range(4):
            b_i = r1[i] - r0
            g_i = 0.5 * ((r2[i] - r0) - 2.0 * b_i)
            g_ii.append(g_i)
            fields.append(b_i - g_i)                      # c_i
        for i, j, r12f in zip([p[0] for p in pairs], [p[1] for p in pairs], r12):
            fields.append(r12f - r1[i] - r1[j] + r0)      # 2 G_ij
        fields.extend(g_ii)

        # monomial order must match nsblowup.reference.quadratic.monomials():
        # 1, a_i, a_i^2, a0a1, a0a2, a0a3, a1a2, a1a3, a2a3
        # fields holds [R0, c_i (4), 2G_ij (6), G_ii (4)] -> reorder to
        # [R0, c_i (4), G_ii (4), 2G_ij (6)]
        ordered = [fields[0]] + fields[1:5] + fields[11:15] + fields[5:11]
        stack = np.array(ordered).reshape(len(ordered), 3, -1)   # (15, 3, n^3)
        w = np.broadcast_to(weight.reshape(1, -1), (3, weight.size))
        gram = np.zeros((len(ordered), len(ordered)))
        for k in range(len(ordered)):
            for l in range(k, len(ordered)):
                value = float(np.sum(w * (stack[k] * stack[l])))
                gram[k, l] = gram[l, k] = value
        energies.append(gram)

    return QuadraticModel(taus=tuple(taus), weights=tuple(weights),
                          energies=np.array(energies),
                          metadata=metadata(cfg, taus=list(taus),
                                            metric="annulus-weighted L2 of residual"))


def optimize_amplitudes(model: QuadraticModel, amp_max: float = 1.0,
                        amp_min: float = 1e-6, steps: int = 15,
                        phases: int = 12) -> dict:
    """Optimize wave amplitudes on the quadratic residual model.

    Evaluates:
    * Local gradient g and Hessian H at a = 0 (Newton point and local Taylor drop).
    * Log-scale grid scan over amplitudes and free phases.
    * Best candidate amplitude vector, verified against true force evaluations.
    """
    energies = np.array(model.energies)
    e_zero = model.energy(np.zeros(4))
    g = np.zeros(4)
    for gram in energies:
        g += 2.0 * gram[0, 1:5]

    h = 1e-4
    hessian = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ei = np.zeros(4); ei[i] = h
            ej = np.zeros(4); ej[j] = h
            hessian[i, j] = (model.energy(ei + ej) - model.energy(ei - ej)
                             - model.energy(-ei + ej) + model.energy(-ei - ej)) / (4.0 * h * h)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.all(eigenvalues > 0.0):
        newton = -np.linalg.solve(hessian, g)
        newton_energy = model.energy(newton)
        # achievable energy drop near zero: 0.5 g^T H^{-1} g  (positive)
        bound_drop = -0.5 * float(g @ newton)
    else:
        newton = np.zeros(4)
        newton_energy = e_zero
        bound_drop = float("nan")

    scale = np.concatenate(([0.0], np.logspace(np.log10(amp_min), np.log10(amp_max), steps)))
    angles = np.linspace(0.0, 2.0 * np.pi, phases, endpoint=False)
    best = (e_zero, np.zeros(4))
    for amp_a in scale:
        for amp_b in scale:
            for phi_a in angles:
                for phi_b in angles:
                    amps = np.array([amp_a * np.cos(phi_a), amp_b * np.cos(phi_b),
                                     amp_a * np.sin(phi_a), amp_b * np.sin(phi_b)])
                    value = model.energy(amps)
                    if value < best[0]:
                        best = (value, amps)
    scan_energy, scan_amps = best
    if newton_energy < scan_energy:
        energy, amps, source = newton_energy, newton, "newton"
    else:
        energy, amps, source = scan_energy, scan_amps, "log-scan"

    return {"amps": amps, "energy": energy, "energy_zero": e_zero,
            "source": source, "gradient": g, "hessian": hessian,
            "hessian_eigenvalues": eigenvalues,
            "newton_amps": newton, "newton_energy": newton_energy,
            "scan_energy": scan_energy, "scan_amps": scan_amps,
            "relative_norm_reduction": 1.0 - np.sqrt(energy / max(e_zero, 1e-30)),
            "local_quadratic_predicted_reduction":
                (1.0 - np.sqrt((e_zero - bound_drop) / e_zero)
                 if np.isfinite(bound_drop) else float("nan"))}


def verify_model_random(cfg: ReferenceConfig, model: QuadraticModel, trials: int = 20,
                        seed: int = 20260910) -> dict[str, float]:
    """Regression test of the model extraction: random amplitudes, every tau.

    The optimizer's own check sits almost on top of a = 0 - one of the
    interpolation points - so it is a weak test by itself.  Here random
    directions are sampled with 1e-5 <= |a| <= 1 and the model is compared
    against fresh force evaluations at EVERY model tau.  Returns the worst
    relative error and the worst absolute pair encountered.
    """
    rng = np.random.default_rng(seed)
    worst_rel = 0.0
    worst_pair = (0.0, 0.0)
    for _ in range(trials):
        direction = rng.standard_normal(4)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-30)
        amps = direction * 10.0 ** rng.uniform(-5.0, 0.0)
        model_value = model.energy(amps)
        true_value = 0.0
        for tau, weight in zip(model.taus, model.weights):
            residual = physical(ForceOracle(cfg, wave_amps=tuple(amps)).force_hat(1.0 - tau))
            true_value += float(np.sum(weight * np.sum(residual * residual, axis=0)))
        rel = abs(model_value - true_value) / max(abs(true_value), 1e-30)
        if rel > worst_rel:
            worst_rel, worst_pair = rel, (model_value, true_value)
    return {"worst_relative_error": float(worst_rel),
            "worst_model": worst_pair[0], "worst_true": worst_pair[1],
            "trials": trials}


def verify_optimizer(cfg: ReferenceConfig, model: QuadraticModel, result: dict) -> dict:
    """Check the model optimum with true force evaluations (no model involved)."""
    amps = result["amps"]
    values = []
    for tau in model.taus:
        residual = physical(ForceOracle(cfg, wave_amps=tuple(amps)).force_hat(1.0 - tau))
        values.append(residual)
    weights = model.weights
    true_energy = float(sum(np.sum(w * np.sum(r * r, axis=0))
                            for w, r in zip(weights, values)))
    true_zero = 0.0
    for tau, w in zip(model.taus, weights):
        residual = physical(ForceOracle(cfg, wave_amps=(0.0, 0.0, 0.0, 0.0)).force_hat(1.0 - tau))
        true_zero += float(np.sum(w * np.sum(residual * residual, axis=0)))
    return {"model_energy": result["energy"], "true_energy": true_energy,
            "relative_model_error": abs(true_energy - result["energy"])
            / max(true_energy, 1e-30),
            "true_norm_ratio": np.sqrt(true_zero / max(true_energy, 1e-30)),
            "true_zero": true_zero}
