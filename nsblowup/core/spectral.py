"""Spectral operators shared by the reference construction and the DNS solver.

Both sides must use *identical* discrete operators, otherwise the frozen force
is not the discrete residual of the reference field and the DNS cannot converge
to the reference solution for purely numerical reasons.  The nonlinearity is
carried in rotational form (curl u) x u, equal to (u.grad)u after the Leray
projection; `convective_product` provides the literal advective form for
diagnostics.  Two dealiasing modes are provided:

* ``strict23``: keep |m| <= floor((n-1)/3) (classic 2/3 rule, strict edge)
* ``pad32``:   compute quadratic products on a 3n/2 grid (3/2 rule) and
               truncate back to the same retained band.  ~3x more expensive,
               meant for high-accuracy validation runs.

Independence of the reference force is assessed separately, by recomputing the
residual with fourth-order finite differences (see
`nsblowup.reference.report.advection_fd_check`).
"""

from __future__ import annotations

import numpy as np


def wavenumbers(n: int):
    """Return (kx, ky, kz, k2, (mx, my, mz)) with 3-D integer mode grids."""
    m = np.fft.fftfreq(n, d=1.0 / n)
    mx, my, mz = np.meshgrid(m, m, m, indexing="ij")
    kx = 2.0 * np.pi * mx
    ky = 2.0 * np.pi * my
    kz = 2.0 * np.pi * mz
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2, (mx, my, mz)


def retained_mask(n: int, mode: str = "strict23") -> np.ndarray:
    """True 3-D 2/3-rule mask: |m_x|, |m_y|, |m_z| <= floor((n-1)/3).

    The mask must be a 3-D array: a 1-D mode vector broadcasts against
    (3, n, n, n) and truncates only the last axis, i.e. dealiases a single
    direction while leaving the other two untouched.
    """
    _, _, _, _, (mx, my, mz) = wavenumbers(n)
    cut = (n - 1) // 3
    mask = (np.abs(mx) <= cut) & (np.abs(my) <= cut) & (np.abs(mz) <= cut)
    return mask


def physical(field_hat: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifftn(field_hat, axes=(1, 2, 3)))


def curl_hat(a_hat: np.ndarray, k) -> np.ndarray:
    kx, ky, kz, _, _ = k
    out = np.empty_like(a_hat)
    out[0] = 1j * (ky * a_hat[2] - kz * a_hat[1])
    out[1] = 1j * (kz * a_hat[0] - kx * a_hat[2])
    out[2] = 1j * (kx * a_hat[1] - ky * a_hat[0])
    return out


def project(u_hat: np.ndarray, k) -> np.ndarray:
    """Leray projector: remove the compressible part, zero the k = 0 mode."""
    kx, ky, kz, k2, _ = k
    div = kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]
    out = u_hat.copy()
    nz = k2 != 0.0
    out[0][nz] -= kx[nz] * div[nz] / k2[nz]
    out[1][nz] -= ky[nz] * div[nz] / k2[nz]
    out[2][nz] -= kz[nz] * div[nz] / k2[nz]
    out[:, 0, 0, 0] = 0.0
    return out


def _embed(u_hat: np.ndarray, n: int, n_pad: int) -> np.ndarray:
    """Zero-pad a spectral field from an n-grid to an n_pad-grid.

    numpy's ifft divides by the number of grid points, so coefficients copied
    to a larger grid reconstruct a field smaller by (n/n_pad)^3.  The factor
    below cancels that: without it the padded nonlinearity comes out ~0.296x
    in 3-D, which is exactly the (2/3)^3 artefact seen in the pad32 mode.
    """
    half = n // 2
    padded = np.zeros((3, n_pad, n_pad, n_pad), dtype=np.complex128)
    for bx, sx in ((slice(0, half), slice(0, half)),
                   (slice(n_pad - half, n_pad), slice(n - half, n))):
        for by, sy in ((slice(0, half), slice(0, half)),
                       (slice(n_pad - half, n_pad), slice(n - half, n))):
            for bz, sz in ((slice(0, half), slice(0, half)),
                           (slice(n_pad - half, n_pad), slice(n - half, n))):
                padded[:, bx, by, bz] = u_hat[:, sx, sy, sz]
    padded *= (n_pad / n) ** 3
    return padded


def _truncate(u_hat_pad: np.ndarray, n: int, n_pad: int) -> np.ndarray:
    """Restrict a spectral field from an n_pad-grid back to the n-grid.

    The fft on the n_pad grid sums over n_pad points, so the coefficients must
    be rescaled by n/n_pad in each dimension (see `_embed`).
    """
    half = n // 2
    out = np.empty((3, n, n, n), dtype=np.complex128)
    for bx, sx in ((slice(0, half), slice(0, half)),
                   (slice(n_pad - half, n_pad), slice(n - half, n))):
        for by, sy in ((slice(0, half), slice(0, half)),
                       (slice(n_pad - half, n_pad), slice(n - half, n))):
            for bz, sz in ((slice(0, half), slice(0, half)),
                           (slice(n_pad - half, n_pad), slice(n - half, n))):
                out[:, sx, sy, sz] = u_hat_pad[:, bx, by, bz]
    out *= (n / n_pad) ** 3
    return out


def rotational_product(u_hat: np.ndarray, k, mode: str = "strict23"):
    """Return FFT[(curl u) x u] (rotational form) and the physical velocity.

    This is the form the time loop integrates: for divergence-free fields
    P[(u.grad)u] = P[(curl u) x u], so projecting the rotational product and
    projecting the advective product give the same vector.  Diagnostics that
    need the literal (u.grad)u use `convective_product` below.
    """
    n = u_hat.shape[1]
    kx, ky, kz, _, _ = k
    if mode == "pad32":
        n_pad = 3 * n // 2
        u_pad = _embed(u_hat, n, n_pad)
        k_pad = wavenumbers(n_pad)
        u_phys = physical(u_pad)
        curl = physical(curl_hat(u_pad, k_pad))
        adv = np.cross(curl, u_phys, axisa=0, axisb=0, axisc=0)
        adv_pad = np.fft.fftn(adv, axes=(1, 2, 3))
        adv_hat = _truncate(adv_pad, n, n_pad)
    else:
        u_phys = physical(u_hat)
        curl = physical(curl_hat(u_hat, k))
        adv = np.cross(curl, u_phys, axisa=0, axisb=0, axisc=0)
        adv_hat = np.fft.fftn(adv, axes=(1, 2, 3))
    return adv_hat, u_phys


def convective_product(u_hat: np.ndarray, k) -> tuple[np.ndarray, np.ndarray]:
    """True (u.grad)u in spectral space, unprojected (diagnostics only).

    Components: [(u.grad)u]_i = sum_j u_j d_j u_i, with pointwise products in
    physical space and spectral derivatives.  The identity

        (u.grad)u = (curl u) x u + grad(|u|^2 / 2)

    is exact here, so P[convective] = P[rotational] to machine precision and the
    difference is a pure gradient (absorbed into the pressure).
    """
    u = physical(u_hat)
    out = np.zeros_like(u)
    for j in range(3):
        d_j = physical(1j * k[j] * u_hat)          # d_j u_i for every component
        out += u[j] * d_j
    return np.fft.fftn(out, axes=(1, 2, 3)), u


def resample_origin_phase(n_in: int, n_out: int) -> np.ndarray:
    """Origin phase correction for cell-centred spectral resampling.

    Cell centres sit at x_j = (j + 1/2)/N - 1/2, so grids of different sizes
    have offset origins by delta = 1/(2 n_in) - 1/(2 n_out) per axis.
    Multiplying by exp(-2 pi i m delta) aligns the physical origin across grids.
    """
    freq = np.fft.fftfreq(n_in, d=1.0 / n_in)
    delta = 0.5 / n_in - 0.5 / n_out
    mx, my, mz = np.meshgrid(freq, freq, freq, indexing="ij")
    return np.exp(-2j * np.pi * delta * (mx + my + mz))


def restrict_hat(u_hat_fine: np.ndarray, n_coarse: int, n_fine: int) -> np.ndarray:
    """Spectrally restrict a band-limited field from n_fine to n_coarse.

    Includes the origin-phase compensation for cell-centred coordinates
    and DFT coefficient volume rescaling.
    """
    if n_coarse == n_fine:
        return u_hat_fine
    return _truncate(resample_origin_phase(n_fine, n_coarse) * u_hat_fine,
                     n_coarse, n_fine)


def spectral_tail_fraction(u_hat: np.ndarray, n: int, fraction: float = 0.8) -> float:
    """Energy fraction in the top of the *retained band* (resolution monitor).

    Measuring against 0.8 * max|k| (the Nyquist corner) is meaningless once the
    2/3 mask is applied correctly: retained modes never reach that corner, so
    the monitor would read ~0 while the highest retained modes fill up.  Here
    the shell is defined by max(|m_x|, |m_y|, |m_z|) > fraction * cut.
    """
    _, _, _, _, (mx, my, mz) = wavenumbers(n)
    cut = (n - 1) // 3
    shell = (np.maximum(np.maximum(np.abs(mx), np.abs(my)), np.abs(mz))
             > fraction * cut)
    energy = np.sum(np.abs(u_hat) ** 2, axis=0)
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0
    return float(np.sum(energy[shell]) / total)
