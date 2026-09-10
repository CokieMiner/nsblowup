"""Tabulated forcing: loader and time interpolation for DNS runs.

`python cli.py reference --force-grid N* --freeze-force TABLE.npz` writes
f(t_j) built on an absolute reference grid N* and restricted to the DNS grid.
This module reads that table and interpolates in time.

This module maintains strict decoupling from the analytical construction:
the DNS solver interacts with the forcing purely as an external field.

File format (npz, written by `freeze_force_table`):

    times   (nt,)             float64, increasing
    values  (nt, 3, n, n, n)  float32, physical-space force snapshots
    meta    json string       metadata: n, n_ref, h, q_floor, viscosity,
                              dealias, wave_coeffs, checksums and sha256
                              of times + values + metadata (checked on load)

Interpolation: cubic (4-point Lagrange) in time. The piecewise time grid
refines spacing across the startup ramp (t < smooth_start) to maintain
high interpolation accuracy throughout the evolution.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def table_digest(times: np.ndarray, values: np.ndarray, meta: dict) -> str:
    """sha256 over the time nodes, the values and the canonical metadata.

    The nodes are covered as well: changing `times` alone would otherwise pass
    a values-only check while changing the function f(t) the solver sees.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(times, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    canonical = {key: value for key, value in meta.items()
                 if key not in ("table_sha256", "values_sha256")}
    digest.update(json.dumps(canonical, sort_keys=True,
                             separators=(",", ":")).encode())
    return digest.hexdigest()


class FrozenForce:
    """Time-interpolated external forcing.

    Normally loaded from a table written by the reference command; the same
    object can also be built directly from arrays (used by the interpolation
    cross-check in `nsblowup.reference.tables`).
    """

    def __init__(self, path: str | None = None, *, times=None, values=None,
                 meta: dict | None = None):
        if path is not None:
            with np.load(path, allow_pickle=False) as data:
                try:
                    times = np.asarray(data["times"], dtype=float)
                    values = np.asarray(data["values"], dtype=np.float32)
                    meta = json.loads(str(data["meta"]))
                except KeyError:
                    raise ValueError(
                        f"{path} is not a frozen force table "
                        f"(expected arrays 'times', 'values', 'meta')") from None
        elif times is None or values is None or meta is None:
            raise ValueError("FrozenForce needs a path, or times/values/meta")
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=np.float32)
        self.meta = dict(meta)
        if self.times.ndim != 1 or self.times.size < 2:
            raise ValueError("force table needs a 1-D time grid with >= 2 nodes")
        if not np.all(np.isfinite(self.times)) or not np.all(np.diff(self.times) > 0.0):
            raise ValueError("force table times must be finite and strictly increasing")
        if self.values.ndim != 5:
            raise ValueError("force table values must be (nt, 3, n, n, n)")
        if self.values.shape[0] != self.times.size:
            raise ValueError("force table times and values have different lengths")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("force table values contain non-finite entries")
        digest = table_digest(self.times, self.values, self.meta)
        stored = self.meta.get("table_sha256")
        if stored is not None and digest != stored:
            raise ValueError(
                f"force table {path or '<in-memory>'} failed its hash check"
                f" (stored {stored[:12]}, computed {digest[:12]}): "
                "the file was edited or corrupted")
        legacy = self.meta.get("values_sha256")
        if stored is None and legacy is not None:      # values-only legacy files
            legacy_digest = hashlib.sha256(
                np.ascontiguousarray(self.values).tobytes()).hexdigest()
            if legacy_digest != legacy:
                raise ValueError(f"force table {path or '<in-memory>'} failed its "
                                 f"legacy values hash check")
        self.sha256 = digest
        self.n = int(self.meta["n"])
        if self.values.shape[1:] != (3, self.n, self.n, self.n):
            raise ValueError("force table shape does not match its metadata")
        self._spectra: dict[int, np.ndarray] = {}

    # -- lookup ------------------------------------------------------------
    def _spectrum(self, index: int) -> np.ndarray:
        cached = self._spectra.get(index)
        if cached is None:
            cached = np.fft.fftn(self.values[index].astype(np.float64), axes=(1, 2, 3))
            if len(self._spectra) > 4:
                self._spectra.clear()
            self._spectra[index] = cached
        return cached

    def force_hat(self, t: float) -> np.ndarray:
        """f_hat(t) by cubic (4-point Lagrange) interpolation of the snapshots."""
        times = self.times
        if times.size < 4:                       # degenerate tables: linear
            t = min(max(t, float(times[0])), float(times[-1]))
            i = min(max(int(np.searchsorted(times, t, side="right") - 1), 0), times.size - 2)
            t0, t1 = float(times[i]), float(times[i + 1])
            w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return (1.0 - w) * self._spectrum(i) + w * self._spectrum(i + 1)
        t = min(max(t, float(times[0])), float(times[-1]))
        i = int(np.searchsorted(times, t, side="right") - 1)
        start = min(max(i - 1, 0), times.size - 4)
        idx = [start, start + 1, start + 2, start + 3]
        xs = times[idx]
        weights = _lagrange_weights(t, xs)
        out = weights[0] * self._spectrum(idx[0])
        for weight, index in zip(weights[1:], idx[1:]):
            out = out + weight * self._spectrum(index)
        return out

    def describe(self) -> str:
        m = self.meta
        return (f"frozen table {m.get('samples', '?')} samples · n={self.n} · "
                f"N*={m.get('n_ref', '?')} · dtau={m.get('dtau', '?')} · "
                f"sha256={self.sha256[:12]}")


def _lagrange_weights(x: float, xs: np.ndarray) -> np.ndarray:
    """Lagrange basis weights for nodes xs evaluated at x (4 points here)."""
    n = xs.size
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (x - xs[j]) / (xs[i] - xs[j])
    return weights
