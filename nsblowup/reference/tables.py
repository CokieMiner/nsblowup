"""Forcing verification: checksums, tabulated force tables and interpolation checks.

Checksums are tracked in two scopes:
* `parent_force_data_fingerprint`: Canonical full-interval data hash of f_N* on
  the absolute reference grid, shared across DNS runs that restrict it.
* `restricted_force_fingerprint`: Hash of P_N f_N* as consumed by the DNS grid.

`freeze_force_table` writes f(t_j) with metadata and SHA-256 integrity verification.
`check_force_table_interpolation` measures cubic time-interpolation error against
the analytical force across startup and collapse windows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from nsblowup.core.config import ReferenceConfig
from nsblowup.core.frozen import FrozenForce, table_digest
from nsblowup.core.spectral import physical, restrict_hat
from nsblowup.reference.reference import ForceOracle
from nsblowup.reference.waves import code_hash, run_code_hash


PARENT_NODE_DTAU = 0.02   # canonical node spacing of the parent-force hash


def _canonical_nodes(q_floor: float, dtau: float = PARENT_NODE_DTAU) -> np.ndarray:
    """Canonical parent-force nodes: tau = 1, 1-dtau, ..., q_floor.

    The floor endpoint is appended explicitly when it is not a multiple of
    dtau, so the node set - and therefore the digest - always reaches exactly
    q_floor, independently of the spacing.
    """
    steps = int(np.floor((1.0 - q_floor) / dtau + 1e-9))
    taus = 1.0 - dtau * np.arange(steps + 1)
    if taus[-1] - q_floor > 1e-12:
        taus = np.append(taus, q_floor)
    return taus


def _parent_digest(dtau: float, q_floor: float, n_ref: int):
    """Digest seeded with the canonical node grid (data fingerprint header).

    The node array itself is hashed along with metadata to bind values and times.
    """
    nodes = _canonical_nodes(q_floor, dtau)
    digest = hashlib.sha256()
    digest.update(json.dumps({"kind": "canonical float32 data fingerprint of f_N*",
                              "node_dtau": float(dtau),
                              "nodes": int(nodes.size),
                              "n_ref": int(n_ref)},
                             sort_keys=True).encode())
    digest.update(nodes.astype(np.float64).tobytes())
    return digest


def parent_force_data_fingerprint(cfg: ReferenceConfig, dtau: float = PARENT_NODE_DTAU,
                                  verbose: bool = False) -> str:
    """Canonical full-interval fingerprint of the parent force f_N* (DATA only).

    Streams f_N* over the canonical nodes tau = 1, 1-dtau, ..., q_floor on the
    N* grid (float32 physical values) into SHA-256, together with N* and the
    node grid.  The numerical code identity is NOT part of this digest and is
    recorded separately (the `code_hash` field): this digest answers "is it
    exactly the same force?", while the code hash answers "was it produced by
    the same code?".  Two sampled times would not do: the start-up ramp
    saturates before t = smooth_start, so the late-time force can be identical
    while the early forcing (and hence the run) differs.

    Grid-relative mode has no single parent force across resolutions, so
    nothing is claimed there and the empty string is returned.
    """
    if not cfg.force_grid:
        return ""
    oracle = ForceOracle(cfg)
    digest = _parent_digest(dtau, cfg.q_floor, int(cfg.force_grid))
    taus = _canonical_nodes(cfg.q_floor, dtau)
    for index, tau in enumerate(taus):
        # snap t: 1 - j*dtau and 1 - k*dtau/n differ in the last float bits,
        # which would make the digest path-dependent.  The snap is 1e-12, far
        # below any physical time scale and far above float64 noise.
        field = physical(oracle.full_force_on_reference(round(1.0 - tau, 12)))
        digest.update(np.ascontiguousarray(field, dtype=np.float32).tobytes())
        if verbose and (index + 1) % 10 == 0:
            print(f"    parent force data hash: {index + 1}/{taus.size} snapshots",
                  flush=True)
    return digest.hexdigest()[:16]


def restricted_force_fingerprint(cfg: ReferenceConfig, taus=(0.3, 0.6)) -> str:
    """Hash of P_N f_N*(t) as the DNS consumes it (grid dependent by design)."""
    digest = hashlib.sha256()
    oracle = ForceOracle(cfg)
    for tau in taus:
        field = physical(oracle.force_hat(1.0 - tau))
        digest.update(np.ascontiguousarray(field, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def _uniform_nodes(tau_hi: float, tau_lo: float, dtau: float) -> np.ndarray:
    """tau = tau_hi, tau_hi - dtau, ..., tau_lo (both endpoints exact)."""
    count = int(round((tau_hi - tau_lo) / dtau))
    taus = tau_hi - dtau * np.arange(count + 1)
    taus[-1] = tau_lo
    return taus


def _freeze_nodes(q_floor: float, dtau: float, startup_dtau: float,
                  startup_window: float) -> np.ndarray:
    """Piecewise freeze grid: fine across the start-up ramp, coarse after it.

    The C-infinity ramp gives the force a time scale ~startup_window, and a
    uniform grid leaves its largest cubic-interpolation error there; refining
    ONLY the ramp (tau > 1 - startup_window) costs
    ~startup_window/startup_dtau extra samples instead of duplicating the
    whole table.
    """
    fine_count = int(round(max(startup_window, 0.0) / startup_dtau))
    fine = 1.0 - startup_dtau * np.arange(fine_count + 1)
    coarse = _uniform_nodes(float(fine[-1]), q_floor, dtau)
    return np.concatenate((fine, coarse[1:]))


def _table_values(cfg: ReferenceConfig, taus: np.ndarray,
                  progress: str = "", oracle: ForceOracle | None = None,
                  parent_digest=None) -> tuple[np.ndarray, int]:
    """Force snapshots at the given tau nodes (float32), node order given.

    When `parent_digest` is given (absolute grid mode), each node's FULL force
    is computed once: it is streamed into the digest at the canonical nodes and
    restricted for the stored table, so freezing does not pay for the parent
    fingerprint twice.  Returns (values, canonical_count).
    """
    oracle = ForceOracle(cfg) if oracle is None else oracle
    taus = np.asarray(taus, dtype=float)
    values = np.empty((taus.size, 3, cfg.n, cfg.n, cfg.n), dtype=np.float32)
    hashed = 0
    for index, tau in enumerate(taus):
        if parent_digest is not None and oracle.force_on_reference:
            full = oracle.full_force_on_reference(round(1.0 - tau, 12))
            strides = (1.0 - tau) / PARENT_NODE_DTAU
            if (abs(strides - round(strides)) < 1e-9
                    or abs(tau - cfg.q_floor) < 1e-12):
                field = physical(full)
                parent_digest.update(
                    np.ascontiguousarray(field, dtype=np.float32).tobytes())
                hashed += 1
            restricted = restrict_hat(full, cfg.n, oracle.n_ref) * oracle.mask
            values[index] = physical(restricted).astype(np.float32)
        else:
            values[index] = physical(oracle.force_hat(1.0 - tau)).astype(np.float32)
        if progress and (index + 1) % 25 == 0:
            print(f"    {progress}: {index + 1}/{taus.size} snapshots", flush=True)
    return values, hashed


def freeze_force_table(cfg: ReferenceConfig, path: str, dtau: float = 5.0e-3,
                       startup_dtau: float = 1.0e-3,
                       startup_window: float | None = None,
                       verbose: bool = True) -> dict:
    """Write f(t_j), restricted to the DNS grid, as a standalone table.

    Requires an absolute force_grid: only then is the table the restriction of
    one forcing, and only then can a solver run consume it without touching the
    construction.  The table records the parent hash (shared by all DNS
    resolutions of the study) and the restricted hash (what the DNS sees).

    Time grid: piecewise - `startup_dtau` through the C-infinity start-up ramp
    (0 < t < startup_window, default smooth_start), `dtau` elsewhere.  The
    ramp is the only place a uniform 5e-3 grid leaves a large interpolation
    error (measured ~1e-2 of the local force scale, cubic), and refining only
    the ramp adds ~window/startup_dtau samples instead of duplicating the
    table.
    """
    if not cfg.force_grid:
        raise SystemExit("--freeze-force requires --force-grid N*: in the "
                         "n x factor mode the force depends on the DNS grid.")
    if int(cfg.force_grid) == cfg.n:
        print("  note: N* equals the DNS grid; the restriction is the identity "
              "and the forcing is not independent of the resolution")
    if startup_window is None:
        startup_window = cfg.smooth_start
    startup_window = max(float(startup_window), 0.0)
    taus = _freeze_nodes(cfg.q_floor, dtau, startup_dtau, startup_window)

    # one force evaluation per node serves both the stored table and the
    # parent data hash (at the canonical nodes), when the grid lands exactly
    # on the canonical nodes; otherwise the canonical nodes are hashed alone
    canonical = _canonical_nodes(cfg.q_floor)
    digest = _parent_digest(PARENT_NODE_DTAU, cfg.q_floor, int(cfg.force_grid))
    oracle = ForceOracle(cfg)
    values, hashed = _table_values(
        cfg, taus, progress="frozen force" if verbose else "", oracle=oracle,
        parent_digest=digest)
    if hashed != canonical.size:
        # rare: the table grid does not produce exactly the canonical nodes
        digest = _parent_digest(PARENT_NODE_DTAU, cfg.q_floor,
                                int(cfg.force_grid))
        for tau in canonical:
            field = physical(oracle.full_force_on_reference(round(1.0 - tau, 12)))
            digest.update(np.ascontiguousarray(field, dtype=np.float32).tobytes())
    parent_hash = digest.hexdigest()[:16]

    meta = {
        "n": cfg.n,
        "n_ref": int(cfg.force_grid),
        "h": cfg.h,
        "q_floor": cfg.q_floor,
        "viscosity": cfg.viscosity,
        "dealias": cfg.dealias,
        "wave_coeffs": list(cfg.wave_coeffs),
        "reference_config": json.dumps(asdict(cfg)),
        "dtau": float(dtau),
        "dtau_startup": float(startup_dtau),
        "startup_window": float(startup_window),
        "time_grid": ("piecewise: dtau_startup for tau > 1 - startup_window, "
                      "dtau elsewhere"),
        "samples": int(taus.size),
        "tau_first": float(taus[0]),
        "tau_last": float(taus[-1]),
        "fingerprint": restricted_force_fingerprint(cfg),
        "parent_force_data_hash": parent_hash,
        "code_hash": code_hash(),
        "run_code_hash": run_code_hash(),
        "table_sha256": "",                     # filled below (self-excluded)
        "metric": "f_N*(t) restricted to the DNS grid, physical, float32",
    }
    meta["table_sha256"] = table_digest(1.0 - taus, values, meta)
    np.savez_compressed(path, times=1.0 - taus, values=values,
                        meta=json.dumps(meta))
    print(f"  time grid: dtau {dtau:g} · dtau_startup {startup_dtau:g} for "
          f"t < {startup_window:g}")
    print(f"  wrote {path}: {taus.size} samples, {values.nbytes / 1e6:.0f} MB, "
          f"sha256 {meta['table_sha256'][:12]}")
    return meta


def check_force_table_interpolation(cfg: ReferenceConfig,
                                    dtaus: tuple[float, ...] = (0.01, 0.005, 0.0025),
                                    samples: int = 12, seed: int = 11,
                                    startup_dtau: float = 1.0e-3,
                                    dtau: float = 5.0e-3,
                                    startup_window: float | None = None
                                    ) -> list[dict]:
    """Interpolation error of frozen grids against the oracle, per window.

    The force varies rapidly during the start-up ramp AND near the collapse,
    so three windows are tested: ramp (tau = 1 - smooth_start .. 1.0), mid
    collapse (0.50 .. 1 - smooth_start) and late collapse (q_floor .. 0.50) -
    together they cover the whole table interval with randomized sample times
    (a shared set per window, drawn off every tested grid, otherwise the
    errors would be compared over different time sets).  The ramp window also
    tests the SHIPPED piecewise grid (fine dtau_startup inside the ramp,
    coarse dtau elsewhere).  Errors are normalised by the largest force norm
    in the window; cubic interpolation should show a ~dtau^4 trend while the
    errors sit above the float32 storage noise.
    """
    if cfg.force_grid is None:
        raise SystemExit("--check-interpolation requires --force-grid N*")
    if startup_window is None:
        startup_window = cfg.smooth_start
    oracle = ForceOracle(cfg)
    break_tau = 1.0 - startup_window
    ramp_dtaus = (dtau, 0.5 * dtau, startup_dtau)
    windows = (("startup ramp", (break_tau, 1.0), ramp_dtaus),
               ("mid collapse", (0.50, break_tau), dtaus),
               ("late collapse", (cfg.q_floor, 0.50), dtaus))
    piecewise = _freeze_nodes(cfg.q_floor, dtau, startup_dtau, startup_window)

    def measure(taus: np.ndarray, times: list[float], label: str) -> dict:
        values, _ = _table_values(cfg, taus, oracle=oracle)
        table = FrozenForce(times=1.0 - taus, values=values,
                            meta={"n": cfg.n, "n_ref": int(cfg.force_grid)})
        references = [physical(oracle.force_hat(t)) for t in times]
        scale = max(float(np.linalg.norm(r)) for r in references)
        errors = [float(np.linalg.norm(physical(table.force_hat(t)) - r)) / scale
                  for t, r in zip(times, references)]
        return {"grid": label, "nodes": int(taus.size),
                "max_err": max(errors), "mean_err": float(np.mean(errors))}

    results: list[dict] = []
    for name, window, tested in windows:
        if not window[0] < window[1]:
            continue                       # e.g. q_floor above the mid window
        rng = np.random.default_rng(seed)
        t_lo, t_hi = 1.0 - window[1], 1.0 - window[0]   # t = 1 - tau, ascending
        node_times = {d: 1.0 - _uniform_nodes(window[1], window[0], d)
                      for d in tested}
        times = []
        while len(times) < samples:
            t = rng.uniform(t_lo, t_hi)
            if not all(np.min(np.abs(node_times[d] - t)) > 0.2 * d
                       for d in tested):
                continue                       # too close to a node: no test
            times.append(t)
        rows: list[dict] = []
        previous = None
        for d in sorted(tested, reverse=True):
            row = measure(_uniform_nodes(window[1], window[0], d), times,
                          f"uniform dtau={d:g}")
            if previous is not None:
                row["order"] = float(
                    np.log(previous["max_err"] / max(row["max_err"], 1e-30))
                    / np.log(previous["dtau"] / d))
            row["dtau"] = float(d)
            previous = row
            rows.append(row)
        if name == "startup ramp":
            rows.append(measure(piecewise, times, "piecewise (as shipped)"))
        results.append({"window": name,
                        "tau_range": (float(window[0]), float(window[1])),
                        "rows": rows})
    return results

