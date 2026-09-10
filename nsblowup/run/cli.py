"""`dns` command: run the solver and write the snapshot archive and movie.

The forcing is either a frozen table (the solver then never touches the
construction) or the reference oracle built here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np

from nsblowup.core.config import QUICK, DNSConfig, ReferenceConfig
from nsblowup.core.frozen import FrozenForce
from nsblowup.core.spectral import physical
from nsblowup.reference.tables import (parent_force_data_fingerprint,
                                       restricted_force_fingerprint)
from nsblowup.reference.reference import ForceOracle
from nsblowup.reference.waves import code_hash, run_code_hash
from nsblowup.run.solver import Result, simulate
from nsblowup.run.video import build_video, measure_exponents, measure_free_decay


def parse_args(argv: list[str] | None = None):
    dns_defaults = DNSConfig()
    ref_defaults = ReferenceConfig()
    parser = argparse.ArgumentParser(
        prog="nsblowup dns", description="forced DNS of the reference construction")
    parser.add_argument("--quick", action="store_true", help="fast smoke test preset")
    parser.add_argument("--n", type=int, default=dns_defaults.n)
    parser.add_argument("--dealias", choices=("strict23", "pad32"), default=None)
    parser.add_argument("--viscosity", type=float, default=None,
                        help="PDE viscosity (must match the force table, if given)")
    parser.add_argument("--t-end", type=float, default=dns_defaults.t_end)
    parser.add_argument("--frames", type=int, default=dns_defaults.frames,
                        help="number of simulated snapshots")
    parser.add_argument("--fps", type=int, default=dns_defaults.fps)
    parser.add_argument("--hold-frames", type=int, default=dns_defaults.hold_frames)
    parser.add_argument("--cfl", type=float, default=dns_defaults.cfl)
    parser.add_argument("--max-dt", type=float, default=dns_defaults.max_dt,
                        help="upper bound on the time step")
    parser.add_argument("--tail-tolerance", type=float, default=dns_defaults.tail_tolerance)
    parser.add_argument("--tail-arm", type=float, default=None,
                        help="time after which the spectral-tail stop is enforced "
                             "(default: end of the start-up ramp)")
    parser.add_argument("--store-dtype", choices=("float16", "float32"),
                        default=dns_defaults.store_dtype)
    parser.add_argument("--store-swirl", action="store_true",
                        help="store u_theta snapshots (profile-collapse analysis)")
    parser.add_argument("--particles", type=int, default=dns_defaults.particles)
    parser.add_argument("--image", type=int, default=dns_defaults.image)
    parser.add_argument("--samples", type=int, default=dns_defaults.samples)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--q-floor", type=float, default=None)
    parser.add_argument("--amp1", type=float, default=None)
    parser.add_argument("--amp2", type=float, default=None)
    parser.add_argument("--amp3", type=float, default=None)
    parser.add_argument("--amp4", type=float, default=None)
    parser.add_argument("--reference-factor", type=int, default=ref_defaults.reference_factor,
                        help="construction grid = n x factor (force then depends on n)")
    parser.add_argument("--force-grid", type=int, default=None,
                        help="absolute N*: build the whole force there, then restrict")
    parser.add_argument("--allow-large", action="store_true",
                        help="override the memory guard")
    parser.add_argument("--calibration", default=None, metavar="PATH",
                        help="wave coefficients from the reference command; "
                             "their metadata is checked against this run")
    parser.add_argument("--force-table", default=None, metavar="PATH",
                        help="frozen force table; the construction is then not evaluated")
    parser.add_argument("--init-from-reference", type=float, default=None, metavar="T0",
                        help="initialize with u*(T0) instead of rest")
    parser.add_argument("--no-force", action="store_true",
                        help="zero forcing: free evolution (needs --init-from-reference)")
    parser.add_argument("--output", default=dns_defaults.output)
    parser.add_argument("--archive", default="openai_dns.npz")
    args = parser.parse_args(argv)

    viscosity = args.viscosity if args.viscosity is not None else dns_defaults.viscosity
    dealias = args.dealias or dns_defaults.dealias
    h = args.h if args.h is not None else ref_defaults.h
    q_floor = args.q_floor if args.q_floor is not None else ref_defaults.q_floor
    provided = (args.amp1, args.amp2, args.amp3, args.amp4)
    explicit_amps = any(value is not None for value in provided)
    if explicit_amps:
        coeffs = tuple(value if value is not None else ref_defaults.wave_coeffs[i]
                       for i, value in enumerate(provided))
    else:
        coeffs = ref_defaults.wave_coeffs

    dns = replace(dns_defaults, n=args.n, dealias=dealias, viscosity=viscosity,
                  t_end=args.t_end, frames=args.frames, fps=args.fps,
                  hold_frames=args.hold_frames, cfl=args.cfl, max_dt=args.max_dt,
                  tail_tolerance=args.tail_tolerance, store_dtype=args.store_dtype,
                  store_swirl=args.store_swirl,
                  free_evolution=args.no_force,
                  release_time=(args.init_from_reference or 0.0),
                  particles=args.particles,
                  image=args.image, samples=args.samples,
                  output=args.output)
    ref = replace(ref_defaults, h=h, q_floor=q_floor,
                  viscosity=viscosity, n=args.n, dealias=dealias,
                  reference_factor=args.reference_factor,
                  force_grid=args.force_grid,
                  allow_large=args.allow_large,
                  wave_coeffs=coeffs)

    if args.quick:
        dns = replace(dns, **QUICK)
        ref = replace(ref, n=dns.n, q_floor=0.20)
        dns = replace(dns, t_end=min(dns.t_end, 1.0 - ref.q_floor))
    if dns.n % 2 or dns.n < 8:
        raise SystemExit("--n must be an even integer >= 8")
    try:
        ref.validate()
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from None
    if dns.frames < 1 or dns.frames > 200000:
        raise SystemExit("--frames must lie in [1, 200000]")
    if dns.t_end <= 0.0:
        raise SystemExit("--t-end must be positive")
    if dns.cfl <= 0.0 or dns.max_dt <= 0.0:
        raise SystemExit("--cfl and --max-dt must be positive")
    if dns.viscosity < 0.0:
        raise SystemExit("--viscosity must be non-negative")
    if not 0.0 < ref.q_floor < 1.0:
        raise SystemExit("--q-floor must lie in (0, 1)")
    if args.tail_arm is not None and args.tail_arm < 0.0:
        raise SystemExit("--tail-arm must be non-negative")
    if dns.release_time and dns.release_time >= dns.t_end:
        raise SystemExit("--init-from-reference T0 must be < --t-end")
    if args.init_from_reference is not None:
        # the construction domain ends at the collapse clock floor; releasing
        # from u*(T0) with T0 beyond it (or beyond t = 1) is not defined
        boundary = 1.0 - ref.q_floor
        if not 0.0 <= args.init_from_reference <= boundary + 1e-12:
            raise SystemExit(
                f"--init-from-reference T0 must lie in [0, {boundary:g}]: "
                "the construction domain ends at t = 1 - q_floor")
    return dns, ref, args


def _table_conflicts(args, meta: dict) -> list[str]:
    """Parameters the table fixes, compared against explicitly given flags."""
    conflicts = []
    for name, given in (("viscosity", args.viscosity), ("dealias", args.dealias),
                        ("h", args.h), ("q_floor", args.q_floor)):
        expected = meta.get(name)
        if given is not None and expected is not None and given != expected:
            conflicts.append(f"{name}: command line {given!r} vs table {expected!r}")
    given_amps = (args.amp1, args.amp2, args.amp3, args.amp4)
    if any(value is not None for value in given_amps) and "wave_coeffs" in meta:
        given = tuple(value if value is not None else 0.0 for value in given_amps)
        if given != tuple(meta["wave_coeffs"]):
            conflicts.append(f"wave_coeffs: command line {given} vs table "
                             f"{tuple(meta['wave_coeffs'])}")
    return conflicts


def _close(a, b) -> bool:
    """Tolerant equality for metadata numbers (JSON round trip)."""
    try:
        return abs(float(a) - float(b)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _load_calibration(path: str, ref: ReferenceConfig, dns: DNSConfig
                      ) -> tuple[float, ...]:
    """Wave coefficients whose recorded metadata matches this run.

    The file stores the code hash and the physics configuration of the fit;
    coefficients produced by a different configuration would change
    the experiment, so any mismatch aborts instead of loading.
    """
    with open(path) as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata", {})
    mismatches = []
    for name, stored, current, numeric in (
            ("code_hash", metadata.get("code_hash"), code_hash(), False),
            ("n", metadata.get("n"), dns.n, True),
            ("h", metadata.get("h"), ref.h, True),
            ("q_floor", metadata.get("q_floor"), ref.q_floor, True),
            ("viscosity", metadata.get("viscosity"), dns.viscosity, True),
            ("dealias", metadata.get("dealias"), dns.dealias, False),
            # grid-relative and absolute-force-grid runs optimise different
            # residual objectives (nonlinearity formed before/after
            # restriction), so their calibrations must not be interchangeable
            ("reference_factor", metadata.get("reference_factor"),
             ref.reference_factor, True),
            ("reference_grid", metadata.get("reference_grid"),
             ref.reference_grid_size(), True),
            ("force_grid", metadata.get("force_grid"), ref.force_grid, False)):
        same = _close(stored, current) if numeric else str(stored) == str(current)
        if not same:
            mismatches.append(f"{name}: file {stored!r} vs run {current!r}")
    if mismatches:
        raise SystemExit(
            f"calibration file {path} does not match this run:\n  "
            + "\n  ".join(mismatches)
            + "\nRerun `python cli.py reference --calibrate --save-calibration` "
              "for this configuration, or omit --calibration.")
    coeffs = tuple(float(value) for value in payload["wave_coeffs"])
    print(f"  loaded wave calibration from {path}: {coeffs} "
          f"(code_hash {metadata.get('code_hash')})")
    return coeffs


def _write_archive(path: str, res: Result, dns: DNSConfig, ref: ReferenceConfig,
                   fingerprint: str, parent_hash: str, table_hash: str,
                   initial_state_hash: str = "") -> None:
    arrays = dict(
        times=res.times, speeds=res.speeds, tracers=res.tracers,
        beyond_resolution=res.beyond_resolution,
        analytic_reference=res.analytic_reference,
        max_speed=res.max_speed, max_swirl=res.max_swirl,
        max_radial=res.max_radial, length_r=res.length_r, length_z=res.length_z,
        energy=res.energy, tail=res.tail, reynolds_swirl=res.reynolds_swirl,
        tracking=res.tracking, stop_index=res.stop_index, div_error=res.div_error,
        vmax=np.float64(res.vmax),
        h=ref.h, q_floor=ref.q_floor, viscosity=dns.viscosity, n=dns.n,
        wave_coeffs=np.array(ref.wave_coeffs), dealias=dns.dealias,
        similarity_scale=ref.similarity_scale, j0=ref.j0,
        force_grid=int(ref.force_grid or 0),
        force_fingerprint=fingerprint, parent_force_hash=parent_hash,
        force_table_hash=table_hash, initial_state_hash=initial_state_hash,
        code_hash=code_hash(), run_code_hash=run_code_hash(),
        tail_arm=np.float64(res.tail_arm),
        tail_prearm_max=np.float64(res.tail_prearm_max),
        reference_config=json.dumps(asdict(ref)),
        dns_config=json.dumps(asdict(dns)),
    )
    if res.swirl is not None:
        arrays["swirl"] = res.swirl
    np.savez_compressed(path, **arrays)
    print(f"  wrote {path}")


def main(argv: list[str] | None = None) -> None:
    dns_cfg, ref_cfg, args = parse_args(argv)
    if args.no_force and args.init_from_reference is None:
        raise SystemExit("--no-force requires --init-from-reference T0")
    if args.init_from_reference is not None and args.force_table:
        raise SystemExit("--init-from-reference works with the reference oracle, "
                         "not with a frozen force table")

    table = None
    initial_hat = None
    initial_state_hash = ""
    if args.force_table:
        try:
            table = FrozenForce(args.force_table)
        except ValueError as exc:
            raise SystemExit(f"cannot load force table: {exc}") from None
        if table.n != dns_cfg.n:
            raise SystemExit(f"force table n={table.n} does not match DNS n={dns_cfg.n}")
        conflicts = _table_conflicts(args, table.meta)
        if conflicts:
            raise SystemExit("the force table is the specification of the run; "
                             "it disagrees with the command line:\n  "
                             + "\n  ".join(conflicts))
        meta = table.meta
        stored = meta.get("reference_config")
        if stored:
            # the table carries the complete construction config; restore ALL
            # of it, not just the subset the solver happens to need
            ref_cfg = ReferenceConfig(**json.loads(str(stored)))
            ref_cfg = replace(ref_cfg, wave_coeffs=tuple(ref_cfg.wave_coeffs))
        else:                                    # legacy subset
            ref_cfg = replace(ref_cfg, h=meta.get("h", ref_cfg.h),
                              q_floor=meta.get("q_floor", ref_cfg.q_floor),
                              viscosity=meta.get("viscosity", ref_cfg.viscosity),
                              dealias=meta.get("dealias", ref_cfg.dealias),
                              force_grid=meta.get("n_ref", ref_cfg.force_grid),
                              wave_coeffs=tuple(meta.get("wave_coeffs",
                                                         ref_cfg.wave_coeffs)))
        # the table is the specification: it also fixes the solver parameters,
        # not only the construction ones
        if "viscosity" in meta:
            dns_cfg = replace(dns_cfg, viscosity=float(meta["viscosity"]))
        if "dealias" in meta:
            dns_cfg = replace(dns_cfg, dealias=str(meta["dealias"]))
        clamped = min(dns_cfg.t_end, 1.0 - ref_cfg.q_floor)
        if clamped < dns_cfg.t_end:
            print(f"  t_end clamped to {clamped:.3f}: the table ends at q_floor")
            dns_cfg = replace(dns_cfg, t_end=clamped)
        force_fn, field_fn = table.force_hat, None
        fingerprint = str(meta.get("fingerprint", ""))
        parent_hash = str(meta.get("parent_force_data_hash",
                                   meta.get("parent_fingerprint", "")))
        table_hash = table.sha256
    else:
        explicit_amps = any(value is not None for value in (args.amp1, args.amp2,
                                                            args.amp3, args.amp4))
        if args.calibration and explicit_amps:
            raise SystemExit("use either --calibration or --amp* flags, not both")
        if args.calibration:
            ref_cfg = replace(ref_cfg, wave_coeffs=_load_calibration(
                args.calibration, ref_cfg, dns_cfg))
        oracle = ForceOracle(replace(ref_cfg, n=dns_cfg.n, dealias=dns_cfg.dealias))
        if args.init_from_reference is not None:
            initial_hat = oracle.field_hat(args.init_from_reference)
        if args.no_force:
            zeros_hat = np.zeros_like(initial_hat)
            force_fn = lambda t: zeros_hat          # noqa: E731
            field_fn = None
            initial_state_hash = hashlib.sha256(
                np.ascontiguousarray(initial_hat).tobytes()).hexdigest()[:16]
            fingerprint, parent_hash = "", ""
        else:
            force_fn = oracle.force_hat
            field_fn = lambda t: physical(oracle.field_hat(t))   # noqa: E731
            fingerprint = restricted_force_fingerprint(ref_cfg)
            parent_hash = parent_force_data_fingerprint(ref_cfg, verbose=True)
        table_hash = ""

    # the spectral-tail stop is armed only after the start-up ramp: during the
    # ramp the state is dominated by the force's onset transient, so a large
    # tail fraction there measures the ramp rather than accumulated unresolved
    # physics.  The pre-arm peak is recorded and printed, never hidden.
    if args.tail_arm is not None:
        dns_cfg = replace(dns_cfg, tail_arm=float(args.tail_arm))
    else:
        dns_cfg = replace(dns_cfg, tail_arm=float(ref_cfg.smooth_start))

    # forced runs stop where the construction still means something: beyond
    # t = 1 - q_floor the clock is floored and a continued forced integration
    # would use a frozen field (table mode clamps for the same reason)
    if not dns_cfg.free_evolution:
        bound = 1.0 - ref_cfg.q_floor
        if dns_cfg.t_end > bound:
            print(f"  t_end clamped to {bound:g}: forced runs stop at the "
                  f"collapse clock floor (t = 1 - q_floor)")
            dns_cfg = replace(dns_cfg, t_end=bound)
        if dns_cfg.release_time >= dns_cfg.t_end:
            raise SystemExit("--init-from-reference T0 must be < the clamped "
                             "--t-end")

    print("Forced DNS of the reference construction"
          + (" (free evolution, no forcing)" if dns_cfg.free_evolution else ""))
    print(f"  grid {dns_cfg.n}^3 · nu={dns_cfg.viscosity} · dealias={dns_cfg.dealias} · "
          f"h={ref_cfg.h} · q_floor={ref_cfg.q_floor} · t_end={dns_cfg.t_end} · "
          f"tail-arm={dns_cfg.tail_arm:g}")
    if args.no_force:
        print(f"  forcing: none (free evolution released from u*(t="
              f"{args.init_from_reference:g}))")
    elif table:
        print("  forcing: frozen table (" + table.describe() + ")")
    else:
        print("  forcing: reference oracle; only force_hat(t) is called by the time loop")
    if args.no_force:
        print(f"  release state: u*(t={args.init_from_reference:g}) · "
              f"state hash {initial_state_hash}")
    elif parent_hash and parent_hash != fingerprint:
        print(f"  forcing hashes: parent f_N* data hash {parent_hash} · "
              f"restricted P_N f_N* {fingerprint}")
    elif fingerprint:
        print(f"  forcing hash (restricted P_N f_N*): {fingerprint}")

    res = simulate(dns_cfg, ref_cfg, force_fn=force_fn, field_fn=field_fn,
                   initial_hat=initial_hat)
    fits = (measure_free_decay(res, dns_cfg) if dns_cfg.free_evolution
            else measure_exponents(res, ref_cfg))
    print(f"  done: {res.steps} steps in {res.runtime:.1f} s · "
          f"final ||div u||_2 = {res.div_error:.3e}")
    if res.tail_prearm_max > dns_cfg.tail_tolerance:
        print(f"  note: pre-arm tail peak {res.tail_prearm_max:.2e} > tolerance "
              f"{dns_cfg.tail_tolerance:.2e} before t={res.tail_arm:g} "
              f"(true peak over all internal steps; recorded in the archive, "
              f"not hidden)")
    if dns_cfg.free_evolution:
        if np.isfinite(fits.velocity[0]):
            print(f"  free-evolution rates per unit time (d ln y / ds): "
                  f"max|u| {fits.velocity[0]:+.3f} ± {fits.velocity[1]:.3f} · "
                  f"l_r {fits.length_r[0]:+.3f} · l_z {fits.length_z[0]:+.3f}")
        else:
            print("  free-evolution rates: not fitted "
                  "(fewer than 4 resolved post-release frames)")
    elif np.isfinite(fits.velocity[0]):
        print(f"  measured max|u| slope {fits.velocity[0]:.3f} ± {fits.velocity[1]:.3f} "
              f"(theory {-(0.5 + ref_cfg.h):.3f})")
        print(f"  descriptive |u|^4-weighted widths (not the paper's core sizes): "
              f"l_r slope {fits.length_r[0]:.3f} ± {fits.length_r[1]:.3f} · "
              f"l_z slope {fits.length_z[0]:.3f} ± {fits.length_z[1]:.3f}")
    else:
        print("  measured slopes: not fitted (fewer than 4 resolved post-ramp frames)")
    last = max(res.stop_index - 1, 0)
    print(f"  energy: peak {np.nanmax(res.energy):.5f} · "
          f"end {res.energy[last]:.5f}")
    if np.isfinite(res.tracking).any():
        print(f"  tracking (DNS vs construction, post-hoc, ramp excluded): "
              f"max {np.nanmax(res.tracking) * 100:.2f} % · "
              f"final {res.tracking[last] * 100:.2f} %")
    elif dns_cfg.free_evolution:
        print("  tracking: not measured (free evolution)")
    elif table is not None:
        print("  tracking: not measured (frozen-table run)")
    else:
        print("  tracking: not measured (no frame reached the tracking window)")

    _write_archive(args.archive, res, dns_cfg, ref_cfg, fingerprint, parent_hash,
                   table_hash, initial_state_hash)
    print("rendering video ...")
    build_video(dns_cfg, res, fits, ref_cfg)
    print("finished")
