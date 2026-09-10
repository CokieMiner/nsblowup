"""Reference-side command: construction report, wave fit, frozen force table.

Run through the package entry point:

    python cli.py reference --n 64 --calibrate --save-calibration
    python cli.py reference --n 64 --force-grid 128 --freeze-force force.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from nsblowup.core.config import ReferenceConfig
from nsblowup.reference import waves
from nsblowup.reference.tables import (check_force_table_interpolation,
                                       freeze_force_table,
                                       parent_force_data_fingerprint,
                                       restricted_force_fingerprint)
from nsblowup.reference.report import report


def parse_args(argv: list[str] | None = None) -> tuple[ReferenceConfig, argparse.Namespace]:
    defaults = ReferenceConfig()
    parser = argparse.ArgumentParser(
        prog="nsblowup reference",
        description="construction report, wave-sector fit and frozen force tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h", type=float, default=defaults.h)
    parser.add_argument("--q-floor", type=float, default=defaults.q_floor)
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--dealias", choices=("strict23", "pad32"), default=defaults.dealias)
    parser.add_argument("--viscosity", type=float, default=defaults.viscosity)
    parser.add_argument("--amp1", type=float, default=defaults.wave_coeffs[0])
    parser.add_argument("--amp2", type=float, default=defaults.wave_coeffs[1])
    parser.add_argument("--amp3", type=float, default=defaults.wave_coeffs[2])
    parser.add_argument("--amp4", type=float, default=defaults.wave_coeffs[3])
    parser.add_argument("--calibrate", action="store_true",
                        help="quadratic model of the wave sector, solved and verified")
    parser.add_argument("--save-calibration", action="store_true",
                        help="write wave_calibration.json with the fitted coefficients")
    parser.add_argument("--reference-factor", type=int, default=defaults.reference_factor,
                        help="construction grid = n x factor (force then depends on n)")
    parser.add_argument("--force-grid", type=int, default=None,
                        help="absolute N*: build the whole force there, then restrict")
    parser.add_argument("--allow-large", action="store_true",
                        help="override the memory guard")
    parser.add_argument("--freeze-force", default=None, metavar="PATH",
                        help="write a frozen force table (requires --force-grid)")
    parser.add_argument("--freeze-dtau", type=float, default=5.0e-3,
                        help="spacing of the frozen table in tau = 1 - t")
    parser.add_argument("--freeze-startup-dtau", type=float, default=1.0e-3,
                        help="finer spacing of the frozen table inside the "
                             "start-up ramp (piecewise time grid)")
    parser.add_argument("--freeze-startup-window", type=float, default=None,
                        help="ramp window refined with the finer spacing "
                             "(default: smooth_start = 0.06)")
    parser.add_argument("--check-interpolation", action="store_true",
                        help="measure the frozen-table interpolation error "
                             "against the oracle (requires --force-grid)")
    args = parser.parse_args(argv)
    cfg = replace(defaults, h=args.h, q_floor=args.q_floor, n=args.n,
                  dealias=args.dealias, viscosity=args.viscosity,
                  reference_factor=args.reference_factor,
                  force_grid=args.force_grid,
                  allow_large=args.allow_large,
                  wave_coeffs=(args.amp1, args.amp2, args.amp3, args.amp4))
    return cfg, args


def main(argv: list[str] | None = None) -> None:
    cfg, args = parse_args(argv)
    try:
        cfg.validate()
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from None
    if args.check_interpolation:
        sections = check_force_table_interpolation(
            cfg, startup_dtau=args.freeze_startup_dtau,
            startup_window=args.freeze_startup_window)
        print("  frozen-grid interpolation error vs the oracle, randomized")
        print("  times, normalised by the window force scale; the ramp window")
        print("  also tests the shipped piecewise grid")
        for section in sections:
            lo, hi = section["tau_range"]
            print(f"\n  window {section['window']} (tau {lo:.2f}..{hi:.2f})")
            print(f"  {'grid':>28} {'nodes':>6} {'max err':>10} "
                  f"{'mean err':>10} {'order':>7}")
            for row in section["rows"]:
                order = row.get("order")
                order_text = f"{order:7.2f}" if order is not None else " " * 7
                print(f"  {row['grid']:>28} {row['nodes']:6d} "
                      f"{row['max_err']:10.2e} {row['mean_err']:10.2e} "
                      f"{order_text}")
        return
    if args.freeze_force:
        freeze_force_table(cfg, args.freeze_force, dtau=args.freeze_dtau,
                           startup_dtau=args.freeze_startup_dtau,
                           startup_window=args.freeze_startup_window)
        return
    fitted = report(cfg, args.calibrate or args.save_calibration)
    if args.save_calibration:
        if fitted is None:
            raise SystemExit("internal error: report() returned no fit")
        amps, before, after = fitted
        # the hashes and metadata must describe the CALIBRATED construction,
        # not the defaults the report ran with
        fitted_cfg = replace(cfg, wave_coeffs=tuple(float(value) for value in amps))
        meta_dict = waves.metadata(
            fitted_cfg, taus=[0.4, 0.6],
            metric="annulus-weighted L2 of the residual")
        payload = {
            "wave_coeffs": [float(value) for value in amps],
            "residual_energy_before": before,
            "residual_energy_after": after,
            "relative_norm_reduction": 1.0 - float(np.sqrt(after / max(before, 1e-30))),
            "method": ("quadratic model of the discrete residual over all four "
                       "coefficients, free phases; optimum verified with true "
                       "force evaluations"),
            "force_fingerprint": restricted_force_fingerprint(fitted_cfg),
            "parent_force_data_hash": parent_force_data_fingerprint(fitted_cfg),
            "metadata": meta_dict,
        }
        with open("wave_calibration.json", "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote wave_calibration.json (coeffs={payload['wave_coeffs']}, "
              f"relative norm reduction {payload['relative_norm_reduction']:.3e})")
