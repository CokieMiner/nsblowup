"""Re-render the movie from the snapshot archive without re-running the DNS.

Layout and encoding tweaks do not require re-integrating the equations; this
command rebuilds the movie from the stored snapshots, which keeps iteration on
the visualisation cheap (the physics stays exactly as simulated).

Usage:
    python cli.py rerender                    # openai_dns.npz -> openai_dns.mp4
    python cli.py rerender --image 520 --fps 30 --output openai_dns_hd.mp4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from nsblowup.core.config import DNSConfig, ReferenceConfig
from nsblowup.run import solver, video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", default="openai_dns.npz")
    parser.add_argument("--image", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--hold-frames", type=int, default=None)
    parser.add_argument("--output", default="openai_dns.mp4")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    if "dns_config" in data.files:
        cfg = DNSConfig(**json.loads(str(data["dns_config"])))
        cfg = replace(cfg, output=args.output)
    else:                       # legacy archives: fields stored one by one
        dealias = str(data["dealias"]) if "dealias" in data.files else "strict23"
        cfg = DNSConfig(
            n=int(data["n"]),
            viscosity=float(data["viscosity"]),
            t_end=float(data["times"][-1]),
            frames=int(data["times"].size),
            dealias=dealias,
            output=args.output,
        )
    if args.image is not None:
        cfg = replace(cfg, image=args.image)
    if args.samples is not None:
        cfg = replace(cfg, samples=args.samples)
    if args.fps is not None:
        cfg = replace(cfg, fps=args.fps)
    if args.hold_frames is not None:
        cfg = replace(cfg, hold_frames=args.hold_frames)

    stop = int(data["stop_index"])
    if "beyond_resolution" in data.files:
        beyond = data["beyond_resolution"].astype(bool)
        analytic = data["analytic_reference"].astype(bool)
    else:                       # legacy archives stored a single flag
        beyond = (data["from_reference"].astype(bool)
                  if "from_reference" in data.files else np.zeros(cfg.frames, bool))
        analytic = beyond
    swirl = data["swirl"] if "swirl" in data.files else None
    if "vmax" in data.files:
        vmax = float(data["vmax"])
    else:
        vmax = float(data["speeds"][:max(stop, 1)].max())

    res = solver.Result(
        times=data["times"], speeds=data["speeds"], swirl=swirl,
        beyond_resolution=beyond, analytic_reference=analytic,
        tracers=data["tracers"],
        max_speed=data["max_speed"], max_swirl=data["max_swirl"],
        max_radial=data["max_radial"], length_r=data["length_r"],
        length_z=data["length_z"], energy=data["energy"], tail=data["tail"],
        reynolds_swirl=data["reynolds_swirl"], tracking=data["tracking"],
        stop_index=stop, steps=0, runtime=0.0,
        div_error=float(data["div_error"]), vmax=vmax,
    )
    if "reference_config" in data.files:
        ref = ReferenceConfig(**json.loads(str(data["reference_config"])))
        ref = replace(ref, wave_coeffs=tuple(ref.wave_coeffs))
    else:                       # legacy archives: fields stored one by one
        ref = ReferenceConfig(n=cfg.n, dealias=cfg.dealias, viscosity=cfg.viscosity,
                              h=float(data["h"]), q_floor=float(data["q_floor"]))
    fits = (video.measure_free_decay(res, cfg) if cfg.free_evolution
            else video.measure_exponents(res, ref))
    print(f"re-rendering {args.npz} -> {cfg.output} "
          f"({len(res.times)} snapshots, image {cfg.image}, {cfg.fps} fps)")
    video.build_video(cfg, res, fits, ref)
    print("done")


if __name__ == "__main__":
    main()
