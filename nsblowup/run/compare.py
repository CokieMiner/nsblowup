"""Post-hoc comparison of the DNS run against the reference construction.

Reads the archive written by the `dns` command (`openai_dns.npz`) and reports,
with least-squares uncertainties:

    max |u_theta|, |u_r|        ~ tau^{-(1/2+h)},  tau^{-1/2}
    l_r, l_z, Re_theta          descriptive |u|^4-weighted quantities; no
                                theory target is claimed for them
    kinetic energy              over the resolved interval
    tracking error              ||u_num - u_ref|| / ||u_ref|| (oracle runs only)

and, from the reference layer, the momentum-balance table that shows the big
terms (d_t u, (u.grad)u, -nu Lap u, grad p) and their cancellation into the
force, plus the residual reduction achieved by the annular wave packets.
Archives with `swirl` snapshots also report the profile-collapse defect.

Usage
-----
    python cli.py compare                   # openai_dns.npz
    python cli.py compare a.npz b.npz       # overlay several resolutions
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from nsblowup.core.config import ReferenceConfig
from nsblowup.reference.reference import ForceOracle
from nsblowup.reference.waves import residual_objective
from nsblowup.run.video import fit_loglog
from nsblowup.run.volume import trilinear


def load(path: str) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def swirl_collapse_records(data: dict, ref: ReferenceConfig,
                           taus=(0.15, 0.25, 0.4, 0.6, 0.75),
                           reference_tau: float = 0.45, points: int = 120) -> list[dict]:
    """Profile-collapse defect of the stored u_theta snapshots.

    Rescales q^{1/2+h} u_theta and evaluates it at fixed X = (r/L)^2/(2q) on
    the z = 0 half-line by choosing r = sqrt(2X) L sqrt(q) per time.  On a
    self-similar field the curves coincide; the defect is the relative RMS
    difference from the curve at `reference_tau`.

    Requests whose closest stored frame lies beyond the resolution stop are
    REJECTED: post-stop frames are frozen copies, and rescaling one to a later
    tau would fake a collapse.  q is computed from the frame time actually
    used, not the nominal request (no time interpolation here).
    """
    times, swirl = data["times"], data["swirl"]
    scale = ref.similarity_scale
    beyond = data.get("beyond_resolution")
    requested = list(taus)
    reference_tau = min(requested, key=lambda tau: abs(tau - reference_tau))
    frames = {tau: int(np.argmin(np.abs(times - (1.0 - tau)))) for tau in requested}

    def usable(tau: float) -> bool:
        return beyond is None or not bool(beyond[frames[tau]])

    if not usable(reference_tau):
        return [{"tau": tau, "q": None, "reference_tau": reference_tau,
                 "defect": None, "note": "reference frame beyond resolution"}
                for tau in requested]
    q_of = {tau: float(max(1.0 - float(times[frames[tau]]), ref.q_floor))
            for tau in requested if usable(tau)}
    x_hi = 0.9 * (0.42 ** 2) / (2.0 * scale ** 2 * max(q_of.values()))
    x_grid = np.linspace(0.02, x_hi, points)
    curves = {}
    for tau, q in q_of.items():
        radius = np.sqrt(2.0 * x_grid) * scale * np.sqrt(q)
        positions = np.stack((radius, np.zeros_like(radius), np.zeros_like(radius)),
                             axis=1)
        sample = trilinear(swirl[frames[tau]].astype(np.float64), positions, ref.n)
        curves[tau] = q ** (0.5 + ref.h) * sample
    reference = curves[reference_tau]
    norm = max(float(np.sqrt(np.mean(reference ** 2))), 1e-30)
    records = []
    for tau in requested:
        if not usable(tau):
            records.append({"tau": tau, "q": None, "reference_tau": reference_tau,
                            "defect": None, "note": "frame beyond resolution"})
        else:
            records.append({
                "tau": tau, "q": q_of[tau], "reference_tau": reference_tau,
                "defect": float(np.sqrt(np.mean((curves[tau] - reference) ** 2))
                                / norm)})
    return records


def exponent_table(data: dict, ref: ReferenceConfig) -> list[tuple[str, float, float, str]]:
    taus = 1.0 - data["times"]
    window = (taus <= 0.5) & (taus >= 1.7 * ref.q_floor) & (data["times"] >= 2.5 * ref.smooth_start)
    if np.count_nonzero(window) < 4:
        window = (taus <= 0.6) & (taus >= 1.7 * ref.q_floor)
    if "beyond_resolution" in data:
        window &= ~data["beyond_resolution"].astype(bool)   # stop frame unresolved

    rows = []
    slope, err, _ = fit_loglog(taus, data["max_speed"], window)
    rows.append(("max |u|", slope, err, f"{-(0.5 + ref.h):.3f}"))
    slope, err, _ = fit_loglog(taus, data["max_swirl"], window)
    rows.append(("max |u_theta|", slope, err, f"{-(0.5 + ref.h):.3f}"))
    slope, err, _ = fit_loglog(taus, data["max_radial"], window)
    rows.append(("max |u_r|", slope, err, "-0.500"))
    # widths and Re_theta use the descriptive |u|^4-weighted definition, which
    # is not the paper's core size: no theory target is printed for them
    slope, err, _ = fit_loglog(taus, data["length_r"], window)
    rows.append(("l_r", slope, err, ""))
    slope, err, _ = fit_loglog(taus, data["length_z"], window)
    rows.append(("l_z", slope, err, ""))
    slope, err, _ = fit_loglog(taus, data["reynolds_swirl"], window)
    rows.append(("Re_theta", slope, err, ""))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", nargs="*", default=["openai_dns.npz"])
    parser.add_argument("--allow-unverified", action="store_true",
                        help="compare archives even when some lack a "
                             "parent-force hash")
    args = parser.parse_args()

    runs = [(path, load(path)) for path in args.files]
    first = runs[0][1]
    # a resolution study compares archives produced by ONE parent forcing
    parents = {str(data.get("parent_force_hash", "")) for _, data in runs}
    parents.discard("")
    missing = [path for path, data in runs
               if not str(data.get("parent_force_hash", ""))]
    resolutions = {int(data["n"]) for _, data in runs}
    if len(parents) > 1:
        raise SystemExit("not a resolution convergence set: the archives were "
                         "produced by different parent forces:\n  "
                         + "\n  ".join(sorted(parents)))
    if parents and missing and len(resolutions) > 1 and not args.allow_unverified:
        raise SystemExit(
            "mixed checksums: some archives carry a parent-force hash and "
            "some do not, across different resolutions:\n  "
            + "\n  ".join(missing)
            + "\nPass --allow-unverified to compare anyway (the unverified "
              "runs may not share the parent forcing).")
    if parents:
        print(f"  common parent force hash: {parents.pop()} "
              "(canonical float32 data fingerprint)")
        if missing:
            print(f"  note: {len(missing)} archive(s) carry no parent-force hash")
    elif len(resolutions) > 1:
        print("  note: no parent-force hash recorded (grid-relative forcing): "
              "cross-resolution identity is not established")
    codes = sorted({str(data.get("code_hash", "")) for _, data in runs} - {""})
    run_codes = sorted({str(data.get("run_code_hash", "")) for _, data in runs}
                       - {""})
    if codes:
        print("  archive construction code hash(es): " + ", ".join(codes))
    if run_codes:
        print("  archive run code hash(es): " + ", ".join(run_codes))

    # restore the exact configuration that produced the archive: wave
    # coefficients, dealiasing, similarity scale and force grid are all part of
    # the frozen force, so a manual reconstruction could silently analyse a
    # different construction.
    if "reference_config" in first:
        ref = ReferenceConfig(**json.loads(str(first["reference_config"])))
        ref = replace(ref, wave_coeffs=tuple(ref.wave_coeffs))
        source = "config JSON stored in the archive"
    else:                        # legacy archives: fields stored one by one
        saved_coeffs = first.get("wave_coeffs", np.zeros(4))
        ref = ReferenceConfig(
            h=float(first["h"]), q_floor=float(first["q_floor"]),
            viscosity=float(first["viscosity"]), n=int(first["n"]),
            dealias=str(first.get("dealias", "strict23")),
            wave_coeffs=tuple(float(value) for value in np.atleast_1d(saved_coeffs)),
        )
        if "similarity_scale" in first:
            ref = replace(ref, similarity_scale=float(first["similarity_scale"]))
        if "j0" in first:
            ref = replace(ref, j0=float(first["j0"]))
        source = "legacy fields (this archive predates the config JSON)"
    print(f"  reference config restored ({source}):\n"
          f"    n={ref.n} dealias={ref.dealias} h={ref.h} q_floor={ref.q_floor} "
          f"force_grid={ref.force_grid}\n"
          f"    wave_coeffs={ref.wave_coeffs} "
          f"similarity_scale={ref.similarity_scale} j0={ref.j0}")
    dns_saved = json.loads(str(first["dns_config"])) if "dns_config" in first else {}
    free_run = bool(dns_saved.get("free_evolution", False))

    if free_run:
        print("\n  free-evolution run: the τ-based exponent table does not apply")
    else:
        print("measured exponents (DNS) vs paper prediction")
        print(f"  {'quantity':>14} {'measured':>10} {'±':>8} {'theory':>10}")
        rows = exponent_table(first, ref)
        for name, slope, err, theory in rows:
            print(f"  {name:>14} {slope:10.3f} {err:8.3f} {theory:>10}")

    energy = first["energy"]
    stop = int(first["stop_index"])
    print(f"\n  energy over the resolved interval: peak {np.nanmax(energy):.6f} · "
          f"final resolved {energy[max(stop - 1, 0)]:.6f}")
    print(f"  divergence (final): {float(first['div_error']):.3e}")
    tracking = first.get("tracking")
    if tracking is not None and np.isfinite(tracking).any():
        print(f"  tracking error (DNS vs construction): max "
              f"{np.nanmax(tracking) * 100:.2f} %")
    else:
        print("  tracking: not measured (frozen-table run; the solver never "
              "sees the construction)")

    print("\n  momentum balance from the reference layer (term norms filtered")
    print("  to the force's retained band, max norms; f/max term is NOT a")
    print("  cancellation measure)")
    oracle_cfg = replace(ref, n=min(ref.n, 48))
    print(f"  {'tau':>6} {'d_t u':>10} {'(u.g)u':>10} {'-nu Lap u':>10} "
          f"{'grad p':>10} {'f':>10} {'f/max term':>10} {'cancellation':>12}")
    taus = [0.10, 0.20, 0.40, 0.70]
    for tau in taus:
        terms = ForceOracle(oracle_cfg).momentum_terms(1.0 - tau)
        print(f"  {tau:6.2f} {terms['d_t u']:10.3f} {terms['(u.grad)u']:10.3f} "
              f"{terms['-nu Lap u']:10.3f} {terms['grad p']:10.3f} "
              f"{terms['f (sum)']:10.3f} "
              f"{terms['force / largest-term ratio']:10.4f} "
              f"{terms['cancellation']:12.4f}")

    print("\n  annular wave packets: residual reduction (box RMS and the")
    print("  annulus-weighted L2 objective the calibration minimises)")
    print(f"  {'tau':>6} {'base box':>10} {'waves box':>10} {'box ratio':>10} "
          f"{'base ann':>10} {'waves ann':>10} {'ann ratio':>10}")
    zero = (0.0, 0.0, 0.0, 0.0)
    for tau in (0.15, 0.25, 0.4, 0.6):
        base = residual_objective(oracle_cfg, 1.0 - tau, zero)
        with_waves = residual_objective(oracle_cfg, 1.0 - tau,
                                        oracle_cfg.wave_coeffs)
        print(f"  {tau:6.2f} {base['box_rms']:10.5f} {with_waves['box_rms']:10.5f} "
              f"{base['box_rms'] / max(with_waves['box_rms'], 1e-30):9.3f}x "
              f"{base['annulus_norm']:10.5f} "
              f"{with_waves['annulus_norm']:10.5f} "
              f"{base['annulus_norm'] / max(with_waves['annulus_norm'], 1e-30):9.3f}x")

    print("\n  swirl-profile self-similarity (DNS, z = 0 half-line)")
    if first.get("swirl") is None:
        print("  not available: rerun the dns command with --store-swirl")
    else:
        for rec in swirl_collapse_records(first, ref):
            if rec["defect"] is None:
                print(f"  tau={rec['tau']:.2f}  not usable: {rec['note']}")
                continue
            note = "  (reference)" if rec["tau"] == rec["reference_tau"] else ""
            print(f"  tau={rec['tau']:.2f} q={rec['q']:.3f}  "
                  f"defect {rec['defect']:.3e}{note}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="#0b1018")
    for ax in axes.ravel():
        ax.set_facecolor("#0b1018")
        ax.tick_params(colors="#9fb6cc")
        for spine in ax.spines.values():
            spine.set_color("#22364f")
        ax.grid(alpha=0.2, color="#7fa8c9")

    def run_x(data: dict):
        """Panel coordinates of one archive in its own mode (tau, or s after release)."""
        saved = json.loads(str(data["dns_config"])) if "dns_config" in data else {}
        if bool(saved.get("free_evolution", False)):
            return data["times"] - float(saved.get("release_time", 0.0))
        return 1.0 - data["times"]

    taus_arr = 1.0 - first["times"]
    x_first = run_x(first)
    mode_mixed = False
    for path, data in runs:
        saved = json.loads(str(data["dns_config"])) if "dns_config" in data else {}
        run_free = bool(saved.get("free_evolution", False))
        mode_mixed = mode_mixed or (run_free != free_run)
        axes[0, 0].plot(run_x(data), data["max_speed"], lw=1.4,
                        label=f"n={data['n']} max|u|" + (", free" if run_free else ""))
    if mode_mixed:
        print("  note: archives mix forced and free runs; each panel uses each "
              "run's own clock (tau vs s)")
    axes[0, 0].set_yscale("log")
    if free_run:
        axes[0, 0].axvline(0.0, color="#5d7a99", lw=0.8, ls="--", label="release")
        axes[0, 0].set_title("measured speed (free evolution, no forcing)",
                             color="#eaf4ff")
        axes[0, 0].set_xlabel("s = time after release", color="#cfe8ff")
    else:
        theory = first["max_speed"][np.argmin(np.abs(taus_arr - 0.5))]
        axes[0, 0].plot(taus_arr, theory * (taus_arr / 0.5) ** (-(0.5 + ref.h)),
                        "w:", lw=0.9, label=fr"$\tau^{{-(1/2+h)}}$")
        axes[0, 0].set_xscale("log")
        axes[0, 0].invert_xaxis()
        axes[0, 0].set_title("measured speed vs theory", color="#eaf4ff")
        axes[0, 0].set_xlabel(r"$\tau$", color="#cfe8ff")
    axes[0, 0].legend(fontsize=7, facecolor="#111a26", edgecolor="#22364f",
                      labelcolor="#cfe8ff")

    axes[0, 1].plot(x_first, first["length_r"], color="#7dffc4", lw=1.4, label=r"$\ell_r$")
    axes[0, 1].plot(x_first, first["length_z"], color="#f9a03f", lw=1.4, label=r"$\ell_z$")
    axes[0, 1].set_yscale("log")
    if free_run:
        axes[0, 1].set_xlabel("s = time after release", color="#cfe8ff")
    else:
        axes[0, 1].set_xscale("log")
        axes[0, 1].invert_xaxis()
        axes[0, 1].set_xlabel(r"$\tau$", color="#cfe8ff")
    axes[0, 1].set_title("descriptive |u|^4-weighted widths (no theory claim)",
                         color="#eaf4ff")
    axes[0, 1].legend(fontsize=7, facecolor="#111a26", edgecolor="#22364f",
                      labelcolor="#cfe8ff")

    for path, data in runs:
        axes[1, 0].plot(data["times"], data["energy"], lw=1.4, label=f"n={data['n']}")
    if stop < first["times"].size:
        stop_t = first["times"][stop]
        axes[1, 0].axvline(stop_t, color="#ff8fa3", lw=0.9, ls="--",
                           label="DNS resolution limit")
    axes[1, 0].set_title("kinetic energy over the resolved interval", color="#eaf4ff")
    axes[1, 0].set_xlabel("t", color="#cfe8ff")
    axes[1, 0].legend(fontsize=7, facecolor="#111a26", edgecolor="#22364f",
                      labelcolor="#cfe8ff")

    plotted_tracking = False
    for path, data in runs:
        series = data.get("tracking")
        if series is not None and np.isfinite(series).any():
            axes[1, 1].plot(data["times"], series * 100.0, lw=1.4,
                            label=f"n={data['n']}")
            plotted_tracking = True
    if plotted_tracking:
        axes[1, 1].set_title("tracking error  ||u_num - u_ref|| / ||u_ref||  [%]",
                             color="#eaf4ff")
        axes[1, 1].legend(fontsize=7, facecolor="#111a26", edgecolor="#22364f",
                          labelcolor="#cfe8ff")
    else:
        axes[1, 1].set_title("tracking not measured (frozen-table or free run)",
                             color="#eaf4ff")
    axes[1, 1].set_xlabel("t", color="#cfe8ff")

    fig.suptitle("DNS vs OpenAI-style reference construction",
                 color="#eaf4ff")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=130)
    plt.close(fig)
    print("\nwrote comparison.png")


if __name__ == "__main__":
    main()
