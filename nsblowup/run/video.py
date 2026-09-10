"""Movie and measurement panels: presentation only, no physics.

Kept separate from the solver so that the solver contains exactly the
experiment (force-only time loop) and this module contains exactly the
presentation.  The movie can be rebuilt from a stored archive by the
`rerender` command through this module, without re-integrating anything.

The viewport draws DNS data only: when the run stopped at the resolution
limit, later (analytic reference) frames are stored in the archive but not
drawn here.  Panels show the measured collapse exponents against the paper's
theory lines, the bounded kinetic energy and the spectral-tail monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from nsblowup.core.config import ReferenceConfig
from nsblowup.run.volume import WATER, camera_basis, render_volume, trilinear

if TYPE_CHECKING:                    # type annotations only (no runtime use)
    from nsblowup.run.solver import DNSConfig, Result


# --------------------------------------------------------------------------
# measured exponents (with least-squares uncertainty)
# --------------------------------------------------------------------------
def fit_loglog(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, stderr, anchor value) for y ~ x**slope on the mask."""
    xm, ym = x[mask], y[mask]
    good = ym > 0.0
    xm, ym = xm[good], ym[good]
    if xm.size < 4:
        return float("nan"), float("nan"), float("nan")
    logx, logy = np.log(xm), np.log(ym)
    slope, intercept = np.polyfit(logx, logy, 1)
    residual = logy - (slope * logx + intercept)
    dof = max(xm.size - 2, 1)
    stderr = float(np.sqrt(np.sum(residual ** 2) / dof /
                           np.sum((logx - logx.mean()) ** 2)))
    return float(slope), stderr, float(np.exp(intercept))


@dataclass
class Fits:
    velocity: tuple[float, float, float]
    length_r: tuple[float, float, float]
    length_z: tuple[float, float, float]


def measure_exponents(res: Result, ref: ReferenceConfig) -> Fits:
    taus = 1.0 - res.times
    tau_lo = 1.7 * ref.q_floor
    window = (taus <= 0.5) & (taus >= tau_lo) & (res.times >= 2.5 * ref.smooth_start)
    if np.count_nonzero(window) < 4:
        window = (taus <= 0.6) & (taus >= tau_lo)
    window &= ~res.beyond_resolution          # the stop frame is unresolved
    return Fits(
        velocity=fit_loglog(taus, res.max_speed, window),
        length_r=fit_loglog(taus, res.length_r, window),
        length_z=fit_loglog(taus, res.length_z, window),
    )


def _theory_line(taus: np.ndarray, measured: np.ndarray, exponent: float) -> np.ndarray:
    k = int(np.argmin(np.abs(taus - 0.5)))
    anchor = max(measured[k], 1e-30)
    return anchor * (taus / taus[k]) ** exponent


def measure_free_decay(res: Result, cfg: DNSConfig) -> Fits:
    """Log-linear decay rates d ln y / ds of a free-evolution run.

    Fitted over the window s >= 10% of the span, which excludes the release
    transient and still covers the late viscous decay.
    """
    s = res.times - cfg.release_time
    window = (s >= 0.1 * max(float(s[-1]), 1e-9)) & ~res.beyond_resolution

    def rate(values) -> tuple[float, float, float]:
        y = np.asarray(values, dtype=float)
        good = window & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(good) < 4:
            return (float("nan"), float("nan"), float("nan"))
        slope, intercept = np.polyfit(s[good], np.log(y[good]), 1)
        residual = np.log(y[good]) - (slope * s[good] + intercept)
        dof = max(int(np.count_nonzero(good)) - 2, 1)
        stderr = float(np.sqrt(np.sum(residual ** 2) / dof
                               / np.sum((s[good] - s[good].mean()) ** 2)))
        return (float(slope), stderr, float(np.exp(intercept)))

    return Fits(velocity=rate(res.max_speed), length_r=rate(res.length_r),
                length_z=rate(res.length_z))


def _slope_label(slope: float, stderr: float, theory: float | None = None) -> str:
    """Slope with uncertainty; explicit text when the fit window is too short."""
    if not np.isfinite(slope):
        return "slope not fitted (window < 4 points)"
    text = f"slope {slope:.3f} ± {stderr:.3f}"
    if theory is not None:
        text += f"  (theory {theory:.3f})"
    return text


# --------------------------------------------------------------------------
# figure / video
# --------------------------------------------------------------------------
def build_video(cfg: DNSConfig, res: Result, fits: Fits, ref: ReferenceConfig) -> None:
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#05070d")

    ax_img = fig.add_axes([0.015, 0.03, 0.61, 0.885])
    ax_img.set_axis_off()
    frame0 = render_volume(res.speeds[0].astype(np.float32), res.vmax, cfg.image,
                           cfg.samples, cfg.zoom, cfg.azim, cfg.elev,
                           cfg.exposure, cfg.opacity, cfg.density_power)
    image_artist = ax_img.imshow(frame0, extent=(0, cfg.image, cfg.image, 0),
                                 interpolation="bilinear", zorder=0)
    ax_img.set_xlim(0, cfg.image)
    ax_img.set_ylim(cfg.image, 0)
    trail_artist = LineCollection([], linewidths=1.0, zorder=2)
    ax_img.add_collection(trail_artist)
    points_artist = ax_img.scatter([], [], s=3.5, zorder=3, linewidths=0)

    # collapse-following display zoom (pure camera): zoom ~ sqrt(tau), so the
    # shrinking core stays visible.  Labelled on screen; the physics is the DNS.
    # Free evolution has no collapse clock: fixed zoom on the released state.
    free = cfg.free_evolution
    s_axis = res.times - cfg.release_time
    taus = np.clip(1.0 - res.times, ref.q_floor, 1.0)
    if free:
        zooms = np.full_like(s_axis, cfg.zoom)
    elif cfg.follow_collapse:
        zooms = cfg.zoom * np.sqrt(taus / taus[0])
    else:
        zooms = np.full_like(taus, cfg.zoom)
    zoom_schedule = np.clip(zooms, 0.05, 0.95)
    ref_label = ax_img.text(0.5 * cfg.image, 0.5 * cfg.image,
                            "ANALYTIC REFERENCE\n(beyond DNS resolution)",
                            color="#ffd7a8", fontsize=11, ha="center", va="center",
                            alpha=0.0, zorder=4)

    cax = fig.add_axes([0.622, 0.335, 0.009, 0.400])
    sm = plt.cm.ScalarMappable(cmap=WATER)
    sm.set_clim(0.0, res.vmax)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("speed |u|", color="#cfe8ff", fontsize=7, labelpad=3)
    cb.ax.tick_params(colors="#cfe8ff", labelsize=6.5)
    cb.outline.set_edgecolor("#22364f")

    ax_v = fig.add_axes([0.700, 0.590, 0.268, 0.240])
    ax_l = fig.add_axes([0.700, 0.320, 0.268, 0.175])
    ax_e = fig.add_axes([0.700, 0.075, 0.268, 0.170])
    for ax in (ax_v, ax_l, ax_e):
        ax.set_facecolor("#070b13")
        ax.tick_params(colors="#9fb6cc", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#22364f")
        ax.grid(alpha=0.18, color="#7fa8c9")

    slope, stderr, _ = fits.velocity
    if free:
        ax_v.plot(s_axis, res.max_speed, color="#6fd3ff", lw=1.5,
                  label="DNS max |u|")
        ax_v.axvline(0.0, color="#5d7a99", lw=0.8, ls="--")
        ax_v.annotate("release", xy=(0.0, 0.98),
                      xycoords=("data", "axes fraction"), fontsize=6,
                      color="#9fb6cc", va="top", ha="left")
        ax_v.set_yscale("log")
        title = (f"max |u|: d ln|u|/ds = {slope:+.3f} ± {stderr:.3f}"
                 if np.isfinite(slope) else "max |u| (rate not fitted)")
        ax_v.set_title(title, color="#eaf4ff", fontsize=7.5)
        ax_v.set_xlabel("s = time after release", color="#cfe8ff",
                        fontsize=7, labelpad=1)
    else:
        ax_v.plot(taus, res.max_speed, color="#6fd3ff", lw=1.5, label="DNS max |u|")
        ax_v.plot(taus, _theory_line(taus, res.max_speed, -(0.5 + ref.h)),
                  color="white", lw=0.9, ls=":",
                  label=fr"theory $\tau^{{-(1/2+h)}}$")
        ax_v.axvline(ref.q_floor, color="#5d7a99", lw=0.8, ls="--")
        ax_v.annotate("q_floor", xy=(ref.q_floor, 0.98),
                      xycoords=("data", "axes fraction"), fontsize=6,
                      color="#9fb6cc", va="top", ha="right")
        ax_v.set_xscale("log"); ax_v.set_yscale("log"); ax_v.invert_xaxis()
        ax_v.set_title(f"max |u|: {_slope_label(slope, stderr, -(0.5 + ref.h))}",
                       color="#eaf4ff", fontsize=7.5)
        ax_v.set_xlabel(r"$\tau$", color="#cfe8ff", fontsize=7, labelpad=1)
    peak = float(np.nanmax(res.max_speed))
    ax_v.set_ylim(max(peak * 2.0e-2, 1e-6), peak * 1.6)
    if free:
        ax_v.set_xlim(0.0, max(float(s_axis[-1]), 1e-9))
    else:
        finite = np.isfinite(res.max_speed) & (res.max_speed > 0.02 * peak)
        if finite.any():
            ax_v.set_xlim(0.62, float(np.min(1.0 - res.times[finite])) * 0.85)
    ax_v.set_ylabel("max |u|", color="#cfe8ff", fontsize=7)
    ax_v.tick_params(labelsize=6.5)
    ax_v.legend(fontsize=5.5, loc="upper left", facecolor="#0b1422",
                edgecolor="#22364f", labelcolor="#cfe8ff")
    marker_v, = ax_v.plot([], [], "o", color="white", ms=5)

    sl_r, se_r, _ = fits.length_r
    sl_z, se_z, _ = fits.length_z

    def _width_label(slope: float, stderr: float) -> str:
        if not free:
            return _slope_label(slope, stderr)
        if not np.isfinite(slope):
            return "rate not fitted"
        return f"d ln/ds = {slope:+.3f} ± {stderr:.3f}"

    x_widths = s_axis if free else taus
    ax_l.plot(x_widths, res.length_r, color="#7dffc4", lw=1.4)
    ax_l.plot(x_widths, res.length_z, color="#f9a03f", lw=1.4)
    ax_l.set_yscale("log")
    if free:
        ax_l.axvline(0.0, color="#5d7a99", lw=0.8, ls="--")
        ax_l.set_xlim(0.0, max(float(s_axis[-1]), 1e-9))
    else:
        ax_l.set_xscale("log"); ax_l.invert_xaxis()
        ax_l.axvline(ref.q_floor, color="#5d7a99", lw=0.8, ls="--")
    ax_l.set_title("|u|^4-weighted widths (descriptive)",
                   color="#eaf4ff", fontsize=7.0)
    ax_l.set_xlabel("s = time after release" if free else r"$\tau$",
                    color="#cfe8ff", fontsize=7)
    ax_l.tick_params(labelsize=6.5)
    # slope annotations in the free top-right corner (the curves fall to the
    # right, so a legend box there would not overlap them); no legend frame
    ax_l.text(0.98, 0.96, fr"$\ell_r$  {_width_label(sl_r, se_r)}",
              transform=ax_l.transAxes, ha="right", va="top",
              fontsize=5.5, color="#7dffc4")
    ax_l.text(0.98, 0.87, fr"$\ell_z$  {_width_label(sl_z, se_z)}",
              transform=ax_l.transAxes, ha="right", va="top",
              fontsize=5.5, color="#f9a03f")

    x_e = s_axis if free else res.times
    ax_e.plot(x_e, res.energy, color="#ffd166", lw=1.4, label="energy")
    ax_e.set_xlim(0.0, float(x_e[-1]))
    ax_e.set_title("kinetic energy and spectral tail", color="#eaf4ff",
                   fontsize=7.5)
    ax_e.set_xlabel("s = time after release" if free else "time t",
                    color="#cfe8ff", fontsize=7)
    ax_e.set_ylabel("energy", color="#ffd166", fontsize=7)
    ax_e.tick_params(labelsize=7)
    ax_tail = ax_e.twinx()
    ax_tail.plot(x_e, res.tail, color="#ff8fa3", lw=1.0, label="tail")
    ax_tail.set_yscale("log")
    ax_tail.axhline(cfg.tail_tolerance, color="#ff8fa3", lw=0.7, ls="--")
    ax_tail.set_ylabel("spectral tail", color="#ff8fa3", fontsize=7)
    ax_tail.tick_params(colors="#ff8fa3", labelsize=6)
    if res.stop_index < cfg.frames:
        ax_e.axvspan(x_e[res.stop_index], float(x_e[-1]), color="#2a0d14",
                     alpha=0.6, zorder=0)
    marker_e, = ax_e.plot([], [], "o", color="white", ms=4)

    if free:
        fig.text(0.015, 0.960, "Free evolution of the constructed state (no forcing)",
                 color="#eaf4ff", fontsize=12.5, fontweight="bold")
        fig.text(0.015, 0.930,
                 f"released from u*(τ = {1.0 - cfg.release_time:.2f}) · {cfg.n}³ · "
                 f"ν={cfg.viscosity} · {cfg.dealias}",
                 color="#9fb6cc", fontsize=7.5)
    else:
        fig.text(0.015, 0.960,
                 "DNS · u(x, 0) = 0 · constructed external force",
                 color="#eaf4ff", fontsize=12.5, fontweight="bold")
        fig.text(0.015, 0.930,
                 f"finite-scale surrogate of the OpenAI construction · {cfg.n}³ · "
                 f"ν={cfg.viscosity} · h={ref.h} · q_floor={ref.q_floor} · {cfg.dealias}",
                 color="#9fb6cc", fontsize=7.5)
    status = fig.text(0.690, 0.965, "", color="#eaf4ff", fontsize=8.0, va="top")
    if free:
        footer = (f"no forcing: free evolution released from u*(t={cfg.release_time:.2f})"
                  "  ·  panels: max|u|, widths and energy vs s")
    else:
        footer = ("solver reads only the external force; u_ref compared post-hoc  ·  "
                  "panels: max|u| with theory line, |u|^4-weighted widths, energy and tail")
    if res.stop_index < cfg.frames:
        footer += (f"  ·  DNS stopped at t={res.times[res.stop_index]:.3f}")
    fig.text(0.985, 0.012, footer, color="#5d7a99", fontsize=6.5, ha="right")
    mode = ("display zoom follows the core (camera only)" if cfg.follow_collapse
            else "fixed camera")
    fig.text(0.985, 0.032, mode, color="#5d7a99", fontsize=6.5, ha="right")

    # The main viewport shows DNS data only: if the simulation stopped at the
    # resolution limit, later (analytic reference) frames are not shown here.
    resolved = res.stop_index if res.stop_index < len(res.times) else len(res.times)
    n_movie = max(resolved, 1)
    last = n_movie - 1

    def update(frame: int):
        # frames beyond the last resolved snapshot freeze that state (end hold)
        frame = min(frame, last)
        view = float(zoom_schedule[frame])
        image_artist.set_data(render_volume(res.speeds[frame].astype(np.float32), res.vmax,
                                            cfg.image, cfg.samples, view, cfg.azim,
                                            cfg.elev, cfg.exposure, cfg.opacity,
                                            cfg.density_power))
        pos = res.tracers[frame]
        if pos.shape[0] > 0:
            d, right, up = camera_basis(cfg.azim, cfg.elev)
            sx = ((pos @ right) / view * 0.5 + 0.5) * cfg.image
            sy = (0.5 - (pos @ up) / view * 0.5) * cfg.image
            speed_now = trilinear(res.speeds[frame].astype(np.float32), pos, cfg.n)
            hot = np.clip(speed_now / max(res.vmax, 1e-9), 0.0, 1.0)
            rgba = np.empty((pos.shape[0], 4), dtype=np.float32)
            rgba[:, 0] = 0.72 + 0.28 * hot
            rgba[:, 1] = 0.90 + 0.10 * hot
            rgba[:, 2] = 1.0
            rgba[:, 3] = 0.18 + 0.62 * hot
            if res.beyond_resolution[frame]:
                rgba[:, 3] *= 0.25
            first = max(0, frame - cfg.trail + 1)
            nseg = frame - first
            if nseg > 0 and not res.beyond_resolution[first]:
                q_hist = res.tracers[first:frame + 1]
                qx = ((q_hist @ right) / view * 0.5 + 0.5) * cfg.image
                qy = (0.5 - (q_hist @ up) / view * 0.5) * cfg.image
                segs = np.stack((np.stack((qx[:-1], qy[:-1]), axis=-1),
                                 np.stack((qx[1:], qy[1:]), axis=-1)), axis=-2)
                segs = segs.reshape(-1, 2, 2)
                rgba_seg = np.empty((segs.shape[0], 4), dtype=np.float32)
                rgba_seg[:, :3] = np.tile(rgba[:, :3] * 0.9, (nseg, 1))
                fade = np.repeat(np.linspace(0.05, 0.40, nseg), pos.shape[0])
                rgba_seg[:, 3] = fade * np.tile(rgba[:, 3], nseg) ** 0.5
                trail_artist.set_segments(segs)
                trail_artist.set_color(rgba_seg)
            else:
                trail_artist.set_segments([])
            points_artist.set_offsets(np.column_stack((sx, sy)))
            points_artist.set_facecolors(rgba)
        else:
            trail_artist.set_segments([])
            points_artist.set_offsets(np.empty((0, 2)))

        tt = res.times[frame]
        x_marker = s_axis[frame] if free else 1.0 - tt
        marker_v.set_data([x_marker], [res.max_speed[frame]])
        marker_e.set_data([(s_axis[frame] if free else tt)], [res.energy[frame]])
        tag = "  [ANALYTIC REFERENCE]" if res.analytic_reference[frame] else (
            "  [DNS, frozen after the limit]" if res.beyond_resolution[frame]
            else "  [DNS]")
        ref_label.set_alpha(0.9 if res.analytic_reference[frame] else 0.0)
        head = (f"t = {tt:.3f}   s = {max(tt - cfg.release_time, 0.0):.3f}{tag}"
                if free else f"t = {tt:.3f}   τ = {1.0 - tt:.3f}{tag}")
        status.set_text(
            head + "\n"
            f"max |u| = {res.max_speed[frame]:.3f}   "
            f"ℓ_r = {res.length_r[frame]:.4f}   ℓ_z = {res.length_z[frame]:.4f}"
            f"  (descriptive widths)\n"
            f"Re_θ = {res.reynolds_swirl[frame]:.1f}   "
            f"tail = {res.tail[frame]:.2e}"
        )
        return (image_artist, trail_artist, points_artist, marker_v, marker_e,
                status, ref_label)

    total = n_movie + max(cfg.hold_frames, 0)
    animation = FuncAnimation(fig, update, frames=total, interval=1000 // max(cfg.fps, 1),
                              blit=False)
    print(f"  duration: {total} frames at {cfg.fps} fps = {total / max(cfg.fps, 1):.1f} s "
          f"({n_movie} DNS-resolved + {max(cfg.hold_frames, 0)} hold)")
    if res.stop_index < len(res.times):
        if res.analytic_reference.any():
            print(f"  viewport shows DNS frames only (0..{n_movie - 1}); "
                  f"analytic reference frames are stored in the archive, not drawn")
        else:
            print(f"  viewport shows DNS frames only (0..{n_movie - 1}); later "
                  f"frames are frozen after the resolution stop, not drawn")
    try:
        animation.save(cfg.output, writer=FFMpegWriter(fps=cfg.fps, bitrate=4000))
        print(f"wrote {cfg.output}")
    except (FileNotFoundError, RuntimeError):
        fallback = cfg.output.rsplit(".", 1)[0] + ".gif"
        print(f"ffmpeg unavailable; writing {fallback}")
        animation.save(fallback, writer=PillowWriter(fps=cfg.fps))
    plt.close(fig)
