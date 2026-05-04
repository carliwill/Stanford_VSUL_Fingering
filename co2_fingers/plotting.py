"""
co2_fingers.plotting
====================
All visualization functions for CO2 fingering analysis.

Each function is self-contained and returns the matplotlib Figure so callers
can further customize or save it.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Sequence

_SECONDS_PER_HOUR = 3600.0


def _to_hours(times_sec: np.ndarray, time_unit: str) -> tuple[np.ndarray, str]:
    """Convert time array and return (converted_array, xlabel_string)."""
    if time_unit == "h":
        return times_sec / _SECONDS_PER_HOUR, "Time (h)"
    return times_sec, "Time (s)"


# ---------------------------------------------------------------------------
# Baseline / interface inspection
# ---------------------------------------------------------------------------

def plot_baseline_frame(
    image: np.ndarray,
    baseline: dict,
    y_top: int = 2300,
    y_bot: int = 3500,
    x_left: int = 0,
    x_right: int = 6000,
    scale: float = 0.25,
    save_path: str | None = None,
    title: str = "Static baseline frame",
) -> plt.Figure:
    """
    Two-panel plot: (A) baseline frame image with overlays, (B) 1-D signal.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR baseline image.
    baseline : dict
        Output of :func:`co2_fingers.baseline.compute_static_baseline`.
    y_top, y_bot, x_left, x_right : int
        Crop limits (pixels).
    scale : float
        Display scale for panel A (default 0.25).
    save_path : str or None
        If provided, save figure to this path.
    title : str
        Figure super-title.

    Returns
    -------
    plt.Figure
    """
    from .fingers import detect_fingers

    x_base = baseline["x_base"]
    y_bar  = baseline["y_bar_static"]
    smooth = baseline["smooth_iface"]
    resid  = baseline["residual"]
    upper  = baseline["upper_band"]
    lower  = baseline["lower_band"]
    prom   = baseline["prominence"]
    sigma  = baseline["sigma_h"]

    v_idx = detect_fingers(smooth, resid, x_base,
                            prominence_override=prom,
                            baseline_y_bar=y_bar)
    v_x = x_base[v_idx]
    v_y = smooth[v_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={"wspace": 0.08})
    fig.suptitle(f"{title}  --  sigma_h = {sigma:.1f} px   prom = {prom:.1f} px   N = {len(v_idx)}",
                 fontsize=12, fontweight="bold")

    cropped = image[y_top:y_bot, x_left:x_right]
    disp    = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)
    rgb     = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    x_sc    = x_base * scale

    axes[0].imshow(rgb)
    axes[0].plot(x_sc, smooth * scale, color="cyan",   lw=1,   label="Gaussian outline")
    axes[0].plot(x_sc, y_bar  * scale, color="yellow", lw=1.5, ls="--", label="y_bar (static baseline)")
    axes[0].fill_between(x_sc, lower * scale, upper * scale,
                         color="yellow", alpha=0.18, label="+-2.5 sigma band")
    axes[0].scatter(v_x * scale, v_y * scale, c="red", s=35, zorder=5,
                    label=f"fingers  N={len(v_idx)}")
    y_tick = np.mean(smooth * scale) * 0.6
    for vx, vy in zip(v_x * scale, v_y * scale):
        axes[0].vlines(vx, y_tick - 8, y_tick + 8, colors="red", lw=1.5)
    axes[0].set_title("(A) Baseline frame with overlay", fontsize=10)
    axes[0].axis("off")
    axes[0].legend(fontsize=7, loc="lower right")

    axes[1].plot(x_base, smooth, color="cyan",   lw=1,   label="smooth interface")
    axes[1].plot(x_base, y_bar,  color="yellow", lw=1.5, ls="--", label="y_bar (static baseline)")
    axes[1].fill_between(x_base, lower, upper, color="yellow", alpha=0.18, label="+-2.5 sigma band")
    axes[1].scatter(v_x, v_y, c="red", s=25, zorder=5, label=f"valleys  N={len(v_idx)}")
    axes[1].plot(x_base, resid, color="white", lw=0.6, alpha=0.4, label="residual")
    axes[1].axhline(0, color="gray", lw=0.5, ls=":")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="x (pixels)", ylabel="y (pixels)",
                title="(B) 1-D interface + static band + residual")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


def plot_finger_check(
    image: np.ndarray,
    smooth_iface: np.ndarray,
    x_vals: np.ndarray,
    baseline: dict,
    widths_info: dict,
    filename: str = "",
    y_top: int = 2300,
    y_bot: int = 3500,
    x_left: int = 0,
    x_right: int = 6000,
    scale: float = 0.25,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel finger-detection check for a single frame.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR image.
    smooth_iface : np.ndarray
        Smoothed interface from :func:`co2_fingers.interface.gauss_outline`.
    x_vals : np.ndarray
        Corresponding x-coordinates.
    baseline : dict
        Output of :func:`co2_fingers.baseline.compute_static_baseline`.
    widths_info : dict
        Output of :func:`co2_fingers.fingers.measure_finger_widths`.
    filename : str
        Frame filename for the title.
    y_top, y_bot, x_left, x_right, scale : see :func:`plot_baseline_frame`.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    from .baseline import interpolate_baseline

    sigma  = baseline["sigma_h"]
    prom   = baseline["prominence"]

    interp  = interpolate_baseline(baseline, x_vals)
    y_bar_i = interp["y_bar"]
    upper_i = interp["upper_band"]
    lower_i = interp["lower_band"]

    n      = widths_info["n_fingers"]
    mean_w = widths_info["mean_width_cm"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={"wspace": 0.05})
    fig.suptitle(f"{filename}  --  N = {n}   sigma_h = {sigma:.1f} px   prom = {prom:.1f} px   "
                 f"mean width = {mean_w:.1f} cm",
                 fontsize=11, fontweight="bold")

    cropped = image[y_top:y_bot, x_left:x_right]
    disp    = cv2.resize(cropped, (0, 0), fx=scale, fy=scale)
    rgb     = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    x_sc    = x_vals * scale

    axes[0].imshow(rgb)
    axes[0].plot(x_sc, smooth_iface * scale, color="cyan",   lw=1,   label="Gaussian outline")
    axes[0].plot(x_sc, y_bar_i * scale,      color="yellow", lw=1.5, ls="--",
                 label="static baseline")
    axes[0].fill_between(x_sc, lower_i * scale, upper_i * scale,
                         color="yellow", alpha=0.15, label="+-2.5 sigma static band")
    axes[0].set_title("(A) Image + overlay", fontsize=10)
    axes[0].axis("off")
    axes[0].legend(fontsize=6, loc="lower right")

    axes[1].plot(x_vals, smooth_iface, color="cyan",   lw=1, label="smooth interface")
    axes[1].plot(x_vals, y_bar_i,      color="yellow", lw=1.5, ls="--", label="static baseline")
    axes[1].fill_between(x_vals, lower_i, upper_i,
                         color="yellow", alpha=0.18, label="+-2.5 sigma static band")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="x (pixels)", ylabel="y (pixels)",
                title="(B) 1-D interface + finger widths")
    axes[1].legend(fontsize=6, loc="lower right")
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------

def plot_finger_widths_per_frame(
    times_sec: np.ndarray,
    mean_widths_cm: np.ndarray,
    std_widths_cm: np.ndarray | None = None,
    title: str = "Average finger width over time",
    time_unit: str = "h",
    t_bottom: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Line plot of mean finger width vs. time, with optional +-1 sigma shading.

    Parameters
    ----------
    times_sec : np.ndarray
        Elapsed time in seconds.
    mean_widths_cm : np.ndarray
        Mean finger width per frame (cm).
    std_widths_cm : np.ndarray or None
        Standard deviation per frame for shading (optional).
    title : str
    time_unit : str
        ``'h'`` for hours (default) or ``'s'`` for seconds.
    t_bottom : float or None
        If provided (seconds), draws a vertical "first finger hits bottom" line.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    t, xlabel = _to_hours(times_sec, time_unit)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean_widths_cm, color="#1f77b4", lw=1.8, label="mean width (cm)")
    if std_widths_cm is not None:
        ax.fill_between(t,
                        mean_widths_cm - std_widths_cm,
                        mean_widths_cm + std_widths_cm,
                        alpha=0.3, label="+-1 sigma")
    if t_bottom is not None:
        tb = t_bottom / _SECONDS_PER_HOUR if time_unit == "h" else t_bottom
        ax.axvline(tb, color="#8B0000", lw=1.8, ls="-", zorder=4)
        ax.text(tb, ax.get_ylim()[1], "first finger\nhits bottom",
                color="#8B0000", fontsize=7, ha="left", va="top",
                rotation=0, fontweight="bold")
    ax.set(xlabel=xlabel, ylabel="Finger width (cm)", title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


def plot_finger_count(
    times_sec: np.ndarray,
    n_fingers: np.ndarray,
    title: str = "Finger count over time",
    time_unit: str = "h",
    t_bottom: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Line plot of finger count vs. time.

    Parameters
    ----------
    times_sec, n_fingers : np.ndarray
    title : str
    time_unit : str
        ``'h'`` for hours (default) or ``'s'`` for seconds.
    t_bottom : float or None
        If provided (seconds), draws a vertical "first finger hits bottom" line.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    t, xlabel = _to_hours(times_sec, time_unit)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, n_fingers, color="#2ca02c", lw=1.8,
            marker="o", markersize=3, label="N fingers")
    if t_bottom is not None:
        tb = t_bottom / _SECONDS_PER_HOUR if time_unit == "h" else t_bottom
        ax.axvline(tb, color="#8B0000", lw=1.8, ls="-", zorder=4)
        ax.text(tb, ax.get_ylim()[1], "first finger\nhits bottom",
                color="#8B0000", fontsize=7, ha="left", va="top",
                fontweight="bold")
    ax.set(xlabel=xlabel, ylabel="Finger count N", title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


def plot_merging_metric(
    times_sec: np.ndarray,
    n_fingers: np.ndarray,
    mean_widths_cm: np.ndarray,
    title: str = "Merging metric: N x mean width",
    time_unit: str = "h",
    t_bottom: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the merging metric ``N_fingers x mean_width_cm`` vs. time.

    Parameters
    ----------
    times_sec, n_fingers, mean_widths_cm : np.ndarray
    title : str
    time_unit : str
        ``'h'`` for hours (default) or ``'s'`` for seconds.
    t_bottom : float or None
        If provided (seconds), draws a vertical "first finger hits bottom" line.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    t, xlabel = _to_hours(times_sec, time_unit)
    metric = n_fingers * mean_widths_cm
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, metric, color="#d62728", lw=1.8, label="N x mean width (cm)")
    if t_bottom is not None:
        tb = t_bottom / _SECONDS_PER_HOUR if time_unit == "h" else t_bottom
        ax.axvline(tb, color="#8B0000", lw=1.8, ls="-", zorder=4)
        ax.text(tb, ax.get_ylim()[1], "first finger\nhits bottom",
                color="#8B0000", fontsize=7, ha="left", va="top",
                fontweight="bold")
    ax.set(xlabel=xlabel, ylabel="N x mean width (cm)", title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


def plot_loglog_tip(
    times_sec: np.ndarray,
    tip_depths_px: np.ndarray,
    title: str = "Log-log tip penetration depth",
    time_unit: str = "h",
    t_bottom: float | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Log-log plot of deepest finger-tip depth vs. time with reference slopes.

    Reference lines:
    - slope 0.5  -> diffusive (sqrt(t)) growth
    - slope 1.0  -> convective (linear) growth

    Parameters
    ----------
    times_sec, tip_depths_px : np.ndarray
    title : str
    time_unit : str
        ``'h'`` for hours (default) or ``'s'`` for seconds.
    t_bottom : float or None
        If provided (seconds), draws a vertical "first finger hits bottom" line.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    t, xlabel = _to_hours(times_sec, time_unit)
    valid = (t > 0) & (tip_depths_px > 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(t[valid], tip_depths_px[valid],
              color="#3C3489", lw=1.8, label="Tip depth (px)")

    t_ref   = np.array([t[valid].min(), t[valid].max()])
    tip_mid = np.nanmedian(tip_depths_px[valid])
    ax.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 0.5,
              ls="--", color="#888780", lw=1, label="slope 0.5 (diffusive)")
    ax.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 1.0,
              ls=":",  color="#639922", lw=1, label="slope 1.0 (convective)")

    if t_bottom is not None:
        tb = t_bottom / _SECONDS_PER_HOUR if time_unit == "h" else t_bottom
        ax.axvline(tb, color="#8B0000", lw=1.8, ls="-", zorder=4)
        ylims = ax.get_ylim()
        ax.text(tb, ylims[1], "first finger\nhits bottom",
                color="#8B0000", fontsize=7, ha="left", va="top",
                fontweight="bold")

    xlabel_log = xlabel.replace("Time", "Time (log)").replace("(h)", "(h, log)").replace("(s)", "(s, log)")
    ax.set(xlabel=xlabel_log, ylabel="Tip depth (px, log)", title=title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    return fig


def plot_heatmap_overlay(
    images: list,
    filenames: list[str] | None = None,
    y_top: int = 2300,
    y_bot: int = 3500,
    x_left: int = 0,
    x_right: int = 6000,
    figsize: tuple = (20, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Convenience wrapper -- delegates to :func:`co2_fingers.heatmap.build_heatmap_overlay`.
    """
    from .heatmap import build_heatmap_overlay
    build_heatmap_overlay(images, filenames, y_top, y_bot, x_left, x_right,
                          figsize=figsize, save_path=save_path)


def plot_time_regimes(detector, save_path: str | None = None) -> plt.Figure:
    """
    Convenience wrapper -- calls :meth:`TimeRegimeDetector.plot`.

    Parameters
    ----------
    detector : TimeRegimeDetector
        An already-fitted detector (``detect()`` must have been called).
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    detector.plot(save_path=save_path)
