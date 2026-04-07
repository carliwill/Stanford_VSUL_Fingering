"""
co2_fingers.baseline
====================
Compute a static baseline (mean front + roughness band) from a single
reference frame, then interpolate it onto any later frame.

This matches the static-baseline approach used in the notebook
``time_regime_detector_v3``.
"""

import numpy as np
from .preprocessing import preprocess
from .interface import raw_outline, gauss_outline
from .fingers import mean_front, roughness_sigma


def compute_static_baseline(
    baseline_image,
    y_top: int = 2300,
    y_bot: int = 3300,
    x_left: int = 230,
    x_right: int = 5700,
    window_frac: float = 0.6,
    y_bar_offset: float = 5.0,
    roughness_k: float = 0.5,
    contour_epsilon: float = 0.0001,
    median_ksize: int = 9,
    gaussian_sigma: float = 1.0,
):
    """
    Build a static baseline from a single reference (pre-fingering) image.

    The function preprocesses the image, extracts the interface, fits a
    Savitzky–Golay mean front, and computes the roughness σ_h.
    A small *y_bar_offset* (default +5 px) is added to the baseline to
    match the original notebook's ``y_bar_static = y_bar_static + 5``.

    Parameters
    ----------
    baseline_image : np.ndarray
        Full-resolution BGR image of a baseline (flat-front) frame.
    y_top, y_bot, x_left, x_right : int
        Crop limits (pixels).  Defaults match C2R5 / C2R1.
    window_frac : float
        Fraction of signal length for Savitzky–Golay window (default 0.6).
    y_bar_offset : float
        Pixels added to y_bar after fitting (default 5.0, as in notebook).
    roughness_k : float
        Prominence multiplier: ``prom = roughness_k * sigma_h`` (default 0.5).
    contour_epsilon : float
        Contour approximation coefficient for ``approxPolyDP`` (default 0.0001).
    median_ksize : int
        Kernel size for median filter in :func:`gauss_outline` (default 9).
    gaussian_sigma : float
        Sigma for Gaussian smoothing in :func:`gauss_outline` (default 1.0).

    Returns
    -------
    dict with keys:
        - ``x_base``        : np.ndarray of x-coordinates for the baseline.
        - ``y_bar_static``  : np.ndarray, smoothed mean front (y_bar + offset).
        - ``smooth_iface``  : np.ndarray, raw smoothed interface values.
        - ``residual``      : np.ndarray, ``smooth_iface - y_bar_static``.
        - ``sigma_h``       : float, roughness σ_h (pixels).
        - ``prominence``    : float, finger-detection prominence threshold.
        - ``upper_band``    : np.ndarray, ``y_bar_static + 2.5*sigma_h``.
        - ``lower_band``    : np.ndarray, ``y_bar_static - 2.5*sigma_h``.
    """
    mask = preprocess(baseline_image, y_top, y_bot, x_left, x_right)
    smooth_cnt, _ = raw_outline(mask, contour_epsilon=contour_epsilon)
    x_base, smooth_iface = gauss_outline(smooth_cnt, median_ksize=median_ksize,
                                         gaussian_sigma=gaussian_sigma)
    x_base = np.array(x_base)

    y_bar = mean_front(smooth_iface, window_frac=window_frac)
    y_bar_static = y_bar + y_bar_offset

    residual = smooth_iface - y_bar_static
    sigma_h = roughness_sigma(smooth_iface, y_bar_static)
    prominence = roughness_k * sigma_h

    upper_band = y_bar_static + 2.5 * sigma_h
    lower_band = y_bar_static - 2.5 * sigma_h

    return {
        "x_base": x_base,
        "y_bar_static": y_bar_static,
        "smooth_iface": smooth_iface,
        "residual": residual,
        "sigma_h": sigma_h,
        "prominence": prominence,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }


def interpolate_baseline(baseline: dict, x_new: np.ndarray) -> dict:
    """
    Interpolate a baseline computed at ``baseline['x_base']`` onto *x_new*.

    Parameters
    ----------
    baseline : dict
        Output of :func:`compute_static_baseline`.
    x_new : np.ndarray
        New x-coordinate array (from a later frame's interface).

    Returns
    -------
    dict with keys ``y_bar``, ``upper_band``, ``lower_band`` interpolated
    onto *x_new*.
    """
    x_base = baseline["x_base"]
    return {
        "y_bar": np.interp(x_new, x_base, baseline["y_bar_static"]),
        "upper_band": np.interp(x_new, x_base, baseline["upper_band"]),
        "lower_band": np.interp(x_new, x_base, baseline["lower_band"]),
    }
