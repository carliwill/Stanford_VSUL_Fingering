"""
co2_fingers.interface
=====================
Extract the CO₂–brine interface from a binary mask and produce a
smoothed, single-valued y(x) representation.

Matches the ``raw_outline`` and ``gauss_outline`` functions in the
original notebooks exactly.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import medfilt


def raw_outline(mask: np.ndarray, contour_epsilon: float = 0.0001):
    """
    Find the largest contour in *mask* and apply ``approxPolyDP`` smoothing.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, 0/255) from :func:`co2_fingers.preprocessing.preprocess`.
    contour_epsilon : float
        Coefficient for ``cv2.approxPolyDP``.  Smaller values preserve more
        detail; the original notebooks used ``0.0001`` (default).

    Returns
    -------
    smooth_cnt : np.ndarray
        Smoothed contour array, shape ``(N, 1, 2)``.
    trace : np.ndarray
        Binary image of the drawn contour (same shape as *mask*).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    epsilon = contour_epsilon * cv2.arcLength(cnt, True)
    smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)
    trace = np.zeros_like(mask)
    cv2.drawContours(trace, [smooth_cnt], -1, 255, 6)
    return smooth_cnt, trace


def gauss_outline(
    smooth_cnt: np.ndarray,
    median_ksize: int = 9,
    gaussian_sigma: float = 1.0,
    flow_direction: str = "down",
):
    """
    Convert a contour to a single-valued smoothed interface y(x).

    For each unique x value in the contour, the function takes the
    *maximum* y (deepest point, i.e. furthest into the CO₂ zone when
    CO₂ sinks).  The resulting 1-D signal is median-filtered and then
    Gaussian-smoothed — exactly replicating the notebook implementation.

    Parameters
    ----------
    smooth_cnt : np.ndarray
        Output of :func:`raw_outline`.
    median_ksize : int
        Kernel size for ``scipy.signal.medfilt`` (must be odd, default 9).
    gaussian_sigma : float
        Sigma for ``scipy.ndimage.gaussian_filter1d`` (default 1.0).
    flow_direction : str
        ``'down'`` (CO₂ sinks — take max y) or ``'right'`` (lateral flow —
        take max x).  Default ``'down'``.

    Returns
    -------
    x_vals : np.ndarray
        1-D array of x-coordinates (pixels).
    smooth_interface : np.ndarray
        Smoothed interface y-values, same length as *x_vals*.
    """
    pts = smooth_cnt.squeeze()  # (N, 2)
    x_sorted = np.argsort(pts[:, 0])
    pts = pts[x_sorted]

    x_vals, y_vals = [], []
    for x in np.unique(pts[:, 0]):
        ys = pts[pts[:, 0] == x][:, 1]
        x_vals.append(x)
        if flow_direction == "down":
            y_vals.append(np.max(ys))   # lowest point in image (y increases downward)
        else:
            y_vals.append(np.max(pts[pts[:, 0] == x][:, 0]))

    y_clean = medfilt(y_vals, kernel_size=median_ksize)
    smooth_interface = ndi.gaussian_filter1d(y_clean, sigma=gaussian_sigma)
    return np.array(x_vals), smooth_interface
