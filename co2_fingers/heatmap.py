"""
co2_fingers.heatmap
===================
Build and overlay grayscale difference heatmaps for a user-selected
subset of frames.

Matches the seismic difference-plot approach in ``valley_fingers_single``.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def build_heatmap_overlay(
    images: list,
    filenames: list[str] | None = None,
    y_top: int = 2300,
    y_bot: int = 3300,
    x_left: int = 230,
    x_right: int = 5700,
    cmap: str = "seismic",
    figsize: tuple = (20, 5),
    save_path: str | None = None,
) -> None:
    """
    Plot consecutive-frame grayscale differences as seismic-style heatmaps.

    For N images, N−1 difference panels are plotted side-by-side.  The
    colour scale is symmetric around zero and shared across all panels.

    Parameters
    ----------
    images : list of np.ndarray
        List of BGR images loaded with :func:`co2_fingers.io.load_image`.
        Must contain at least 2 images.
    filenames : list of str or None
        Optional filenames used as panel titles.  If None, panels are
        titled by index.
    y_top, y_bot, x_left, x_right : int
        Crop limits (pixels) matching the preprocessing parameters.
    cmap : str
        Matplotlib colormap for the difference images (default ``'seismic'``).
    figsize : tuple
        Figure size (default ``(20, 5)``).
    save_path : str or None
        If provided, save the figure to this path.

    Raises
    ------
    ValueError
        If fewer than 2 images are supplied.
    """
    if len(images) < 2:
        raise ValueError("At least 2 images are required to compute differences.")

    filenames = filenames or [f"frame_{i}" for i in range(len(images))]

    # Convert to grayscale floats and crop
    grays = []
    for img in images:
        cropped = img[y_top:y_bot, x_left:x_right]
        g = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(float)
        grays.append(g)

    diffs = [grays[i + 1] - grays[i] for i in range(len(grays) - 1)]
    v = max(np.max(np.abs(d)) for d in diffs)

    n = len(diffs)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, d in enumerate(diffs):
        im = axes[i].imshow(d, cmap=cmap, vmin=-v, vmax=v)
        axes[i].set_title(os.path.basename(filenames[i + 1]), fontsize=9)
        axes[i].axis("off")

    fig.colorbar(im, ax=axes[-1], label="ΔGrayscale intensity")
    plt.suptitle("Frame-difference heatmaps", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap figure saved → {save_path}")
    plt.show()


def overlay_interfaces(
    images: list,
    x_vals_list: list,
    interfaces_list: list,
    filenames: list[str] | None = None,
    colors: list[str] | None = None,
    y_top: int = 2300,
    y_bot: int = 3300,
    x_left: int = 230,
    x_right: int = 5700,
    figsize: tuple = (14, 6),
    title: str = "Interface overlay",
    save_path: str | None = None,
) -> None:
    """
    Overlay smoothed interfaces from multiple frames on a single image.

    Parameters
    ----------
    images : list of np.ndarray
        BGR images; the first image is used as the background.
    x_vals_list : list of np.ndarray
        Per-frame x-coordinates from :func:`co2_fingers.interface.gauss_outline`.
    interfaces_list : list of np.ndarray
        Per-frame smoothed interface arrays.
    filenames : list of str or None
        Labels for the legend.
    colors : list of str or None
        Line colours; defaults to matplotlib default cycle.
    y_top, y_bot, x_left, x_right : int
        Crop limits.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure.
    """
    import matplotlib.cm as cm

    filenames = filenames or [f"frame_{i}" for i in range(len(images))]
    if colors is None:
        colors = [f"C{i}" for i in range(len(images))]

    bg = cv2.cvtColor(images[0][y_top:y_bot, x_left:x_right], cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(bg)

    for i, (xs, ys) in enumerate(zip(x_vals_list, interfaces_list)):
        label = os.path.basename(filenames[i]) if filenames else f"frame {i}"
        ax.plot(xs, ys, color=colors[i], lw=1.2, label=label)

    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Interface overlay saved → {save_path}")
    plt.show()
