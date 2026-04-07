"""
co2_fingers.io
==============
Time extraction from CSV data files and image loading utilities.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd


def load_timestamps(
    csv_path: str,
    filename_col: str = "Filename",
    time_col: str = "TimeSinceStart(min)",
) -> pd.DataFrame:
    """
    Load the image-timestamp CSV produced alongside a FluidFlower experiment.

    The CSV must contain at least a filename column and an elapsed-time column
    (in minutes).  Additional columns are preserved but not used internally.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    filename_col : str
        Name of the column containing image filenames (default 'Filename').
    time_col : str
        Name of the column containing elapsed time in minutes
        (default 'TimeSinceStart(min)').

    Returns
    -------
    pd.DataFrame
        DataFrame with at minimum ``filename_col``, ``time_col``, and a derived
        ``t_sec`` column (elapsed time in seconds).

    Example
    -------
    >>> df = load_timestamps("/data/image_last_modified.csv")
    >>> print(df.head())
    """
    df = pd.read_csv(csv_path)
    if filename_col not in df.columns:
        raise KeyError(f"Column '{filename_col}' not found in {csv_path}. "
                       f"Available columns: {list(df.columns)}")
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in {csv_path}. "
                       f"Available columns: {list(df.columns)}")
    df = df.rename(columns={filename_col: "filename", time_col: "time_min"})
    df["t_sec"] = df["time_min"] * 60.0
    return df


def fmt_time(t_sec: float) -> str:
    """
    Convert elapsed seconds to a human-readable string ``'Xh YYm'``.

    Parameters
    ----------
    t_sec : float
        Elapsed time in seconds.

    Returns
    -------
    str
        Formatted string, e.g. ``'2h 17m'``.
    """
    h = int(t_sec // 3600)
    m = int((t_sec % 3600) // 60)
    return f"{h}h {m:02d}m"


def load_image(path: str) -> np.ndarray:
    """
    Load a single image (BGR) from *path*, raising a clear error if missing.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file (JPG, PNG, etc.).

    Returns
    -------
    np.ndarray
        BGR image array of shape ``(H, W, 3)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or OpenCV cannot read it.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def load_images(
    image_dir: str,
    pattern: str = "*.JPG",
    sort: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """
    Load all images matching *pattern* inside *image_dir*.

    Parameters
    ----------
    image_dir : str
        Directory containing the image series.
    pattern : str
        Glob pattern for filenames (default ``'*.JPG'``).
    sort : bool
        If True (default), sort filenames lexicographically before loading.

    Returns
    -------
    list of (filename, ndarray)
        Each element is a ``(basename, BGR image array)`` tuple.
        Images that cannot be read are silently skipped with a warning printed
        to stdout.

    Example
    -------
    >>> pairs = load_images("/data/C2R5", pattern="IMG_12*.JPG")
    >>> filenames, images = zip(*pairs)
    """
    paths = sorted(glob.glob(os.path.join(image_dir, pattern))) if sort else \
            glob.glob(os.path.join(image_dir, pattern))
    result = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  WARNING: cannot read {p} — skipped")
            continue
        result.append((os.path.basename(p), img))
    return result
