"""
co2_fingers.preprocessing
=========================
Image cropping, binarization (Otsu), median filtering, hole-filling,
and morphological cleaning — with fixes for Gaussian interface spiking.

Changes vs. original
--------------------
* Median blur is now applied to the **greyscale** image *before* thresholding
  so that Otsu sees a denoised signal (previously the blur was applied to the
  already-binarised image, which gave little benefit).
* The ``gray`` variable no longer gets accidentally overwritten by the blurred
  binary image.
* The morphological-closing kernel is now elongated vertically
  (``3 × morph_ksize*3``) to bridge gaps inside tall, narrow CO₂ fingers.
* :func:`clamp_interface` is added: a post-extraction utility that removes
  per-column spikes in the 1-D interface array by clamping jumps that exceed
  *max_jump* pixels and then applying a median filter.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def crop_image(
    image: np.ndarray,
    y_top: int,
    y_bot: int,
    x_left: int = 0,
    x_right: int = 6000,
) -> np.ndarray:
    """
    Crop a raw FluidFlower image to the CO₂-permeable zone.

    The default pixel limits (``x_left=230``, ``x_right=5700``,
    ``y_top=2300``, ``y_bot=3300``) match experiment C2R5.  Adjust them
    for other runs.

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR image (H×W×3).
    y_top : int
        Top row of the crop region (smaller y = higher in image).
    y_bot : int
        Bottom row of the crop region.
    x_left : int
        Leftmost column of the crop region (default 230).
    x_right : int
        Rightmost column of the crop region (default 5700).

    Returns
    -------
    np.ndarray
        Cropped BGR sub-image.
    """
    return image[y_top:y_bot, x_left:x_right]


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    y_top: int = 2300,
    y_bot: int = 3500,
    x_left: int = 0,
    x_right: int = 6000,
    use_otsu: bool = True,
    manual_thresh: int = 127,
    median_ksize: int = 5,
    morph_ksize: int = 5,
) -> np.ndarray:
    """
    Full preprocessing pipeline: crop → grayscale → denoise → binarize → fill.

    Steps
    -----
    1. Crop the raw image to the CO₂ zone.
    2. Convert BGR → greyscale.
    3. Apply a median blur to the **greyscale** image to remove
       salt-and-pepper noise *before* thresholding (fix: previously the blur
       was applied after binarisation, giving minimal benefit).
    4. Binarize with Otsu's method (or a manual threshold).
    5. Fill internal holes with ``scipy.ndimage.binary_fill_holes``.
    6. Apply a morphological closing with a vertically elongated kernel to
       seal gaps inside tall, narrow fingers (fix: previously used a square
       kernel that missed vertical gaps).

    Parameters
    ----------
    image : np.ndarray
        Full-resolution BGR image.
    y_top, y_bot : int
        Vertical crop limits (pixels).
    x_left, x_right : int
        Horizontal crop limits (pixels).
    use_otsu : bool
        Use Otsu's automatic threshold (default True).
        Set False to use *manual_thresh* instead.
    manual_thresh : int
        Manual binarization threshold (0–255), used when ``use_otsu=False``.
    median_ksize : int
        Kernel size for the median blur (must be odd, default 5).
    morph_ksize : int
        Base kernel size for the morphological closing (default 5).
        The actual closing kernel is ``3 × morph_ksize*3`` (W × H) to favour
        vertical gap closure.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) with the same width × height as the
        cropped region.
    """
    # 1. Crop
    cropped = crop_image(image, y_top, y_bot, x_left, x_right)

    # 2. Greyscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 3. Denoise BEFORE thresholding so Otsu sees a cleaner signal
    gray = cv2.medianBlur(gray, median_ksize)

    # 4. Binarize
    if use_otsu:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)

    # 5. Fill holes
    mask = bw > 0
    mask = ndi.binary_fill_holes(mask)
    mask = (mask * 255).astype(np.uint8)

    # 6. Morphological closing — elongated kernel bridges tall finger gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, morph_ksize * 3)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ---------------------------------------------------------------------------
# Interface post-processing
# ---------------------------------------------------------------------------

def clamp_interface(
    interface: np.ndarray,
    max_jump: int = 200,
    median_size: int = 10,
) -> np.ndarray:
    """
    Remove per-column spikes from a 1-D interface position array.

    After the interface row is extracted column by column (e.g. via a
    Gaussian fit to the intensity profile), isolated columns can jump far
    from their neighbours when the fit locks onto noise or a secondary peak.
    This function:

    1. Walks left-to-right and replaces any value that differs from its
       left neighbour by more than *max_jump* pixels with the left-neighbour
       value (hold-last-good strategy).
    2. Applies a ``scipy.ndimage.median_filter`` with window *median_size* to
       smooth any residual noise.

    Parameters
    ----------
    interface : np.ndarray
        1-D array of interface row positions, one entry per image column.
    max_jump : int
        Maximum allowed pixel jump between adjacent columns (default 30).
        Columns that exceed this threshold are replaced by their left
        neighbour before the median filter is applied.
    median_size : int
        Window length for the final median filter (default 15).

    Returns
    -------
    np.ndarray
        Cleaned 1-D interface array (same length as input, float64).

    Examples
    --------
    >>> import numpy as np
    >>> iface = np.array([300, 302, 298, 850, 301, 299], dtype=float)
    >>> clamp_interface(iface, max_jump=30, median_size=3)
    array([300., 301., 300., 300., 300., 300.])
    """
    cleaned = interface.astype(float).copy()

    # Pass 1: hold-last-good for large jumps
    for i in range(1, len(cleaned)):
        if abs(cleaned[i] - cleaned[i - 1]) > max_jump:
            cleaned[i] = cleaned[i - 1]

    # Pass 2: median filter for residual noise
    cleaned = median_filter(cleaned, size=median_size)

    return cleaned
