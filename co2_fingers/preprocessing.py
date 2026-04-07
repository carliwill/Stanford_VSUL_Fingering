"""
co2_fingers.preprocessing
=========================
Image cropping, binarization (Otsu), median filtering, hole-filling,
and morphological cleaning — exactly as implemented in the original notebooks.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi


def crop_image(image: np.ndarray, y_top: int, y_bot: int,
               x_left: int = 230, x_right: int = 5700) -> np.ndarray:
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


def preprocess(
    image: np.ndarray,
    y_top: int = 2300,
    y_bot: int = 3300,
    x_left: int = 230,
    x_right: int = 5700,
    use_otsu: bool = True,
    manual_thresh: int = 127,
    median_ksize: int = 5,
    morph_ksize: int = 5,
) -> np.ndarray:
    """
    Full preprocessing pipeline: crop → grayscale → binarize → denoise → fill.

    Steps (identical to the original notebooks):

    1. Crop the raw image to the CO₂ zone.
    2. Convert BGR → greyscale.
    3. Binarize with Otsu's method (or a manual threshold).
    4. Apply a median blur to remove salt-and-pepper noise.
    5. Fill internal holes with ``scipy.ndimage.binary_fill_holes``.
    6. Apply a morphological closing to seal small gaps.

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
        Kernel size for the morphological closing (default 5).

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255) with the same width × height as the
        cropped region.
    """
    cropped = crop_image(image, y_top, y_bot, x_left, x_right)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if use_otsu:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)

    gray = cv2.medianBlur(bw, median_ksize)
    mask = gray > 0
    mask = ndi.binary_fill_holes(mask)
    mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
