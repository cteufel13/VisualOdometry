"""Frontend tracking logic"""

import numpy as np
from datatypes import Frame


def detect_features_grid(
    image: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Detect FAST features ensuring grid distribution.

    Args:
        image: Grayscale input image.
        mask: Optional exclusion mask.

    Returns:
        (N, 2) array of feature coordinates [u, v].

    """
    # TODO: Implement grid-based FAST detection
    # 1. Divide image into grid cells (defined in config).
    # 2. Detect features in each cell.
    # 3. Select top N features per cell to ensure distribution.
    return np.empty((0, 2))


def track_klt(
    img_prev: np.ndarray, img_curr: np.ndarray, px_prev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bi-directional KLT tracking.

    Args:
        img_prev: Previous image.
        img_curr: Current image.
        px_prev: Keypoints in previous image (N, 2).

    Returns:
        Tuple containing:
        - px_curr: Tracked points in current image (N, 2).
        - status: Boolean mask (N,) where True indicates a valid, robust track.

    """
    # TODO: Implement Bi-directional Optical Flow
    # 1. Calc Optical Flow Forward (Prev -> Curr).
    # 2. Calc Optical Flow Backward (Curr -> Prev).
    # 3. Check Euclidean distance between Original and Back-tracked.
    # 4. Filter points with error > 0.5px or status=0.
    return np.empty((0, 2)), np.empty((0,), dtype=bool)
