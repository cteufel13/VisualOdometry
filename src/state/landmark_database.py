from dataclasses import dataclass

import numpy as np


@dataclass
class LandmarkDatabase:
    """Database of 3D landmarks with their descriptors."""

    landmarks_3d: np.ndarray  # (N, 3) array of 3D positions [X, Y, Z]
    descriptors: np.ndarray  # (N, D) array of feature descriptors
    track_ids: np.ndarray  # (N,) array of unique landmark IDs
    last_seen_n_frames_ago: (
        np.ndarray
    )  # (N,) array containing when this feature was last matched
