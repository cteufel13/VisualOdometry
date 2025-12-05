from dataclasses import dataclass

import numpy as np


@dataclass
class VOState:
    """Current state of the visual odometry system."""

    R: np.ndarray  # (3x3) rotation matrix (current camera orientation)
    t: np.ndarray  # (3x1) translation vector (current camera position)
    img: np.ndarray  # Image when RT are logged with frame id
    matched_track_ids: np.ndarray  # (Nx1) track ids of matched points to the lm_db
    matched_keypoints_2d: np.ndarray  # (Nx2) matched image keypoints
