from dataclasses import dataclass, field

import numpy as np


@dataclass
class VOState:
    """Current state of the visual odometry system."""

    R: np.ndarray  # 3x3 rotation matrix (current camera orientation)
    t: np.ndarray  # 3x1 translation vector (current camera position)
    frame_id: int  # current frame number
    timestamp: float  # timestamp of current frame
    # keypoints that are being tracked but not yet triangulated
    candidate_keypoints: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    # position in the image in the beginning
    first_observation_keypoints: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2))
    )

    # camera pose in the beginning
    # store as (N, 3, 4) where [:, :, :3] is R and [:, :, 3] is t
    first_observation_poses: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3, 4))
    )

    # candidates descriptors
    candidate_descriptors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 32), dtype=np.uint8)
    )
