from dataclasses import dataclass

import numpy as np


@dataclass
class VOState:
    """Current state of the visual odometry system."""

    R: np.ndarray  # 3x3 rotation matrix (current camera orientation)
    t: np.ndarray  # 3x1 translation vector (current camera position)
    frame_id: int  # current frame number
    img: np.ndarray  # Image when RT are logged with frame id
    matched_track_ids: np.ndarray  # N, track ids of matched points to the lm_db
    matched_keypoints_2d: np.ndarray

    def get_homogenous_tf(self) -> np.ndarray:
        """
        Construct the 4x4 homogeneous transformation matrix from R and t.

        The transformation matrix T represents the camera pose in the form:
        T = [R  t]
            [0  1]

        where R is the 3x3 rotation matrix and t is the 3x1 translation vector.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()  # Ensure t is a 1D array for assignment
        return T
