import numpy as np

from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def create_empty_landmark_database() -> LandmarkDatabase:
    """Create an empty landmark database."""
    return LandmarkDatabase(
        landmarks_3d=np.empty((0, 3), dtype=np.float64),
        descriptors=np.empty((0, 0), dtype=np.float32),
        track_ids=np.empty(0, dtype=np.int32),
        num_observations=np.empty(0, dtype=np.int32),
    )


def get_homogeneous_transform(vo_state: VOState) -> np.ndarray:
    """
    Convert VOState to 4x4 homogeneous transformation matrix.

    Args:
        vo_state: VOState object

    Returns:
        T: 4x4 transformation matrix [R|t; 0 0 0 1]

    """
    T = np.eye(4)
    T[:3, :3] = vo_state.R
    T[:3, 3] = vo_state.t.flatten()
    return T
