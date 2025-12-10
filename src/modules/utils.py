from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

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


def extract_trajectory_positions(trajectory: list[VOState]) -> np.ndarray:
    """
    Extract camera positions from trajectory.

    Args:
        trajectory: List of VOState objects

    Returns:
        positions: (N, 3) array of camera positions

    """
    if not trajectory:
        return np.empty((0, 3))
    return np.array([state.t.flatten() for state in trajectory])


def compute_trajectory_length(trajectory: list[VOState]) -> float:
    """
    Compute total length of camera trajectory.

    Args:
        trajectory: List of VOState objects

    Returns:
        length: Total trajectory length (m)

    """
    if len(trajectory) < 2:
        return 0.0

    positions = extract_trajectory_positions(trajectory)
    # euclidian distance
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def save_trajectory(trajectory: list[VOState], filename: str) -> None:
    """
    Save trajectory to file in TUM format (timestamp tx ty tz qx qy qz qw).

    Args:
        trajectory: List of VOState objects
        filename: Output filename

    """
    with Path.open(filename, "w") as f:
        for state in trajectory:
            t = state.t.flatten()
            # rotation matrix to quaternion
            quat = R_scipy.from_matrix(state.R).as_quat()

            f.write(
                f"{state.timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )


def load_trajectory(filename: str) -> list[VOState]:
    """
    Load trajectory from file.

    Args:
        filename: Input filename

    Returns:
        trajectory: List of VOState objects

    """
    trajectory = []
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    for row in data:
        timestamp = row[0]
        t = row[1:4].reshape(3, 1)
        quat = row[4:8]  # x,y,z,w
        R_mat = R_scipy.from_quat(quat).as_matrix()

        trajectory.append(
            VOState(
                R=R_mat,
                t=t,
                frame_id=0,
                timestamp=timestamp,
            )
        )
    return trajectory


def visualize_trajectory_2d(trajectory: list[VOState]) -> None:
    """
    Visualize trajectory in 2D (top-down view).

    Args:
        trajectory: List of VOState objects

    """
    pass


def visualize_trajectory_3d(
    trajectory: list[VOState], landmark_db: LandmarkDatabase
) -> None:
    """
    Visualize trajectory and landmarks in 3D.

    Args:
        trajectory: List of VOState objects
        landmark_db: Landmark database to visualize

    """
    pass


def load_camera_calibration_from_file(filepath: str) -> np.ndarray:
    """
    Load camera calibration from file.

    Expected format: fx fy cx cy (one per line or space-separated)

    Args:
        filepath: Path to calibration file

    Returns:
        K: 3x3 camera calibration matrix

    """
    pass


def create_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Create 3x3 camera calibration matrix K.

    Args:
        fx, fy: focal lengths
        cx, cy: principal point

    Returns:
        K: 3x3 camera matrix

    """
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
