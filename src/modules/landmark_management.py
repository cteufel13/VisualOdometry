import numpy as np

from src.state.landmark_database import LandmarkDatabase
from src.state.vo_state import VOState


def update_landmark_database(
    vo_state: VOState,
    vo_state_prev: VOState,
    landmark_db: LandmarkDatabase,
    inlier_mask: np.ndarray,
    matched_landmark_indices: np.ndarray,
    new_keypoints_curr: np.ndarray | None,
    new_descriptors_curr: np.ndarra | None,
    new_keypoints_prev: np.ndarray | None,
    K: np.ndarray,
) -> LandmarkDatabase:
    """
    Update landmark database: add new landmarks and filter old ones.

    Steps:
    1. Update observation counts for matched landmarks (using inlier_mask)
    2. Check if current frame should be a keyframe
    3. If keyframe: triangulate new landmarks from unmatched features
    4. Filter low-quality landmarks
    5. Return updated database

    Args:
        vo_state: Current VO state (after PnP)
        vo_state_prev: Previous VO state
        landmark_db: Current landmark database
        inlier_mask: Boolean mask from PnP indicating which matches are inliers
        matched_landmark_indices: Indices into landmark database for the matches
        new_keypoints_curr: Unmatched keypoints in current frame (for triangulation)
        new_descriptors_curr: Descriptors for unmatched keypoints in current frame
        new_keypoints_prev: Corresponding keypoints in previous frame (for triangulation)
        K: 3x3 camera matrix

    Returns:
        updated_landmark_db: Updated LandmarkDatabase

    """
    pass


def should_create_keyframe(
    vo_state_curr: VOState,
    vo_state_prev: VOState,
    num_matches: int,
    num_landmarks: int,
) -> bool:
    """
    Decide whether current frame should be a keyframe.

    Keyframe creation criteria:
    - Sufficient baseline (translation) between frames
    - Sufficient rotation between frames
    - Number of tracked features dropping below threshold

    Args:
        vo_state_curr: Current VO state
        vo_state_prev: Previous VO state
        num_matches: Number of matched features
        num_landmarks: Total number of landmarks in database

    Returns:
        is_keyframe: True if should create keyframe

    """
    pass


def add_new_landmarks(
    landmark_db: LandmarkDatabase,
    new_landmarks_3d: np.ndarray,
    new_descriptors: np.ndarray,
    next_track_id: int,
) -> tuple[LandmarkDatabase, int]:
    """
    Add new landmarks to the database.

    Args:
        landmark_db: Current landmark database
        new_landmarks_3d: (K, 3) new 3D landmark positions
        new_descriptors: (K, D) descriptors for new landmarks
        next_track_id: Next available track ID

    Returns:
        updated_db: Updated landmark database
        next_track_id: Updated next available track ID

    """
    pass


def filter_landmarks(
    landmark_db: LandmarkDatabase,
    min_observations: int = 3,
    vo_state: VOState | None = None,
    K: np.ndarray | None = None,
) -> LandmarkDatabase:
    """
    Remove low-quality landmarks from database.

    Filtering criteria:
    - Too few observations
    - Behind camera (if vo_state and K provided)
    - Too far from camera

    Args:
        landmark_db: Current landmark database
        min_observations: Minimum number of observations to keep landmark
        vo_state: Current VO state (optional, for visibility check)
        K: Camera matrix (optional, for visibility check)

    Returns:
        filtered_db: Filtered landmark database

    """
    pass


def compute_triangulation_quality(
    points_3d: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    Compute quality metric for triangulated points.

    Quality based on:
    - Baseline distance between camera positions
    - Angle between viewing rays (parallax angle)

    Args:
        points_3d: (N, 3) triangulated 3D points
        R1, t1: First camera pose
        R2, t2: Second camera pose

    Returns:
        quality_scores: (N,) quality metric for each point (higher is better)

    """
    pass


def update_observation_counts(
    landmark_db: LandmarkDatabase,
    matched_indices: np.ndarray,
    inlier_mask: np.ndarray,
) -> LandmarkDatabase:
    """
    Update observation counts for landmarks that were successfully matched.

    Only increment for inlier matches.

    Args:
        landmark_db: Current landmark database
        matched_indices: (M,) indices of landmarks that were matched
        inlier_mask: (M,) boolean mask indicating which matches are inliers

    Returns:
        updated_db: Database with updated observation counts

    """
    pass
