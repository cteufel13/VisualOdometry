import numpy as np

from config import config
from modules.feature_matching import detect_keypoints_and_descriptors, match_descriptors
from modules.initialization import estimate_pose_2d_to_2d, triangulate_points
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def update_landmark_database(
    vo_state: VOState,
    last_keyframe_vo_state: VOState,
    landmark_db: LandmarkDatabase,
    inlier_lm_db_indices: np.ndarray,
    next_track_id: int,
    K: np.ndarray,
) -> tuple[LandmarkDatabase, VOState, int]:
    """
    Update landmark database: add new landmarks and filter old ones.

    Steps:
    1. Update observation counts for matched landmarks (using inlier indices)
    2. Filter low-quality landmarks based on last seen threshold
    3. Detect and match features between current and last keyframe
    4. Estimate pose and triangulate candidate 3D points
    5. Check keyframe ratio: baseline distance / median depth
    6. If ratio exceeds threshold, skip adding landmarks (return last keyframe state)
    7. Otherwise, add new triangulated landmarks and update keyframe state

    Args:
        vo_state: Current VO state (after PnP) with pose R, t and image
        last_keyframe_vo_state: Previous keyframe VO state with pose R, t and image
        landmark_db: Current landmark database
        inlier_lm_db_indices: Indices into landmark database for PnP inliers
        next_track_id: Next available track ID for new landmarks
        K: 3x3 camera intrinsic matrix

    Returns:
        updated_landmark_db: Updated LandmarkDatabase with new/filtered landmarks
        keyframe_vo_state: Either vo_state (if new keyframe) or last_keyframe_vo_state
        next_track_id: Updated next available track ID

    """
    landmark_db = update_observation_counts(landmark_db, inlier_lm_db_indices)
    landmark_db = filter_landmarks(landmark_db)

    kp1, d1 = detect_keypoints_and_descriptors(vo_state.img)
    kp2, d2 = detect_keypoints_and_descriptors(last_keyframe_vo_state.img)

    matches = match_descriptors(d1, d2)
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    _, _, inliers = estimate_pose_2d_to_2d(kp1[idx1], kp2[idx2], K)
    points_3d = triangulate_points(
        kp1[idx1][inliers],
        kp2[idx2][inliers],
        vo_state.R,
        vo_state.t,
        last_keyframe_vo_state.R,
        last_keyframe_vo_state.t,
        K,
    )

    keyframe_ratio = np.linalg.norm(vo_state.t - last_keyframe_vo_state.t) / np.median(
        points_3d[:, 2]
    )

    # don't add any new landmarks
    if keyframe_ratio < config.KEYFRAME_RATIO_THRESH:
        return landmark_db, last_keyframe_vo_state, next_track_id

    landmark_db, next_track_id = add_new_landmarks(
        landmark_db, points_3d, d1[idx1][inliers], next_track_id
    )

    return landmark_db, vo_state, next_track_id


def update_observation_counts(
    landmark_db: LandmarkDatabase,
    inlier_matched_indices: np.ndarray,
) -> LandmarkDatabase:
    """
    Update observation counts for landmarks based on matching results.

    For landmarks that were successfully matched as inliers, reset their
    'last_seen_n_frames_ago' counter to 0. For all other landmarks in the
    database, increment their counter by 1.

    Args:
        landmark_db: Current landmark database
        inlier_matched_indices: (M,) indices of landmarks that were matched as inliers

    Returns:
        updated_db: Database with updated observation counts

    """
    all_indices = np.arange(len(landmark_db.last_seen_n_frames_ago))
    inverted_indices = np.setdiff1d(all_indices, inlier_matched_indices)
    landmark_db.last_seen_n_frames_ago[inlier_matched_indices] = 0
    landmark_db.last_seen_n_frames_ago[inverted_indices] += 1

    return landmark_db


def filter_landmarks(
    landmark_db: LandmarkDatabase,
    vo_state: VOState | None = None,
    K: np.ndarray | None = None,
) -> LandmarkDatabase:
    """
    Remove low-quality landmarks from database.

    Filtering criterion:
    - Landmarks not seen for MAX_LAST_SEEN_FRAMES or more frames are removed

    Note: The vo_state and K parameters are optional for future extensions
    (e.g., visibility checks, depth filtering) but are not currently used.

    Args:
        landmark_db: Current landmark database
        vo_state: Current VO state (optional, for future visibility checks)
        K: Camera intrinsic matrix (optional, for future visibility checks)

    Returns:
        filtered_db: Filtered landmark database with only recently observed landmarks

    """
    keep_mask = landmark_db.last_seen_n_frames_ago < config.MAX_LAST_SEEN_FRAMES

    landmark_db.landmarks_3d = landmark_db.landmarks_3d[keep_mask]
    landmark_db.descriptors = landmark_db.descriptors[keep_mask]
    landmark_db.track_ids = landmark_db.track_ids[keep_mask]
    landmark_db.last_seen_n_frames_ago = landmark_db.last_seen_n_frames_ago[keep_mask]

    return landmark_db


def add_new_landmarks(
    landmark_db: LandmarkDatabase,
    new_landmarks_3d: np.ndarray,
    new_descriptors: np.ndarray,
    next_track_id: int,
) -> tuple[LandmarkDatabase, int]:
    """
    Add new landmarks to the database.

    Appends new 3D landmark positions, descriptors, and track IDs to the database.
    Track IDs are assigned sequentially starting from next_track_id. The
    'last_seen_n_frames_ago' counter is initialized to 0 for all new landmarks.

    Args:
        landmark_db: Current landmark database
        new_landmarks_3d: (K, 3) new 3D landmark positions in world coordinates
        new_descriptors: (K, D) feature descriptors for new landmarks
        next_track_id: Next available track ID

    Returns:
        updated_db: Updated landmark database with new landmarks appended
        next_track_id: Updated next available track ID (incremented by K)

    """
    landmark_db.landmarks_3d = np.vstack([landmark_db.landmarks_3d, new_landmarks_3d])
    landmark_db.descriptors = np.vstack([landmark_db.descriptors, new_descriptors])
    new_track_ids = np.arange(next_track_id, next_track_id + len(new_landmarks_3d))
    landmark_db.track_ids = np.hstack([landmark_db.track_ids, new_track_ids])
    new_last_seen = np.zeros(
        len(new_landmarks_3d), dtype=landmark_db.last_seen_n_frames_ago.dtype
    )
    landmark_db.last_seen_n_frames_ago = np.hstack(
        [landmark_db.last_seen_n_frames_ago, new_last_seen]
    )

    return landmark_db, new_track_ids[-1] + 1
