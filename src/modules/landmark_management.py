import cv2
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
    3. Check if keyframe needed based on current tracking quality (# of matched landmarks)
    4. Detect and match features between current and last keyframe
    5. Triangulate candidate 3D points and validate quality
    6. Check keyframe ratio: baseline distance / median depth
    7. Add new triangulated landmarks if quality checks pass and update keyframe state

    Args:
        vo_state: Current VO state (after PnP) with pose R, t and image
        last_keyframe_vo_state: Previous keyframe VO state with pose R, t and image
        landmark_db: Current landmark database
        inlier_lm_db_indices: Indices of landmarks matched as PnP inliers in current frame
        next_track_id: Next available track ID for new landmarks
        K: 3x3 camera intrinsic matrix

    Returns:
        updated_landmark_db: Updated LandmarkDatabase with new/filtered landmarks
        keyframe_vo_state: Either vo_state (if new keyframe) or last_keyframe_vo_state
        next_track_id: Updated next available track ID

    """
    landmark_db = update_observation_counts(landmark_db, inlier_lm_db_indices)
    landmark_db = filter_landmarks(landmark_db)

    # Check if we need to force keyframe creation based on current tracking quality
    # Use the number of matched landmarks in this frame (PnP inliers), not total database size
    num_matched_landmarks = len(inlier_lm_db_indices)
    force_keyframe = num_matched_landmarks < config.MIN_LANDMARKS_FOR_TRACKING

    kp1, d1 = detect_keypoints_and_descriptors(vo_state.img)
    kp2, d2 = detect_keypoints_and_descriptors(last_keyframe_vo_state.img)

    matches = match_descriptors(d1, d2)
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    # Filter out matches that correspond to landmarks already in the database
    # to avoid adding duplicates
    new_feature_indices = filter_existing_landmarks(d1[idx1], landmark_db)

    if len(new_feature_indices) < 8:
        # Not enough new features to triangulate
        return landmark_db, last_keyframe_vo_state, next_track_id

    # Use filtered indices for triangulation
    kp1_new = kp1[idx1][new_feature_indices]
    kp2_new = kp2[idx2][new_feature_indices]
    d1_new = d1[idx1][new_feature_indices]

    # TODO: Future improvement - use matched keypoints from landmark database
    # to validate scale during triangulation. Currently we only use 2D-2D matches
    # between new keyframes, but we could also triangulate using existing 3D
    # landmarks and their 2D projections to get scale-consistent initialization.

    _, _, inliers = estimate_pose_2d_to_2d(kp1_new, kp2_new, K)
    points_3d = triangulate_points(
        kp1_new[inliers],
        kp2_new[inliers],
        vo_state.R,
        vo_state.t,
        last_keyframe_vo_state.R,
        last_keyframe_vo_state.t,
        K,
    )

    # Validate and filter new landmarks (depth, reprojection error, scale consistency)
    points_3d_valid, descriptors_valid, median_depth = (
        validate_and_filter_new_landmarks(
            points_3d,
            kp1_new[inliers],
            kp2_new[inliers],
            d1_new[inliers],
            vo_state,
            last_keyframe_vo_state,
            landmark_db,
            K,
        )
    )

    # Early return if no valid points after filtering
    if len(points_3d_valid) == 0:
        return landmark_db, last_keyframe_vo_state, next_track_id

    # Compute baseline distance between frames
    baseline_distance = np.linalg.norm(vo_state.t - last_keyframe_vo_state.t)
    keyframe_ratio = baseline_distance / median_depth

    # Adaptive keyframe selection:
    # - Force keyframe if matched landmarks are scarce
    # - Use relaxed threshold if below target landmark count
    # - Otherwise use standard threshold
    if not force_keyframe:
        threshold = config.KEYFRAME_RATIO_THRESH
        if num_matched_landmarks < config.TARGET_LANDMARKS:
            threshold *= 0.5  # Relax threshold when tracking quality is low

        if keyframe_ratio < threshold:
            return landmark_db, last_keyframe_vo_state, next_track_id

    landmark_db, next_track_id = add_new_landmarks(
        landmark_db, points_3d_valid, descriptors_valid, next_track_id
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
    vo_state: VOState | None = None,  # noqa: ARG001
    K: np.ndarray | None = None,  # noqa: ARG001
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


def validate_and_filter_new_landmarks(
    points_3d: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    descriptors: np.ndarray,
    vo_state: VOState,
    last_keyframe_vo_state: VOState,
    landmark_db: LandmarkDatabase,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Consolidated validation and filtering of newly triangulated landmarks.

    Performs all quality checks in one place:
    1. Depth constraints (MIN_TRIANGULATION_DEPTH < depth < MAX_TRIANGULATION_DEPTH)
    2. Reprojection error validation
    3. Soft scale consistency check with existing landmarks

    The scale consistency check uses depth concentration (Median Absolute Deviation)
    to distinguish between:
    - Real scene transitions (close->far): High scale change + low spread = ACCEPT
    - Bad triangulation: High scale change + high spread = REJECT

    Args:
        points_3d: (N, 3) triangulated 3D points in world coordinates
        kp1: (N, 2) keypoints in current frame
        kp2: (N, 2) keypoints in last keyframe
        descriptors: (N, D) descriptors for the points
        vo_state: Current VO state
        last_keyframe_vo_state: Last keyframe VO state
        landmark_db: Current landmark database
        K: 3x3 camera intrinsic matrix

    Returns:
        points_3d_valid: (M, 3) filtered 3D points
        descriptors_valid: (M, D) filtered descriptors
        median_depth: Median depth of valid points (for keyframe selection)

    """
    # 1. Filter by depth constraints
    depth_valid_mask = filter_triangulated_points(points_3d, vo_state.R, vo_state.t)

    if len(points_3d[depth_valid_mask]) == 0:
        return np.empty((0, 3)), np.empty((0, descriptors.shape[1])), 0.0

    # 2. Filter by reprojection error (only check depth-valid points)
    reproj_valid_mask = validate_reprojection_error(
        points_3d[depth_valid_mask],
        kp1[depth_valid_mask],
        kp2[depth_valid_mask],
        vo_state.R,
        vo_state.t,
        last_keyframe_vo_state.R,
        last_keyframe_vo_state.t,
        K,
        max_reproj_error=config.MAX_REPROJECTION_ERROR_NEW_LANDMARKS,
    )

    # Combine depth and reprojection masks
    combined_mask = depth_valid_mask.copy()
    combined_mask[depth_valid_mask] &= reproj_valid_mask

    points_3d_filtered = points_3d[combined_mask]
    descriptors_filtered = descriptors[combined_mask]

    # Return empty if no valid points
    if len(points_3d_filtered) == 0:
        return np.empty((0, 3)), np.empty((0, descriptors.shape[1])), 0.0

    # Calculate median depth for keyframe selection
    points_3d_cam = transform_to_camera_frame(
        points_3d_filtered, vo_state.R, vo_state.t
    )
    median_depth = np.median(points_3d_cam[:, 2])

    # 3. Soft scale consistency check with existing landmarks
    # This handles scenes where depth changes rapidly (close -> far or vice versa)
    if len(landmark_db.landmarks_3d) > 0:
        existing_depths_cam = transform_to_camera_frame(
            landmark_db.landmarks_3d, vo_state.R, vo_state.t
        )[:, 2]
        median_existing_depth = np.median(existing_depths_cam[existing_depths_cam > 0])

        scale_ratio = median_depth / median_existing_depth

        # If scale change is large, check if it's a legitimate scene transition
        # or bad triangulation by examining depth concentration
        if scale_ratio < 0.5 or scale_ratio > 2.0:
            # Calculate how concentrated the new landmark depths are
            # Low variance = consistent depth = likely real scene transition
            # High variance = scattered depth = likely bad triangulation
            new_depths = points_3d_cam[:, 2]

            # Use Median Absolute Deviation (MAD) normalized by median for robustness
            mad = np.median(np.abs(new_depths - median_depth))
            normalized_spread = mad / median_depth if median_depth > 0 else float("inf")

            # If depths are concentrated, accept the scale change
            # This indicates many points agree on the depth -> legitimate scene transition
            if normalized_spread > config.SCALE_CONSISTENCY_SPREAD_THRESHOLD:
                # High spread with scale change = likely bad triangulation
                # Reject all new landmarks
                return np.empty((0, 3)), np.empty((0, descriptors.shape[1])), 0.0

            # Otherwise: Low spread with scale change = accept as scene transition
            # (e.g., camera moved from close features to far features)

    # TODO: Future improvement - directional scale validation
    # Consider camera motion direction and spatial distribution of scale changes.
    # Example: If moving right and scale changes on right side of image, that's expected.
    # This would require tracking 2D keypoint positions and correlating with motion vector.

    return points_3d_filtered, descriptors_filtered, median_depth


def transform_to_camera_frame(
    points_3d_world: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D points from world frame to camera frame.

    Args:
        points_3d_world: (N, 3) points in world coordinates
        R_cw: 3x3 rotation matrix (world -> camera)
        t_cw: 3x1 translation vector (world -> camera)

    Returns:
        points_3d_cam: (N, 3) points in camera coordinates

    """
    # Ensure t is a column vector
    t_cw = t_cw.reshape(3, 1)

    # Transform: P_cam = R_cw @ P_world + t_cw
    return (R_cw @ points_3d_world.T).T + t_cw.T


def filter_triangulated_points(
    points_3d: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
) -> np.ndarray:
    """
    Filter out low-quality triangulated points.

    Filters based on:
    1. Points behind the camera (depth < 0)
    2. Points too close (depth < MIN_TRIANGULATION_DEPTH)
    3. Points too far (depth > MAX_TRIANGULATION_DEPTH)

    Args:
        points_3d: (N, 3) triangulated points in world coordinates
        R_cw: 3x3 rotation matrix (world -> camera)
        t_cw: 3x1 translation vector (world -> camera)

    Returns:
        valid_mask: (N,) boolean array indicating valid points

    """
    # Transform points to camera frame to get depth
    points_3d_cam = transform_to_camera_frame(points_3d, R_cw, t_cw)

    # Extract depth (Z coordinate in camera frame)
    depths = points_3d_cam[:, 2]

    # Filter based on depth constraints
    return (
        (depths > config.MIN_TRIANGULATION_DEPTH)  # Not too close or behind camera
        & (depths < config.MAX_TRIANGULATION_DEPTH)  # Not too far
    )


def validate_reprojection_error(
    points_3d: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
    max_reproj_error: float = 3.0,
) -> np.ndarray:
    """
    Validate triangulated points by checking reprojection error in both views.

    This ensures that newly triangulated landmarks are geometrically consistent
    and helps prevent scale drift by rejecting poorly triangulated points.

    Args:
        points_3d: (N, 3) triangulated 3D points in world coordinates
        kp1: (N, 2) keypoints in first view
        kp2: (N, 2) keypoints in second view
        R1, t1: Pose of first camera (world -> camera)
        R2, t2: Pose of second camera (world -> camera)
        K: 3x3 camera intrinsic matrix
        max_reproj_error: Maximum acceptable reprojection error in pixels

    Returns:
        valid_mask: (N,) boolean array indicating points with good reprojection

    """
    N = len(points_3d)
    valid_mask = np.ones(N, dtype=bool)

    # Validate reprojection in both views
    for R, t, kp in [(R1, t1, kp1), (R2, t2, kp2)]:
        # Convert rotation to rodrigues vector
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)

        # Project 3D points to image
        projected, _ = cv2.projectPoints(
            points_3d,
            rvec,
            tvec,
            K,
            distCoeffs=np.zeros(4),
        )
        projected = projected.reshape(-1, 2)

        # Compute reprojection error
        errors = np.linalg.norm(projected - kp, axis=1)

        # Mark points with large error as invalid
        valid_mask &= errors < max_reproj_error

    return valid_mask


def filter_existing_landmarks(
    descriptors: np.ndarray,
    landmark_db: LandmarkDatabase,
) -> np.ndarray:
    """
    Filter out keypoints that are likely already in the landmark database.

    This prevents adding duplicate landmarks by matching descriptors against
    the existing landmark database and rejecting those that match.

    Args:
        descriptors: (N, D) descriptors for candidate keypoints
        landmark_db: Current landmark database

    Returns:
        new_keypoint_indices: Indices of keypoints that are NOT in the database

    """
    if len(landmark_db.landmarks_3d) == 0:
        # No existing landmarks, all keypoints are new
        return np.arange(len(descriptors))

    # Match descriptors against database
    matches = match_descriptors(
        descriptors, landmark_db.descriptors, config.LOWES_RATIO
    )

    # Get indices of keypoints that matched existing landmarks
    matched_indices = set(matches[:, 0]) if len(matches) > 0 else set()

    # Keep only keypoints that didn't match
    unmatched_mask = np.ones(len(descriptors), dtype=bool)
    for idx in matched_indices:
        unmatched_mask[idx] = False

    return np.where(unmatched_mask)[0]
