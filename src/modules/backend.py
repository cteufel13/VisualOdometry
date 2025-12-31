import cv2
import numpy as np

from config.config import VOConfig


def detect_new_features(
    img: np.ndarray, mask: np.ndarray | None, config: VOConfig
) -> np.ndarray:
    """
    Grid-based feature detection to force uniform distribution.

    Args:
        img: Grayscale image to detect features on.
        mask: Optional mask to block existing features.
        config: Configuration object containing grid settings.

    Returns:
        Array of shape (N, 2) containing new keypoints.

    """
    h, w = img.shape

    # base mask
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255

    # sky mask
    sky_cutoff = int(h * config.sky_percentage)
    mask[0:sky_cutoff, :] = 0

    n_rows = config.grid_rows
    n_cols = config.grid_cols
    features_per_cell = config.num_features // (n_rows * n_cols)

    cell_h = h // n_rows
    cell_w = w // n_cols

    all_keypoints = []

    for r in range(n_rows):
        for c in range(n_cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            # skip cells that are fully masked by optimization
            if np.sum(mask[y1:y2, x1:x2]) == 0:
                continue

            cell_mask = np.zeros_like(mask)
            cell_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

            current_quality = config.quality_level
            if r > n_rows // 2:
                current_quality = (
                    config.quality_level * 0.5
                )  # more sensitive on the lower side of the road

            cell_pts = cv2.goodFeaturesToTrack(
                img,
                mask=cell_mask,
                maxCorners=features_per_cell,
                qualityLevel=current_quality,
                minDistance=config.min_distance,
                blockSize=config.block_size,
                useHarrisDetector=True,
                k=0.04,
            )

            if cell_pts is not None:
                all_keypoints.append(cell_pts.reshape(-1, 2))

    if len(all_keypoints) > 0:
        return np.vstack(all_keypoints)
    return np.empty((0, 2))


def track_optical_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    config: VOConfig,
    flow_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust optical flow with optional forward-backward consistency check.

    Args:
        img1: Previous frame image.
        img2: Current frame image.
        pts1: Points in previous frame (N, 2).
        config: Configuration object.
        flow_guess: Optional flow prediction for motion compensation.

    Returns:
        Tuple containing tracked points (N, 2) and boolean validity mask (N,).

    """
    # forward tracking (a -> b)
    # use flow_guess if available
    flags = cv2.OPTFLOW_USE_INITIAL_FLOW if flow_guess is not None else 0
    next_pts = pts1.copy()
    if flow_guess is not None:
        next_pts += flow_guess

    pts2, status, err = cv2.calcOpticalFlowPyrLK(
        img1,
        img2,
        pts1,
        next_pts,
        winSize=config.lk_win_size,
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=flags,
    )

    # backward tracking (b -> a) to guard against drift
    if config.enable_fb_tracking:
        flags_back = cv2.OPTFLOW_USE_INITIAL_FLOW if flow_guess is not None else 0
        back_guess = pts1.copy() if flow_guess is not None else None

        pts1_rev, status_rev, _ = cv2.calcOpticalFlowPyrLK(
            img2,
            img1,
            pts2,
            back_guess,  # use original points as guess
            winSize=config.lk_win_size,
            maxLevel=config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03),
            flags=flags_back,
        )

        # check distance between original and back-tracked
        fb_dist = np.linalg.norm(pts1 - pts1_rev, axis=1)

        # valid if found in fwd, found in bwd, and error small
        valid = (
            (status.flatten() == 1)
            & (status_rev.flatten() == 1)
            & (fb_dist < config.fb_error_thresh)
        )
        return pts2, valid
    valid = status.flatten() == 1
    return pts2, valid


def triangulate_and_filter(
    T_cw1: np.ndarray,
    T_cw2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    config: VOConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Triangulates points and filters outliers with strict checks for rotation.

    Args:
        T_cw1: World-to-camera transform for frame 1.
        T_cw2: World-to-camera transform for frame 2.
        pts1: 2D points in frame 1 (N, 2).
        pts2: 2D points in frame 2 (N, 2).
        K: Camera intrinsic matrix.
        config: Configuration object.

    Returns:
        Tuple of 3D points (N, 3), validity mask (N,), and stats dict.

    """
    # 1. triangulate
    P1 = K @ T_cw1[:3, :]
    P2 = K @ T_cw2[:3, :]

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # 2. filter: chirality (points must be in front of both cameras)
    # transform points to cam 1 frame
    R1, t1 = T_cw1[:3, :3], T_cw1[:3, 3:4]
    pts_cam1 = (R1 @ pts3d.T + t1).T

    # transform points to cam 2 frame
    R2, t2 = T_cw2[:3, :3], T_cw2[:3, 3:4]
    pts_cam2 = (R2 @ pts3d.T + t2).T

    # check positive depth (z > 0.1 to avoid points exactly at camera center)
    valid_chirality = (pts_cam1[:, 2] > 0.1) & (pts_cam2[:, 2] > 0.1)

    # 3. filter: dynamic depth threshold
    # reject points that are unreasonably far compared to the baseline
    # if baseline is 0.5m, a point at 500m is numerical noise
    # 5. filter: bearing angle
    # Calculate Camera Centers Correctly FIRST
    R1, t1 = T_cw1[:3, :3], T_cw1[:3, 3:4]
    R2, t2 = T_cw2[:3, :3], T_cw2[:3, 3:4]

    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    # CORRECTED BASELINE CALCULATION
    # The physical distance the camera actually moved
    baseline = np.linalg.norm(C1 - C2)

    # RE-ORDERED: Now apply dynamic depth threshold using correct baseline
    # Relaxed floor: 50.0 -> 100.0 (safeguard for slow movement)
    max_depth = max(100.0 * baseline, 200.0)

    # Check positive depth (chirality) and max depth
    valid_depth_scale = (
        (pts_cam1[:, 2] > 0.1) & (pts_cam2[:, 2] > 0.1) & (pts_cam2[:, 2] < max_depth)
    )

    # 4. filter: bi-directional reprojection error
    # project back to image 2
    proj_pts2, _ = cv2.projectPoints(pts3d, R2, t2, K, None)
    proj_pts2 = proj_pts2.reshape(-1, 2)
    err2 = np.linalg.norm(proj_pts2 - pts2, axis=1)

    # project back to image 1
    proj_pts1, _ = cv2.projectPoints(pts3d, R1, t1, K, None)
    proj_pts1 = proj_pts1.reshape(-1, 2)
    err1 = np.linalg.norm(proj_pts1 - pts1, axis=1)

    # requires point to fit well in BOTH views
    valid_reproj = (err1 < config.max_reproj_err) & (err2 < config.max_reproj_err)

    # 5. filter: bearing angle
    # cam centers in world frame
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    v1 = pts3d - C1.T
    v2 = pts3d - C2.T

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    dot = np.sum(v1 * v2, axis=1)
    cos_angle = np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))

    valid_angle = angles > config.min_bearing_angle_deg

    # combine filters
    final_mask = valid_chirality & valid_depth_scale & valid_reproj & valid_angle

    # stats for logging
    survivors_errors = err2[final_mask]
    survivors_angles = angles[final_mask]

    stats = {
        "total": len(pts3d),
        "passed_chirality": np.sum(valid_chirality),
        "passed_depth_scale": np.sum(valid_depth_scale),
        "passed_reproj": np.sum(valid_reproj),
        "passed_angle": np.sum(valid_angle),
        "avg_error": np.mean(survivors_errors) if len(survivors_errors) > 0 else 0.0,
        "avg_angle": np.mean(survivors_angles) if len(survivors_angles) > 0 else 0.0,
        "final_keep": np.sum(final_mask),
    }

    return pts3d, final_mask, stats


def detect_features_orb(
    img: np.ndarray, config: VOConfig, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    ORB is often more robust on low-contrast driving datasets because it looks
    for corners (FAST) rather than gradient blobs.
    """
    orb = cv2.ORB_create(
        nfeatures=config.num_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=20,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=21,
        fastThreshold=10,
    )
    keypoints, descriptors = orb.detectAndCompute(img, mask)

    if not keypoints or descriptors is None:
        return np.empty((0, 2)), np.empty((0, 32))  # ORB desc are 32 bytes

    pts = np.float32([kp.pt for kp in keypoints])
    return pts, descriptors


def match_features(
    des_prev: np.ndarray, des_curr: np.ndarray, config: VOConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches descriptors between two frames using Brute Force + Ratio Test.

    Args:
        des_prev: Descriptors from previous frame (N, 128).
        des_curr: Descriptors from current frame (M, 128).
        config: VO Config.

    Returns:
        Tuple (matches_indices_prev, matches_indices_curr).
        These arrays are aligned: prev[i] matches curr[i].

    """
    if len(des_prev) == 0 or len(des_curr) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    # use bfmatcher with L2 norm (standard for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # knn match with k=2 to apply ratio test
    matches = bf.knnMatch(des_prev, des_curr, k=2)

    good_prev_idxs = []
    good_curr_idxs = []

    # apply lowe ratio test
    for m, n in matches:
        if m.distance < config.match_ratio_thresh * n.distance:
            good_prev_idxs.append(m.queryIdx)  # index in prev
            good_curr_idxs.append(m.trainIdx)  # index in curr

    return np.array(good_prev_idxs, dtype=int), np.array(good_curr_idxs, dtype=int)


def triangulate_and_filter(
    T_cw1: np.ndarray,
    T_cw2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    config: VOConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Triangulates points and filters outliers with strict checks for rotation.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool), {}

    # triangulate
    P1 = K @ T_cw1[:3, :]
    P2 = K @ T_cw2[:3, :]

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # filter by chirality
    R1, t1 = T_cw1[:3, :3], T_cw1[:3, 3:4]
    R2, t2 = T_cw2[:3, :3], T_cw2[:3, 3:4]

    pts_cam1 = (R1 @ pts3d.T + t1).T
    pts_cam2 = (R2 @ pts3d.T + t2).T

    # strict z > 0
    valid_chirality = (pts_cam1[:, 2] > 0.1) & (pts_cam2[:, 2] > 0.1)

    # filter by baseline and depth
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    baseline = np.linalg.norm(C1 - C2)

    # dynamic max depth based on baseline
    max_depth = max(100.0 * baseline, 200.0)

    valid_depth_scale = (
        (pts_cam1[:, 2] > 0.1) & (pts_cam2[:, 2] > 0.1) & (pts_cam2[:, 2] < max_depth)
    )

    # reprojection error
    proj_pts2, _ = cv2.projectPoints(pts3d, R2, t2, K, None)
    proj_pts2 = proj_pts2.reshape(-1, 2)
    err2 = np.linalg.norm(proj_pts2 - pts2, axis=1)

    proj_pts1, _ = cv2.projectPoints(pts3d, R1, t1, K, None)
    proj_pts1 = proj_pts1.reshape(-1, 2)
    err1 = np.linalg.norm(proj_pts1 - pts1, axis=1)

    valid_reproj = (err1 < config.max_reproj_err) & (err2 < config.max_reproj_err)

    # filter by camera ray angle
    v1 = pts3d - C1.T
    v2 = pts3d - C2.T
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    dot = np.sum(v1 * v2, axis=1)
    cos_angle = np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))

    valid_angle = angles > config.min_ray_angle_deg

    # combine filters
    final_mask = valid_chirality & valid_depth_scale & valid_reproj & valid_angle

    survivors_angles = angles[final_mask]

    stats = {
        "avg_angle": np.mean(survivors_angles) if len(survivors_angles) > 0 else 0.0,
        "final_keep": np.sum(final_mask),
    }

    return pts3d, final_mask, stats
