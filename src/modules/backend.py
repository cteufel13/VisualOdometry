import cv2
import numpy as np
from config.config import VOConfig


def detect_features(
    img: np.ndarray, config: VOConfig, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detects SIFT features and computes descriptors.

    Args:
        img: Grayscale image.
        config: VO configuration.
        mask: Optional mask (255 = keep, 0 = ignore) for sky/car hood.

    Returns:
        Tuple of (keypoints_2d_array, descriptors).
    """
    # create sift detector
    sift = cv2.SIFT_create(
        nfeatures=config.num_features,
        contrastThreshold=config.sift_contrast_thresh,
        edgeThreshold=config.sift_edge_thresh,
    )

    # detect and compute (Now accepts mask!)
    keypoints, descriptors = sift.detectAndCompute(img, mask)

    if not keypoints or descriptors is None:
        return np.empty((0, 2)), np.empty((0, 128))

    # convert keypoints to numpy (N, 2)
    pts = np.float32([kp.pt for kp in keypoints])

    return pts, descriptors


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
