import cv2
import numpy as np

from config import config
from modules.bundle_adjustment import bundle_adjustment_ceres
from modules.feature_matching import detect_keypoints_and_descriptors, match_descriptors
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def initialize_vo_from_two_frames(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
) -> tuple[VOState, LandmarkDatabase, bool]:
    """
    Bootstrap VO from first two frames using 2D-to-2D motion estimation.

    Steps:
    1. Detect features in both images
    2. Match features between images
    3. Compute relative pose using Essential matrix (5-point algorithm + RANSAC)
    4. Triangulate initial 3D points
    5. Create initial VOState and LandmarkDatabase

    Args:
        img1: First image
        img2: Second image
        K: 3x3 camera calibration matrix


    Returns:
        vo_state: Initial VOState
        landmark_db: Initial LandmarkDatabase
        success: True if initialization successful

    """
    kp1, d1 = detect_keypoints_and_descriptors(img1)
    kp2, d2 = detect_keypoints_and_descriptors(img2)

    matches = match_descriptors(d1, d2)
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    R, t, inliers = estimate_pose_2d_to_2d(kp1[idx1], kp2[idx2], K)
    points_3d = triangulate_points(
        kp1[idx1][inliers], kp2[idx2][inliers], np.eye(3), np.zeros(3), R, t, K
    )

    keyframe_ratio = np.linalg.norm(t) / np.median(points_3d[:, 2])
    if keyframe_ratio > config.KEYFRAME_RATIO_THRESH:
        return None, None, False

    # Prepare observations
    inlier_kp1 = kp1[idx1][inliers]
    inlier_kp2 = kp2[idx2][inliers]
    observations = [inlier_kp1, inlier_kp2]

    # Bundle adjustment
    camera_poses = [(np.eye(3), np.zeros(3)), (R, t)]
    points_3d_refined, poses_refined = bundle_adjustment_ceres(
        points_3d, observations, camera_poses, K
    )

    # Extract refined camera 2 pose
    R_refined, t_refined = poses_refined[1]
    return (
        VOState(
            R_refined,
            t_refined,
            0,
            img2,
            np.arange(points_3d_refined.shape[0]),
            inlier_kp2,
        ),
        LandmarkDatabase(
            points_3d_refined,
            d1[idx1][inliers],
            np.arange(points_3d_refined.shape[0]),
            np.ones(points_3d_refined.shape[0]),
        ),
        True,
    )


def estimate_pose_2d_to_2d(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate relative pose from 2D-2D correspondences using Essential matrix.

    Uses 5-point algorithm with RANSAC for robustness.

    Args:
        kpts1: (N, 2) keypoints in first image
        kpts2: (N, 2) keypoints in second image
        K: 3x3 camera calibration matrix

    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector (unit scale)
        inlier_mask: (N,) boolean array indicating inliers

    """
    # Ensure we have at least 5 points (minimal case for essential matrix)
    if kpts1.shape[0] < 5:
        "Need at least 5 point correspondences"

    # Compute Essential Matrix using 5-point algorithm with RANSAC
    # cv2.findEssentialMat uses Nister's 5-point algorithm internally
    E, mask = cv2.findEssentialMat(
        kpts1,
        kpts2,
        K,
        method=cv2.RANSAC,
        prob=config.RANSAC_PROB,  # Confidence that at least one sample is outlier-free
        threshold=config.RANSAC_THRESH_PIXELS,  # Max reprojection error in pixels for inlier
    )

    # Convert mask to boolean array
    inlier_mask = mask.ravel().astype(bool)

    # Recover pose (R, t) from Essential matrix
    # recoverPose tests all 4 possible solutions and returns the correct one
    # by checking which solution has points in front of both cameras
    _, R, t, pose_mask = cv2.recoverPose(E, kpts1, kpts2, K, mask=mask)

    # Update inlier mask with points that also pass the cheirality check
    # (points must be in front of both cameras)
    inlier_mask = inlier_mask & pose_mask.ravel().astype(bool)

    return R, t, inlier_mask


def triangulate_points(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points from 2D correspondences and camera poses.

    Uses linear triangulation (DLT) with SVD solution.

    Args:
        kpts1: (N, 2) keypoints in first image
        kpts2: (N, 2) keypoints in second image
        R1, t1: First camera pose (rotation and translation)
        R2, t2: Second camera pose
        K: 3x3 camera calibration matrix

    Returns:
        points_3d: (N, 3) array of triangulated 3D points

    """
    N = kpts1.shape[0]

    # Build projection matrices: M = K @ [R | t]
    # t should be a column vector (3, 1)
    t1 = t1.reshape(3, 1)
    t2 = t2.reshape(3, 1)

    M1 = K @ np.hstack([R1, t1])  # (3, 4)
    M2 = K @ np.hstack([R2, t2])  # (3, 4)

    points_3d = np.zeros((N, 3))

    for i in range(N):
        u1, v1 = kpts1[i]
        u2, v2 = kpts2[i]

        # Build the A matrix from the constraint p x (M·P) = 0
        # For each view, the cross product gives us:
        #   u * m3^T - m1^T = 0
        #   v * m3^T - m2^T = 0
        # where m1, m2, m3 are the rows of M

        A = np.array(
            [
                u1 * M1[2, :] - M1[0, :],
                v1 * M1[2, :] - M1[1, :],
                u2 * M2[2, :] - M2[0, :],
                v2 * M2[2, :] - M2[1, :],
            ]
        )  # (4, 4)

        # Solve A @ P = 0 via SVD
        # The solution is the right singular vector corresponding to smallest singular value
        _, _, Vt = np.linalg.svd(A)
        P_homogeneous = Vt[-1, :]  # Last row of V^T

        # Convert from homogeneous to Euclidean coordinates
        points_3d[i] = P_homogeneous[:3] / P_homogeneous[3]

    return points_3d
