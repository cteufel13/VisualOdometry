"""Geometric calculations."""

import numpy as np


def estimate_pose_pnp(
    px: np.ndarray, landmarks: np.ndarray, K: np.ndarray, guess_T: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera pose using PnP RANSAC + Refinement.

    Args:
        px: 2D image points (N, 2).
        landmarks: Corresponding 3D world points (N, 3).
        K: Intrinsic matrix (3, 3).
        guess_T: Initial pose guess (4, 4).

    Returns:
        Tuple containing:
        - T_cw: Refined 4x4 Pose matrix (World -> Camera).
        - inliers: Boolean mask (N,) indicating inliers.

    """
    # TODO: Implement Robust PnP
    # 1. Run solvePnPRansac using guess_T.
    # 2. Run solvePnPRefineLM on inliers.
    # 3. Return refined pose and inlier mask.
    return guess_T, np.zeros(len(px), dtype=bool)


def triangulate_points(
    T1: np.ndarray, T2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate points from two views using DLT.

    Args:
        T1: Pose of View 1 (World -> Camera).
        T2: Pose of View 2 (World -> Camera).
        pts1: 2D points in View 1.
        pts2: 2D points in View 2.
        K: Intrinsic matrix.

    Returns:
        Tuple containing:
        - points_3d: (N, 3) Triangulated points in World frame.
        - valid: Boolean mask (N,) for points passing Cheirality/Depth checks.

    """
    # TODO: Implement DLT Triangulation
    # 1. Construct Projection Matrices P1, P2.
    # 2. cv2.triangulatePoints.
    # 3. Check Cheirality (points must be in front of BOTH cameras).
    return np.empty((0, 3)), np.empty((0,), dtype=bool)
