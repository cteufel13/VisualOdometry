import cv2
import numpy as np

from config import config


def estimate_pose_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate camera pose from 3D-2D correspondences using PnP-RANSAC.

    Steps:
    1. Run P3P-RANSAC to find inliers and initial pose
    2. Refine pose using all inliers (nonlinear optimization)
    3. Compute reprojection error

    Args:
        points_3d: (N, 3) array of 3D landmark positions
        points_2d: (N, 2) array of corresponding 2D image points
        K: 3x3 camera calibration matrix

    Returns:
        R: 3x3 estimated rotation matrix
        t: 3x1 estimated translation vector
        inlier_mask: (N,) boolean array indicating inliers
        reprojection_error: mean reprojection error of inliers (pixels)

    Raises:
        ValueError: If fewer than 4 point correspondences are provided

    """
    # Ensure inputs are contiguous and correct dtype
    points_3d = np.ascontiguousarray(points_3d, dtype=np.float64)
    points_2d = np.ascontiguousarray(points_2d, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)

    # Extract camera intrinsics for cv2.solvePnPRansac
    # OpenCV expects distortion coefficients (use none for calibrated cameras)
    dist_coeffs = np.zeros(4)

    if len(points_3d) < 4:
        error_msg = (
            f"PnP requires at least 4 point correspondences, got {len(points_3d)}"
        )
        raise ValueError(error_msg)

    # Step 1: Run P3P-RANSAC to find inliers and initial pose
    success, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        iterationsCount=100,
        reprojectionError=config.RANSAC_THRESH_PIXELS,  # RANSAC threshold in pixels
        confidence=config.RANSAC_PROB,
        flags=cv2.SOLVEPNP_P3P,
    )

    if not success or inliers is None or len(inliers) < 4:
        # Fallback: return identity pose with no inliers
        R = np.eye(3)
        t = np.zeros((3, 1))
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        return R, t, inlier_mask, float("inf")

    # Create boolean inlier mask
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    inlier_mask[inliers.flatten()] = True

    # Convert RANSAC result to rotation matrix
    R_ransac, _ = cv2.Rodrigues(rvec_ransac)
    t_ransac = tvec_ransac.reshape(3, 1)

    # Step 2: Refine pose using all inliers with nonlinear optimization
    inlier_points_3d = points_3d[inlier_mask]
    inlier_points_2d = points_2d[inlier_mask]

    R, t = refine_pose_pnp(
        points_3d=inlier_points_3d,
        points_2d=inlier_points_2d,
        K=K,
        R_init=R_ransac,
        t_init=t_ransac,
    )

    # Step 3: Compute reprojection error for inliers
    reprojection_error = compute_reprojection_error(
        points_3d=inlier_points_3d, points_2d=inlier_points_2d, K=K, R=R, t=t
    )

    return R, t, inlier_mask, reprojection_error


def refine_pose_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine camera pose using nonlinear optimization (Levenberg-Marquardt).

    Minimizes reprojection error over all provided correspondences.

    Args:
        points_3d: (N, 3) array of 3D landmark positions
        points_2d: (N, 2) array of corresponding 2D image points
        K: 3x3 camera calibration matrix
        R_init: 3x3 initial rotation matrix
        t_init: 3x1 initial translation vector

    Returns:
        R: 3x3 refined rotation matrix
        t: 3x1 refined translation vector

    """
    # Ensure inputs are contiguous and correct dtype
    points_3d = np.ascontiguousarray(points_3d, dtype=np.float64)
    points_2d = np.ascontiguousarray(points_2d, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)

    # Convert initial rotation matrix to rotation vector
    rvec_init, _ = cv2.Rodrigues(R_init)
    tvec_init = t_init.reshape(3, 1)

    # No distortion for calibrated cameras
    dist_coeffs = np.zeros(4)

    # Use iterative refinement (Levenberg-Marquardt)
    success, rvec_refined, tvec_refined = cv2.solvePnP(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,  # Uses Levenberg-Marquardt
    )

    if not success:
        # Return initial guess if refinement fails
        return R_init, t_init

    # Convert back to rotation matrix
    R, _ = cv2.Rodrigues(rvec_refined)
    t = tvec_refined.reshape(3, 1)

    return R, t


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """
    Compute mean reprojection error for 3D-2D correspondences.

    Args:
        points_3d: (N, 3) array of 3D landmark positions
        points_2d: (N, 2) array of corresponding 2D image points
        K: 3x3 camera calibration matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        mean_error: mean L2 reprojection error in pixels

    """
    # Ensure inputs are contiguous and correct dtype
    points_3d = np.ascontiguousarray(points_3d, dtype=np.float64)
    points_2d = np.ascontiguousarray(points_2d, dtype=np.float64)
    K = np.ascontiguousarray(K, dtype=np.float64)

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # No distortion
    dist_coeffs = np.zeros(4)

    # Project 3D points to image
    projected_points, _ = cv2.projectPoints(
        objectPoints=points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
    )
    projected_points = projected_points.reshape(-1, 2)

    # Compute L2 distance in pixels
    errors = np.linalg.norm(points_2d - projected_points, axis=1)

    return np.mean(errors)
