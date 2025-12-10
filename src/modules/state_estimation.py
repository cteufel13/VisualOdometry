import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy


def estimate_pose_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    initial_R: np.ndarray | None = None,
    initial_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate camera pose from 3D-2D correspondences using PnP-RANSAC + Nonlinear Refinement.
    """
    if len(points_3d) < 4:
        return np.eye(3), np.zeros((3, 1)), np.zeros(len(points_3d), dtype=bool), 0.0

    # init with pnp ransac
    success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        K,
        distCoeffs=None,
        iterationsCount=100,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 5:
        return np.eye(3), np.zeros((3, 1)), np.zeros(len(points_3d), dtype=bool), 0.0

    # boolean mask
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    inlier_mask[inliers.ravel()] = True

    # convert rvec to rotation matrix
    R_est, _ = cv2.Rodrigues(rvec_est)
    t_est = tvec_est

    # nonlinear refinement if enough points
    if np.sum(inlier_mask) > 5:
        R_refined, t_refined = refine_pose_nonlinear(
            points_3d[inlier_mask], points_2d[inlier_mask], R_est, t_est, K
        )
    else:
        R_refined, t_refined = R_est, t_est

    # compute final reprojection error
    errors = compute_reprojection_error(
        points_3d[inlier_mask], points_2d[inlier_mask], R_refined, t_refined, K
    )
    mean_error = np.mean(errors)

    return R_refined, t_refined, inlier_mask, mean_error


def pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    num_iterations: int = 1000,
    reprojection_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust PnP using RANSAC with P3P minimal solver.

    Args:
        points_3d: (N, 3) array of 3D points
        points_2d: (N, 2) array of 2D points
        K: 3x3 camera matrix
        num_iterations: Number of RANSAC iterations
        reprojection_threshold: Inlier threshold in pixels

    Returns:
        R_best: 3x3 rotation matrix
        t_best: 3x1 translation vector
        inlier_mask: (N,) boolean array

    """
    pass


def p3p(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Minimal solver for Perspective-3-Point problem.

    Returns up to 4 solutions.

    Args:
        points_3d: (3, 3) array of 3D points
        points_2d: (3, 2) array of 2D image points
        K: 3x3 camera matrix

    Returns:
        solutions: List of (R, t) tuples (up to 4 solutions)

    """
    pass


def refine_pose_nonlinear(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine pose by minimizing reprojection error using nonlinear optimization.

    Args:
        points_3d: (N, 3) inlier 3D points
        points_2d: (N, 2) inlier 2D points
        R_init: 3x3 initial rotation
        t_init: 3x1 initial translation
        K: 3x3 camera matrix

    Returns:
        R_refined: 3x3 refined rotation
        t_refined: 3x1 refined translation

    """
    # angle axis rotation represenation
    r_vec_init, _ = cv2.Rodrigues(R_init)
    x0 = np.hstack((r_vec_init.ravel(), t_init.ravel()))

    def residuals(x: np.ndarray) -> np.ndarray:
        r_vec = x[:3]
        t_vec = x[3:].reshape(3, 1)
        R_curr, _ = cv2.Rodrigues(r_vec)

        # P = K(RX + t)
        X_cam = (R_curr @ points_3d.T + t_vec).T  # (N, 3)

        # avoid points behind camera
        z = X_cam[:, 2] + 1e-9
        u = K[0, 0] * X_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * X_cam[:, 1] / z + K[1, 2]

        proj_2d = np.stack([u, v], axis=1)
        return (proj_2d - points_2d).ravel()

    # reject outliers that RANSAC missed
    res = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0)

    r_vec_final = res.x[:3]
    t_vec_final = res.x[3:].reshape(3, 1)

    R_final, _ = cv2.Rodrigues(r_vec_final)

    return R_final, t_vec_final


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection error for each 3D-2D correspondence.

    Args:
        points_3d: (N, 3) 3D points
        points_2d: (N, 2) 2D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera matrix

    Returns:
        errors: (N,) array of reprojection errors in pixels

    """
    projected = project_points(points_3d, R, t, K)
    diff = projected - points_2d
    return np.linalg.norm(diff, axis=1)


def project_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Project 3D points to 2D image plane.

    Args:
        points_3d: (N, 3) 3D points in world frame
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        K: 3x3 camera matrix

    Returns:
        points_2d: (N, 2) projected 2D points

    """
    # T_cam = R * X + t
    X_cam = (R @ points_3d.T + t).T  # (N, 3)

    # perspective division
    z = X_cam[:, 2] + 1e-9  # avoid zero div

    # apply intrinsics
    # u = fx * x/z + cx
    # v = fy * y/z + cy
    u = K[0, 0] * X_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * X_cam[:, 1] / z + K[1, 2]

    return np.stack([u, v], axis=1)
