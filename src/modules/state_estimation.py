import numpy as np


def estimate_pose_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    initial_R: np.ndarray | None = None,
    initial_t: np.ndarray | None = None,
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
        initial_R: Initial guess for rotation (optional)
        initial_t: Initial guess for translation (optional)

    Returns:
        R: 3x3 estimated rotation matrix
        t: 3x1 estimated translation vector
        inlier_mask: (N,) boolean array indicating inliers
        reprojection_error: mean reprojection error of inliers (pixels)

    """
    pass


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
    pass


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
    pass


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
    pass
