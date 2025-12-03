import numpy as np
import pyceres
from scipy.spatial.transform import Rotation


def bundle_adjustment_ceres(
    points_3d: np.ndarray,  # (N, 3)
    observations: list[
        list[tuple[int, np.ndarray]]
    ],  # observations[cam_idx] = [(uv), ...]
    camera_poses: list[tuple[np.ndarray, np.ndarray]],  # List of (R, t) tuples
    K: np.ndarray,
    fix_first_camera: bool = True,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Bundle adjustment using Ceres

    Args:
        points_3d: Nx3 array of 3D points
        observations: observations[i] = [(uv), ...]
        camera_poses: List of (R, t) tuples where R is 3x3 and t is (3,)
        K: 3x3 intrinsic matrix
        fix_first_camera: Whether to fix the first camera pose

    Returns:
        points_refined: Optimized 3D points (Nx3)
        poses_refined: List of optimized (R, t) tuples

    """
    # Convert (R, t) to axis-angle + translation
    camera_params = []
    for R, t in camera_poses:
        rvec = Rotation.from_matrix(R).as_rotvec()
        params = np.concatenate([rvec, t.flatten()])
        camera_params.append(params)

    points = points_3d.copy()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Define custom cost function
    class ReprojectionCost(pyceres.CostFunction):
        def __init__(self, observed_x, observed_y, fx, fy, cx, cy):
            super().__init__()
            self.set_num_residuals(2)
            self.set_parameter_block_sizes([6, 3])
            self.observed_x = observed_x
            self.observed_y = observed_y
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

        def Evaluate(self, parameters, residuals, jacobians):
            camera = parameters[0]
            point = parameters[1]

            rvec = camera[:3]
            t = camera[3:6]
            R = Rotation.from_rotvec(rvec).as_matrix()

            P_cam = R @ point + t

            if P_cam[2] <= 0:
                residuals[:] = [1e6, 1e6]
                return True

            x_proj = self.fx * P_cam[0] / P_cam[2] + self.cx
            y_proj = self.fy * P_cam[1] / P_cam[2] + self.cy

            residuals[0] = x_proj - self.observed_x
            residuals[1] = y_proj - self.observed_y

            return True

    problem = pyceres.Problem()

    # Add residual blocks
    for cam_idx, keypoints in enumerate(observations):
        for point_idx, uv in enumerate(keypoints):
            cost = ReprojectionCost(uv[0], uv[1], fx, fy, cx, cy)
            problem.add_residual_block(
                cost,
                pyceres.HuberLoss(1.0),
                [camera_params[cam_idx], points[point_idx]],
            )

    if fix_first_camera:
        problem.set_parameter_block_constant(camera_params[0])

    # Solver options
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = False
    options.max_num_iterations = 100

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    # Convert back to (R, t) tuples
    poses_refined = []
    for params in camera_params:
        rvec = params[:3]
        t = params[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()
        poses_refined.append((R, t))

    return points, poses_refined
