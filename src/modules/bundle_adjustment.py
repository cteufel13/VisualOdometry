import numpy as np
import pyceres
from scipy.spatial.transform import Rotation
from collections import defaultdict
import copy
from state.vo_state import VOState
from state.landmark_database import LandmarkDatabase
from typing import List
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
        for point_idx, uv in keypoints:
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



def bundle_adjustment_sw(vo_state_buffer: List[VOState], landmark_db: LandmarkDatabase, K: np.ndarray):
    """
    Sliding window bundle adjustment

    Args:
        vo_state_buffer: List of VOState objects for the sliding window
        landmark_db: LandmarkDatabase containing 3D landmarks
        K: 3x3 intrinsic matrix

    Returns:
        updated_landmark_db: landmark_db with refined 3dpoints
        updated_vo_list: updated VOstates with refined R,t
    """

    # Match observations to each tracked feature
    observation_map = defaultdict(list)
    camera_poses = []

    for frame_idx, vo_state in enumerate(vo_state_buffer):
        for i, track_id in enumerate(vo_state.matched_track_ids):
            point_2d = vo_state.matched_keypoints_2d[i]
            observation_map[track_id].append((frame_idx, point_2d))

        camera_poses.append((vo_state.R, vo_state.t))

    # Filter tracks seen in at least 2 frames
    track_ids = list(observation_map.keys())
    track_ids = [track_id for track_id in track_ids if len(observation_map[track_id]) >= 2]

    # Get corresponding 3D points from landmark database
    mask = np.isin(landmark_db.track_ids, track_ids)
    points_3d = landmark_db.landmarks_3d[mask]
    filtered_track_ids = landmark_db.track_ids[mask]

    # Validate 3D points before proceeding
    valid_point_mask = np.all(np.isfinite(points_3d), axis=1)
    if not np.all(valid_point_mask):
        print(f"Warning: Found {(~valid_point_mask).sum()} invalid 3D points, filtering them out")
        points_3d = points_3d[valid_point_mask]
        filtered_track_ids = filtered_track_ids[valid_point_mask]

    # Build mapping from track_id to point_idx in points_3d array
    track_id_to_point_idx = {track_id: idx for idx, track_id in enumerate(filtered_track_ids)}

    # Convert observation_map to format expected by bundle_adjustment_ceres
    # observations[cam_idx] = [(point_idx, uv), ...]
    observations = [[] for _ in range(len(vo_state_buffer))]

    for track_id in filtered_track_ids:
        point_idx = track_id_to_point_idx[track_id]
        for frame_idx, point_2d in observation_map[track_id]:
            observations[frame_idx].append((point_idx, point_2d))

    # Check if there are any observations left
    total_observations = sum(len(obs) for obs in observations)
    if total_observations < 4:
        print(f"Warning: Too few observations ({total_observations}), skipping BA")
        return landmark_db, list(vo_state_buffer)

    # Run bundle adjustment (don't fix first camera in sliding window)
    updated_3d_points, poses_refined = bundle_adjustment_ceres(
        points_3d=points_3d,
        observations=observations,
        camera_poses=camera_poses,
        K=K,
        fix_first_camera=False  # Optimize all cameras in sliding window
    )

    # Update landmark database with optimized points
    updated_landmark_db = copy.deepcopy(landmark_db)
    # Create mask for landmarks that were actually optimized
    optimized_mask = np.isin(landmark_db.track_ids, filtered_track_ids)
    updated_landmark_db.landmarks_3d[optimized_mask] = updated_3d_points

    updated_vo_list = [vo_state for vo_state in vo_state_buffer ]
    for i,(R,t) in enumerate(poses_refined):
        updated_vo_list[i].R = R
        updated_vo_list[i].t = t

    return updated_landmark_db,updated_vo_list

    
