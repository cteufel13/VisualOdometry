import numpy as np
import pyceres
from scipy.spatial.transform import Rotation
from collections import defaultdict
import copy
from state.vo_state import VOState
from state.landmark_database import LandmarkDatabase
from typing import List


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
    Bundle adjustment using Ceres with analytical Jacobians
    """
    # Convert (R, t) to axis-angle + translation
    camera_params = []
    for R, t in camera_poses:
        rvec = Rotation.from_matrix(R).as_rotvec()
        params = np.concatenate([rvec, t.flatten()])
        camera_params.append(params)

    points = copy.deepcopy(points_3d)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ... [ReprojectionCost class definition remains the same] ...
    class ReprojectionCost(pyceres.CostFunction):
        def __init__(self, observed_x, observed_y, fx, fy, cx, cy):
            super().__init__()
            self.set_num_residuals(2)
            self.set_parameter_block_sizes([6, 3])  # [camera params, 3D point]
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
            
            # Convert rotation vector to rotation matrix
            R = Rotation.from_rotvec(rvec).as_matrix()

            # Transform point to camera coordinates
            P_cam = R @ point + t

            # Check if point is behind camera
            if P_cam[2] <= 0:
                residuals[:] = [1e6, 1e6]
                if jacobians is not None:
                    if jacobians[0] is not None:
                        jacobians[0][:] = np.zeros(12)  # 2 residuals * 6 parameters
                    if jacobians[1] is not None:
                        jacobians[1][:] = np.zeros(6)   # 2 residuals * 3 parameters
                return True

            # Project to image plane
            x_proj = self.fx * P_cam[0] / P_cam[2] + self.cx
            y_proj = self.fy * P_cam[1] / P_cam[2] + self.cy

            # Compute residuals
            residuals[0] = x_proj - self.observed_x
            residuals[1] = y_proj - self.observed_y

            # Compute Jacobians if requested
            if jacobians is not None:
                X, Y, Z = P_cam[0], P_cam[1], P_cam[2]
                Z2 = Z * Z

                # Jacobian of projection w.r.t. camera coordinates
                dproj_dPcam_x = np.array([self.fx / Z, 0, -self.fx * X / Z2])
                dproj_dPcam_y = np.array([0, self.fy / Z, -self.fy * Y / Z2])

                # Jacobian w.r.t. camera parameters [rvec, t]
                if jacobians[0] is not None:
                    # Compute d(R)/d(rvec) using Rodrigues formula
                    theta = np.linalg.norm(rvec)
                    
                    if theta < 1e-8:
                        dR_drvec = self._dR_drvec_small_angle(point)
                    else:
                        dR_drvec = self._dR_drvec_rodrigues(rvec, point, R)
                    
                    dPcam_drvec = dR_drvec
                    dPcam_dt = np.eye(3)
                    
                    jac_camera = np.vstack([
                        np.concatenate([dproj_dPcam_x @ dPcam_drvec, dproj_dPcam_x @ dPcam_dt]),
                        np.concatenate([dproj_dPcam_y @ dPcam_drvec, dproj_dPcam_y @ dPcam_dt])
                    ])
                    jacobians[0][:] = jac_camera.ravel()

                # Jacobian w.r.t. 3D point
                if jacobians[1] is not None:
                    dPcam_dpoint = R
                    jac_point = np.vstack([
                        dproj_dPcam_x @ dPcam_dpoint,
                        dproj_dPcam_y @ dPcam_dpoint
                    ])
                    jacobians[1][:] = jac_point.ravel()

            return True

        def _dR_drvec_small_angle(self, point):
            px, py, pz = point
            return np.array([
                [0, pz, -py],
                [-pz, 0, px],
                [py, -px, 0]
            ])

        def _dR_drvec_rodrigues(self, rvec, point, R):
            theta = np.linalg.norm(rvec)
            w = rvec / theta
            wx = np.array([
                [0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]
            ])
            
            dR_drvec = np.zeros((3, 3))
            
            for i in range(3):
                dtheta_drvec_i = rvec[i] / theta
                dw_drvec_i = np.zeros(3)
                dw_drvec_i[i] = 1.0 / theta
                dw_drvec_i -= w * dtheta_drvec_i / theta
                
                dwx_drvec_i = np.array([
                    [0, -dw_drvec_i[2], dw_drvec_i[1]],
                    [dw_drvec_i[2], 0, -dw_drvec_i[0]],
                    [-dw_drvec_i[1], dw_drvec_i[0], 0]
                ])
                
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                
                d_sin_theta_over_theta = (cos_theta * theta - sin_theta) / (theta * theta)
                d_one_minus_cos_over_theta2 = (sin_theta * theta * theta - 2 * theta * (1 - cos_theta)) / (theta ** 4)
                
                dR_i = (d_sin_theta_over_theta * dtheta_drvec_i * wx + 
                        sin_theta / theta * dwx_drvec_i +
                        d_one_minus_cos_over_theta2 * dtheta_drvec_i * (wx @ wx) +
                        (1 - cos_theta) / (theta * theta) * (dwx_drvec_i @ wx + wx @ dwx_drvec_i))
                
                dR_drvec[:, i] = dR_i @ point

            return dR_drvec

    problem = pyceres.Problem()

    # --- FIX: Explicitly add all camera parameter blocks first ---
    # This ensures "camera_params[0]" exists in the problem even if it has no observations.
    for params in camera_params:
        problem.add_parameter_block(params, 6)
    # -------------------------------------------------------------

    # Add residual blocks
    for cam_idx, keypoints in enumerate(observations):
        for point_idx, uv in keypoints:
            cost = ReprojectionCost(uv[0], uv[1], fx, fy, cx, cy)
            problem.add_residual_block(
                cost,
                pyceres.HuberLoss(3.0),
                [camera_params[cam_idx], points[point_idx]],
            )

    if fix_first_camera:
        # Now this is safe because we called add_parameter_block(camera_params[0], 6) above
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
        fix_first_camera=True  # Optimize all cameras in sliding window
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

    
