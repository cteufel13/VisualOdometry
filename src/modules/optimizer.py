import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions
from scipy.spatial.transform import Rotation
import cv2


def bundle_adjustment_window(
    keyframes: list[dict],
    all_points_dict: dict[int, np.ndarray],
    K: np.ndarray,
    fixed_window_size: bool = False,
) -> tuple[list[dict], dict[int, np.ndarray], int]:
    cam_params_global = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    def build_problem(current_kfs, current_pts, cam_params_array, use_robust_loss=True):
        problem = pyceres.Problem()
        loss = pyceres.CauchyLoss(1.0) if use_robust_loss else None

        qs = []
        ts = []
        point_params = {}

        # track which poses are actually used in the graph
        active_pose_indices = set()
        cam_params_added = False

        # create pose objects
        for kf in current_kfs:
            R = kf["T_cw"][:3, :3]
            t = kf["T_cw"][:3, 3]
            q = Rotation.from_matrix(R).as_quat()
            qs.append(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
            ts.append(t.astype(np.float64))

        # add residuals
        for i, kf in enumerate(current_kfs):
            ids = kf["ids"]
            obs = kf["pts_2d"]

            for j, pt_id in enumerate(ids):
                if pt_id not in current_pts:
                    continue

                # create point param if needed
                if pt_id not in point_params:
                    point_params[pt_id] = current_pts[pt_id].copy().astype(np.float64)

                cost = pycolmap.cost_functions.ReprojErrorCost(
                    pycolmap.CameraModelId.PINHOLE, obs[j]
                )

                problem.add_residual_block(
                    cost, loss, [qs[i], ts[i], point_params[pt_id], cam_params_array]
                )

                # mark as active
                active_pose_indices.add(i)
                cam_params_added = True

        # set manifolds for active poses
        for i in active_pose_indices:
            problem.set_manifold(qs[i], pyceres.QuaternionManifold())

        # set constants
        if cam_params_added:
            problem.set_parameter_block_constant(cam_params_array)

        if fixed_window_size and len(active_pose_indices) > 0:
            # fix oldest keyframe (origin)
            problem.set_parameter_block_constant(qs[0])
            problem.set_parameter_block_constant(ts[0])

        return problem, qs, ts, point_params

    # probe for outlier rejection
    problem, qs, ts, point_params = build_problem(
        keyframes, all_points_dict, cam_params_global, True
    )

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.max_num_iterations = 10
    options.initial_trust_region_radius = 1.0
    options.max_trust_region_radius = 1.0

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    # yeet bad points
    bad_points = set()
    for i, kf in enumerate(keyframes):
        q_curr = qs[i]
        R_curr = Rotation.from_quat(
            [q_curr[1], q_curr[2], q_curr[3], q_curr[0]]
        ).as_matrix()
        t_curr = ts[i]

        pts_local, obs_local, pids_local = [], [], []
        for j, pid in enumerate(kf["ids"]):
            if pid in point_params:
                pts_local.append(point_params[pid])
                obs_local.append(kf["pts_2d"][j])
                pids_local.append(pid)

        if not pts_local:
            continue

        proj, _ = cv2.projectPoints(np.array(pts_local), R_curr, t_curr, K, None)
        errs = np.linalg.norm(proj.reshape(-1, 2) - np.array(obs_local), axis=1)

        for k, err in enumerate(errs):
            if err > 3.0:
                bad_points.add(pids_local[k])

    if len(bad_points) > 0:
        print(f"BA Pruning: Removing {len(bad_points)} outliers.")
        for pid in bad_points:
            all_points_dict.pop(pid, None)
            point_params.pop(pid, None)

    # refine
    problem, qs, ts, point_params = build_problem(
        keyframes, all_points_dict, cam_params_global, True
    )

    options.max_num_iterations = 20
    options.initial_trust_region_radius = 1e4
    options.max_trust_region_radius = 1e16

    pyceres.solve(options, problem, summary)

    # writa back
    for i in range(len(keyframes)):
        q_new = qs[i]
        r_new = Rotation.from_quat([q_new[1], q_new[2], q_new[3], q_new[0]]).as_matrix()
        keyframes[i]["T_cw"][:3, :3] = r_new
        keyframes[i]["T_cw"][:3, 3] = ts[i]

    for pt_id, val in point_params.items():
        all_points_dict[pt_id] = val

    return keyframes, all_points_dict, len(bad_points)
