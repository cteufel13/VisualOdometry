from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import tyro

from modules.dataset_loader import (
    BaseDataset,
    KittiDataset,
    MalagaDataset,
    OwnDataset,
    ParkingDataset,
)
from modules.feature_matching import extract_and_match_features
from modules.initialization import initialize_vo
from modules.landmark_management import (
    add_new_candidates,
    filter_landmarks,
    update_landmark_database,
)
from modules.state_estimation import estimate_pose_pnp, project_points
from modules.utils import (
    create_camera_matrix,
    create_empty_landmark_database,
    save_trajectory,
)
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def init_rerun() -> None:
    """Initialize Rerun logging with correct coordinate systems."""
    rr.init("Visual Odometry", spawn=True)

    # forward +Z, right +X, down +Y
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


def log_frame_rerun(
    image: np.ndarray,
    T_cw: np.ndarray,
    K: np.ndarray,
    frame_id: int,
    landmarks_3d: np.ndarray | None,
    candidate_keypoints: np.ndarray | None,
    trajectory_history: list[np.ndarray],
) -> None:
    rr.set_time_sequence("frame", frame_id)

    # camera-world for rerurn
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    # log camera entity
    rr.log("world/camera", rr.Transform3D(translation=t_wc, mat3x3=R_wc))

    viz_depth = 5
    # pinhole logging
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            image_from_camera=K,
            width=image.shape[1],
            height=image.shape[0],
            image_plane_distance=viz_depth,
        ),
    )
    # log pixel data
    rr.log("world/camera/image", rr.Image(image))

    # project 3D map points back
    if landmarks_3d is not None and len(landmarks_3d) > 0:
        # T_cam = R * X + t
        T_cam = (R_cw @ landmarks_3d.T + t_cw.reshape(3, 1)).T

        # filter points behind camera
        z = T_cam[:, 2]
        valid_depth = z > 0.1
        T_cam = T_cam[valid_depth]
        z = z[valid_depth]

        # u = fx*x/z + cx
        u = (K[0, 0] * T_cam[:, 0] / z) + K[0, 2]
        v = (K[1, 1] * T_cam[:, 1] / z) + K[1, 2]
        reprojections = np.stack([u, v], axis=1)

        rr.log(
            "world/camera/image/reprojections",
            rr.Points2D(reprojections, colors=[0, 255, 0], radii=2),
        )

    # log candidate rays
    if candidate_keypoints is not None and len(candidate_keypoints) > 0:
        # visual dots on the 2D image
        rr.log(
            "world/camera/image/candidates",
            rr.Points2D(candidate_keypoints, colors=[255, 0, 0], radii=2),
        )

        # 3d rays emanating from camera center
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = candidate_keypoints[:, 0]
        v = candidate_keypoints[:, 1]

        # project back to metric coords at depth z_end
        x_end = (u - cx) * viz_depth / fx
        y_end = (v - cy) * viz_depth / fy

        zeros = np.zeros(len(u))
        origins = np.stack([zeros, zeros, zeros], axis=1)
        ends = np.stack([x_end, y_end, np.full_like(x_end, viz_depth)], axis=1)

        strips = np.stack([origins, ends], axis=1)

        # log as child of camera
        rr.log(
            "world/camera/candidates_rays",
            rr.LineStrips3D(strips, colors=[255, 0, 0], radii=0.01),
        )

    # global point map and trajectory
    if len(trajectory_history) > 1:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([trajectory_history], colors=[255, 255, 0], radii=0.02),
        )

    if landmarks_3d is not None and len(landmarks_3d) > 0:
        rr.log(
            "world/landmarks", rr.Points3D(landmarks_3d, colors=[0, 255, 0], radii=0.03)
        )


def process_frame(
    img: np.ndarray,
    img_prev: np.ndarray,
    vo_state: VOState,
    landmark_db: LandmarkDatabase,
    K: np.ndarray,
    timestamp: float,
) -> tuple[VOState, LandmarkDatabase, bool, dict]:
    """
    Process Frame with detailed Debug Stats.
    """
    debug_stats = {}

    # existing landmarks to track using KLT
    lm_3d = landmark_db.landmarks_3d
    if len(lm_3d) == 0:
        return vo_state, landmark_db, False, {}

    # project to prev frame
    p1_projected = project_points(lm_3d, vo_state.R, vo_state.t, K)

    # filter visible frames
    h, w = img.shape
    visible = (
        (p1_projected[:, 0] >= 0)
        & (p1_projected[:, 0] < w)
        & (p1_projected[:, 1] >= 0)
        & (p1_projected[:, 1] < h)
    )

    p1_input = p1_projected[visible]
    indices_input = np.where(visible)[0]

    # track from previous to current frame using KLT
    p2, p3d_tracked, ids_tracked_idx = extract_and_match_features(
        img,
        landmark_db,
        prev_img=img_prev,
        prev_keypoints=p1_input,
        prev_landmark_indices=indices_input,
    )

    debug_stats["tracked_landmarks"] = len(p2)

    if len(p2) < 5:
        return vo_state, landmark_db, False, debug_stats

    # estimate camera pose using pnp ransac
    R_new, t_new, inlier_mask, error = estimate_pose_pnp(
        p3d_tracked, p2, K, initial_R=vo_state.R, initial_t=vo_state.t
    )

    num_inliers = np.sum(inlier_mask)
    debug_stats["pnp_inliers"] = num_inliers

    if num_inliers < 10:
        return vo_state, landmark_db, False, debug_stats

    # distance travelled
    t_diff = t_new - vo_state.t
    step_size = np.linalg.norm(t_diff)
    debug_stats["step_size"] = step_size
    # udpate state
    new_state = VOState(
        R=R_new,
        t=t_new,
        frame_id=vo_state.frame_id + 1,
        timestamp=timestamp,
        candidate_keypoints=vo_state.candidate_keypoints,
        first_observation_keypoints=vo_state.first_observation_keypoints,
        first_observation_poses=vo_state.first_observation_poses,
        candidate_descriptors=vo_state.candidate_descriptors,
    )

    # candidate tracking
    if len(new_state.candidate_keypoints) > 0:
        lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        p1_cand = new_state.candidate_keypoints.astype(np.float32)
        # track candidates for future triangulation
        p2_cand, status, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img, p1_cand, None, **lk_params
        )

        good = status.ravel() == 1
        if np.any(good):
            p2_good = p2_cand[good]
            in_bounds = (
                (p2_good[:, 0] >= 0)
                & (p2_good[:, 0] < w)
                & (p2_good[:, 1] >= 0)
                & (p2_good[:, 1] < h)
            )

            final_mask = np.zeros_like(good)
            good_indices = np.where(good)[0]
            final_mask[good_indices[in_bounds]] = True
            good = final_mask

        new_state.candidate_keypoints = p2_cand[good]
        new_state.first_observation_keypoints = new_state.first_observation_keypoints[
            good
        ]
        new_state.first_observation_poses = new_state.first_observation_poses[good]
        new_state.candidate_descriptors = new_state.candidate_descriptors[good]

    debug_stats["active_candidates"] = len(new_state.candidate_keypoints)

    # map management
    prev_landmark_count = len(landmark_db.landmarks_3d)

    landmark_db, new_state = update_landmark_database(
        new_state,
        vo_state,
        landmark_db,
        inlier_mask,
        ids_tracked_idx,
        current_frame_keypoints=new_state.candidate_keypoints,
        current_frame_descriptors=new_state.candidate_descriptors,
        K=K,
    )

    new_triangulated = len(landmark_db.landmarks_3d) - prev_landmark_count
    debug_stats["new_triangulated"] = new_triangulated

    # add new candidates
    points_to_mask = p2
    if len(new_state.candidate_keypoints) > 0:
        points_to_mask = np.vstack((points_to_mask, new_state.candidate_keypoints))

    new_state = add_new_candidates(new_state, img, points_to_mask, K)

    # filter
    landmark_db = filter_landmarks(landmark_db, vo_state=new_state, K=K)

    return new_state, landmark_db, True, debug_stats


@dataclass
class Args:
    dataset: Literal["kitti", "malaga", "parking", "own"] = "kitti"
    path: Path = Path("data")
    sequence: str = "05"
    cv2_viz: bool = False
    headless: bool = False


def main(args: Args) -> None:
    # setup
    print(f"Initializing {args.dataset}...")
    loader: BaseDataset
    if args.dataset == "kitti":
        loader = KittiDataset(args.path, sequence=args.sequence)
    elif args.dataset == "malaga":
        loader = MalagaDataset(args.path)
    elif args.dataset == "parking":
        loader = ParkingDataset(args.path)
    elif args.dataset == "own":
        loader = OwnDataset(args.path)

    if not loader.image_files:
        print("Error: No images found.")
        return

    if not args.cv2_viz and not args.headless:
        init_rerun()

    # initialization
    print("Bootstrapping...")
    img0 = cv2.imread(str(loader.image_files[0]), cv2.IMREAD_GRAYSCALE)
    idx_init = 2 if len(loader.image_files) > 2 else 1
    img_init = cv2.imread(str(loader.image_files[idx_init]), cv2.IMREAD_GRAYSCALE)

    vo_state, landmark_db, success = initialize_vo(img0, img_init, loader.K)

    if not success:
        print("Bootstrapping failed.")
        return

    vo_state.frame_id = idx_init

    # camera center C = -R.T * t
    C_init = -vo_state.R.T @ vo_state.t
    trajectory_history = [C_init.flatten()]

    full_trajectory = [vo_state]

    print(f"Continuous VO (Starting Frame {idx_init + 1})...")
    prev_img = img_init

    for i in range(idx_init + 1, len(loader.image_files)):
        img = cv2.imread(str(loader.image_files[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        new_state, landmark_db, success, stats = process_frame(
            img, prev_img, vo_state, landmark_db, loader.K, float(i)
        )
        step_size = stats.get("step_size", 0.0)

        print(
            f"Frame {i:04d} | "
            f"LMs: {len(landmark_db.landmarks_3d):04d} | "
            f"Tracked: {stats.get('tracked_landmarks', 0):03d} | "
            f"Inliers: {stats.get('pnp_inliers', 0):03d} | "
            f"Distance delta: {step_size:.4f} | "
            f"Cands: {stats.get('active_candidates', 0):03d} | "
            f"NewTri: {stats.get('new_triangulated', 0):02d}"
        )

        if not success:
            print(f"!!! VO FAILED at Frame {i} !!!")
            break

        vo_state = new_state
        prev_img = img

        # calculate camera center
        C_curr = -vo_state.R.T @ vo_state.t
        trajectory_history.append(C_curr.flatten())
        full_trajectory.append(vo_state)

        # ciz
        if not args.headless and not args.cv2_viz:
            T_cw = np.eye(4)
            T_cw[:3, :3] = vo_state.R
            T_cw[:3, 3] = vo_state.t.flatten()

            log_frame_rerun(
                img,
                T_cw,
                loader.K,
                i,
                landmark_db.landmarks_3d,
                vo_state.candidate_keypoints,
                trajectory_history,
            )

    save_trajectory(full_trajectory, "trajectory.txt")
    print("Done.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
