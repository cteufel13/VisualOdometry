import cv2
import numpy as np

from modules.feature_matching import find_unmatched_keypoints
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def update_landmark_database(
    vo_state: VOState,
    vo_state_prev: VOState,
    landmark_db: LandmarkDatabase,
    inlier_mask: np.ndarray,
    matched_landmark_indices: np.ndarray,
    current_frame_keypoints: np.ndarray | None,
    current_frame_descriptors: np.ndarray | None,
    K: np.ndarray,
) -> tuple[LandmarkDatabase, VOState]:
    # update observation counts
    if inlier_mask is not None and matched_landmark_indices is not None:
        inliers = matched_landmark_indices[inlier_mask.ravel()]
        if len(inliers) > 0:
            landmark_db.num_observations[inliers] += 1

    # check candidates for triangulation
    if vo_state.candidate_keypoints is None or len(vo_state.candidate_keypoints) == 0:
        return landmark_db, vo_state

    # params
    MIN_PARALLAX_DEG = 1.0

    num_candidates = len(vo_state.candidate_keypoints)
    keep_mask = np.ones(num_candidates, dtype=bool)

    new_landmarks_3d = []
    new_descriptors = []

    # prep geometry
    R_curr = vo_state.R
    t_curr = vo_state.t
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # first observation
    kps1 = vo_state.first_observation_keypoints
    vec1 = np.stack(
        [(kps1[:, 0] - cx) / fx, (kps1[:, 1] - cy) / fy, np.ones(num_candidates)],
        axis=1,
    )
    # transform R1.T @ vec1
    R1_all = vo_state.first_observation_poses[:, :, :3]  # (N, 3, 3)
    t1_all = vo_state.first_observation_poses[:, :, 3:]  # (N, 3, 1)

    # batched transpose of R1
    R1_T = np.transpose(R1_all, (0, 2, 1))

    # R1.T * vec1
    bearings1 = (R1_T @ vec1[:, :, np.newaxis]).squeeze()  # (N, 3)
    norm1 = np.linalg.norm(bearings1, axis=1, keepdims=True)
    bearings1 /= norm1 + 1e-9

    # current observation
    kps2 = vo_state.candidate_keypoints
    vec2 = np.stack(
        [(kps2[:, 0] - cx) / fx, (kps2[:, 1] - cy) / fy, np.ones(num_candidates)],
        axis=1,
    )
    # current pose is constant for all
    bearings2 = (R_curr.T @ vec2.T).T
    norm2 = np.linalg.norm(bearings2, axis=1, keepdims=True)
    bearings2 /= norm2 + 1e-9

    # compute parallalx
    dot_products = np.sum(bearings1 * bearings2, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_products))

    for i in range(num_candidates):
        if angles[i] > MIN_PARALLAX_DEG:
            # triangulate
            P1 = K @ np.hstack((R1_all[i], t1_all[i]))
            P2 = K @ np.hstack((R_curr, t_curr))

            pt1 = kps1[i]
            pt2 = kps2[i]

            pts4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
            if pts4d[3] == 0:
                continue

            pt3d = (pts4d[:3] / pts4d[3]).flatten()

            # current cam depth needs to be positive
            pt_cam = R_curr @ pt3d + t_curr.flatten()
            if pt_cam[2] <= 0.1:
                continue  # Behind camera or too close

            # first cam depth also needs to be positive
            pt_cam1 = R1_all[i] @ pt3d + t1_all[i].flatten()
            if pt_cam1[2] <= 0.1:
                continue

            # avoid too far points
            if np.linalg.norm(pt3d) > 200.0:
                continue

            # if good
            new_landmarks_3d.append(pt3d)
            new_descriptors.append(vo_state.candidate_descriptors[i])
            keep_mask[i] = False  # remove from candidates

        else:
            pass

    # add to db
    if len(new_landmarks_3d) > 0:
        new_landmarks_3d = np.array(new_landmarks_3d)
        new_descriptors = np.array(new_descriptors)

        start_id = 0
        if len(landmark_db.track_ids) > 0:
            start_id = landmark_db.track_ids.max() + 1

        new_ids = np.arange(start_id, start_id + len(new_landmarks_3d))
        new_obs = np.full(len(new_landmarks_3d), 2, dtype=np.int32)

        landmark_db.landmarks_3d = np.vstack(
            (landmark_db.landmarks_3d, new_landmarks_3d)
        )
        landmark_db.descriptors = np.vstack((landmark_db.descriptors, new_descriptors))
        landmark_db.track_ids = np.concatenate((landmark_db.track_ids, new_ids))
        landmark_db.num_observations = np.concatenate(
            (landmark_db.num_observations, new_obs)
        )

    # update state candidates
    vo_state.candidate_keypoints = vo_state.candidate_keypoints[keep_mask]
    vo_state.first_observation_keypoints = vo_state.first_observation_keypoints[
        keep_mask
    ]
    vo_state.first_observation_poses = vo_state.first_observation_poses[keep_mask]
    vo_state.candidate_descriptors = vo_state.candidate_descriptors[keep_mask]

    return landmark_db, vo_state


def add_new_candidates(
    vo_state: VOState, img: np.ndarray, current_keypoints: np.ndarray, K: np.ndarray
) -> VOState:
    # grid-based detection to fill gaps
    new_kpts, new_des = find_unmatched_keypoints(
        None, None, None, img, existing_keypoints=current_keypoints
    )

    if len(new_kpts) > 0:
        # create pose matrix for new observations
        pose_mat = np.hstack((vo_state.R, vo_state.t))  # (3, 4)
        poses = np.tile(pose_mat[np.newaxis, :, :], (len(new_kpts), 1, 1))

        if len(vo_state.candidate_keypoints) == 0:
            vo_state.candidate_keypoints = new_kpts
            vo_state.first_observation_keypoints = new_kpts
            vo_state.first_observation_poses = poses
            vo_state.candidate_descriptors = new_des
        else:
            vo_state.candidate_keypoints = np.vstack(
                (vo_state.candidate_keypoints, new_kpts)
            )
            vo_state.first_observation_keypoints = np.vstack(
                (vo_state.first_observation_keypoints, new_kpts)
            )
            vo_state.first_observation_poses = np.vstack(
                (vo_state.first_observation_poses, poses)
            )
            vo_state.candidate_descriptors = np.vstack(
                (vo_state.candidate_descriptors, new_des)
            )

    return vo_state


def filter_landmarks(
    landmark_db: LandmarkDatabase,
    min_observations: int = 2,
    vo_state: VOState | None = None,
    K: np.ndarray | None = None,
) -> LandmarkDatabase:
    # distance check
    valid = np.ones(len(landmark_db.landmarks_3d), dtype=bool)

    # remove crazy far points
    dists = np.linalg.norm(landmark_db.landmarks_3d, axis=1)
    valid &= dists < 300.0

    # remove behind camera
    if vo_state is not None:
        X_cam = (vo_state.R @ landmark_db.landmarks_3d.T + vo_state.t).T
        valid &= X_cam[:, 2] > 0.05

    if np.sum(valid) < len(landmark_db.landmarks_3d):
        landmark_db.landmarks_3d = landmark_db.landmarks_3d[valid]
        landmark_db.descriptors = landmark_db.descriptors[valid]
        landmark_db.track_ids = landmark_db.track_ids[valid]
        landmark_db.num_observations = landmark_db.num_observations[valid]

    return landmark_db


def compute_triangulation_quality(
    points_3d: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    Compute quality metric for triangulated points.

    Quality based on:
    - Baseline distance between camera positions
    - Angle between viewing rays (parallax angle)

    Args:
        points_3d: (N, 3) triangulated 3D points
        R1, t1: First camera pose
        R2, t2: Second camera pose

    Returns:
        quality_scores: (N,) quality metric for each point (higher is better)

    """
    pass


def update_observation_counts(
    landmark_db: LandmarkDatabase,
    matched_indices: np.ndarray,
    inlier_mask: np.ndarray,
) -> LandmarkDatabase:
    """
    Update observation counts for landmarks that were successfully matched.

    Only increment for inlier matches.

    Args:
        landmark_db: Current landmark database
        matched_indices: (M,) indices of landmarks that were matched
        inlier_mask: (M,) boolean mask indicating which matches are inliers

    Returns:
        updated_db: Database with updated observation counts

    """
    pass
