import cv2
import numpy as np

from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def extract_and_match_features(
    img: np.ndarray,
    landmark_db: LandmarkDatabase,
    prev_img: np.ndarray | None = None,
    prev_keypoints: np.ndarray | None = None,
    prev_landmark_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track existing landmarks into the new image using KLT.

    Args:
        img: current frame
        landmark_db: database
        prev_img: Previous frame image
        prev_keypoints: (N, 2) keypoints in previous frame
        prev_landmark_indices: (N,) corresponding landmark IDs

    Returns:
        points_2d: (M, 2) tracked keypoints in current image
        points_3d: (M, 3) corresponding 3D landmarks
        landmark_indices: (M,) indices into landmark database

    """
    # return if no history
    if prev_img is None or prev_keypoints is None or len(prev_keypoints) == 0:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty(0, dtype=int)

    p1 = prev_keypoints.astype(np.float32)

    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    # lukas kanade optical flow
    p2, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, p1, None, **lk_params)

    h, w = img.shape

    # status == 1 means flow was found
    good_matches_mask = status.ravel() == 1

    # check if points are out of bounds
    if np.any(good_matches_mask):
        p2_good = p2[good_matches_mask]
        in_bounds = (
            (p2_good[:, 0] >= 0)
            & (p2_good[:, 0] < w)
            & (p2_good[:, 1] >= 0)
            & (p2_good[:, 1] < h)
        )

        final_mask = np.zeros_like(good_matches_mask)
        # get indices of the good matches
        good_indices = np.where(good_matches_mask)[0]
        # only keep in-bound points
        final_mask[good_indices[in_bounds]] = True
        good_matches_mask = final_mask

    # get 2d points in current frame
    points_2d = p2[good_matches_mask]

    # get indices of succesful tracked landmarks
    tracked_indices = prev_landmark_indices[good_matches_mask]

    # get 3d positions of landmarks
    points_3d = landmark_db.landmarks_3d[tracked_indices]

    return points_2d, points_3d, tracked_indices


def find_unmatched_keypoints(
    all_keypoints: np.ndarray,
    all_descriptors: np.ndarray,
    matched_indices: np.ndarray,
    img: np.ndarray,
    existing_keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Detect new keypoints in current frame to serve as candidates.

    Args:
        img: Current image
        existing_keypoints: (N, 2) Keypoints currently being tracked/matched

    Returns:
        final_kpts: (M, 2) array of new keypoints
        new_des: (M, 32) array of descriptors for the new keypoints

    """
    # mask out region where we already have features
    mask = np.full(img.shape, 255, dtype=np.uint8)

    # draw circles around existing keypoints in the mask so no duplicates are detected
    if existing_keypoints is not None and len(existing_keypoints) > 0:
        for kp in existing_keypoints:
            cv2.circle(mask, (int(kp[0]), int(kp[1])), 10, 0, -1)

    # detect new features using Shi-Tomasi
    feature_params = {
        "maxCorners": 2000,
        "qualityLevel": 0.01,
        "minDistance": 10,
        "blockSize": 7,
    }

    p_new = cv2.goodFeaturesToTrack(img, mask=mask, **feature_params)

    if p_new is None:
        return np.empty((0, 2)), np.empty((0, 32))

    new_kpts = p_new.reshape(-1, 2)

    # orb descriptors for new points
    orb = cv2.ORB_create()

    # convert to keypoint objects
    kps_obj = [cv2.KeyPoint(x=pt[0], y=pt[1], size=20) for pt in new_kpts]
    kps_obj, new_des = orb.compute(img, kps_obj)

    if new_des is None:
        return np.empty((0, 2)), np.empty((0, 32))

    final_kpts = np.array([kp.pt for kp in kps_obj], dtype=np.float32)

    return final_kpts, new_des


def match_against_database(
    query_descriptors: np.ndarray,
    db_descriptors: np.ndarray,
    ratio_threshold: float = 0.8,
) -> list[tuple[int, int]]:
    if len(query_descriptors) == 0 or len(db_descriptors) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(query_descriptors, db_descriptors, k=2)

    matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            matches.append((m.queryIdx, m.trainIdx))

    return matches
