import cv2
import numpy as np

from config import config
from state.landmark_database import LandmarkDatabase
from utils.enums import create_descriptor, create_detector, DescriptorType


def detect_keypoints_and_descriptors(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect keypoints and compute descriptors in image.

    Args:
        img: Input grayscale image

    Returns:
        keypoints: (N, 2) array of [u, v] coordinates
        descriptors: (N, D) array of feature descriptors

    """
    # Feature Detector (Configurable)
    detector = create_detector(config.DETECT_TYPE, threshold=config.FAST_THRESH)

    # Feature Descriptor (Configurable)
    descriptor = create_descriptor(config.DESCRIPT_TYPE)

    # Calculate keypoints and convert to np.ndarray when needed
    keypoints = detector.detect(img)
    _, descriptors = descriptor.compute(img, keypoints)

    keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return keypoints_np, descriptors


def extract_and_match_features(
    img: np.ndarray,
    landmark_db: LandmarkDatabase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect features in new image and match against landmark database.

    Steps:
    1. Detect keypoints and compute descriptors in new image
    2. Match descriptors against landmark database descriptors
    3. Build 2D-3D correspondences

    Args:
        img: New input image
        landmark_db: Database of 3D landmarks with descriptors

    Returns:
        points_2d: (M, 2) array of 2D keypoint locations in new image
        points_3d: (M, 3) array of corresponding 3D landmark positions
        landmark_indices: (M,) array of indices into landmark database

    """
    # Find new keypoints in image
    keypoints, descriptors = detect_keypoints_and_descriptors(img)

    # match those new keypoints to old ones
    matches = match_descriptors(
        descriptors, landmark_db.descriptors, config.LOWES_RATIO
    )

    # if not matches just return zfor all
    if len(matches) == 0:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty(0, dtype=np.int32)

    # Indices of the matches found
    query_indices = matches[:, 0]
    landmark_indices = matches[:, 1]

    # finds the corresponding 2d/3d points
    points_2d = keypoints[query_indices]
    points_3d = landmark_db.landmarks_3d[landmark_indices]
    return points_2d, points_3d, landmark_indices


def match_descriptors(
    query_descriptors: np.ndarray,
    db_descriptors: np.ndarray,
    ratio_threshold: float = 0.8,
) -> np.ndarray:
    """
    Match query descriptors against database descriptors.

    Uses distance ratio test to reject ambiguous matches and enforces
    one-to-one correspondence (each descriptor matched at most once).

    Args:
        query_descriptors: (N, D) descriptors from new image
        db_descriptors: (M, D) descriptors from landmark database
        ratio_threshold: Lowe's ratio test threshold
        descriptor_type: Type of descriptor ("ORB", "SIFT", "SURF", etc.)

    Returns:
        matches: (K, 2) array of (query_idx, db_idx) pairs, where K <= min(N, M)

    """
    # Select appropriate norm based on descriptor type
    if config.DESCRIPT_TYPE in [DescriptorType.ORB]:
        norm_type = cv2.NORM_HAMMING
    elif config.DESCRIPT_TYPE in [DescriptorType.SIFT]:
        norm_type = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown descriptor type: {config.DESCRIPT_TYPE}")

    bf = cv2.BFMatcher(norm_type, crossCheck=False)

    # Find 2 best matches for each descriptor (for ratio test)
    matches = bf.knnMatch(query_descriptors, db_descriptors, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        # Need at least 2 matches to apply ratio test
        if len(match_pair) == 2:
            m, n = match_pair  # m is closest, n is second closest
            if m.distance < ratio_threshold * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx, m.distance))

    # Enforce one-to-one correspondence: each query and db descriptor matched at most once
    # Sort by distance (best matches first) and greedily select unique associations
    good_matches.sort(key=lambda x: x[2])  # Sort by distance

    query_matched = set()
    db_matched = set()
    unique_matches = []

    for query_idx, db_idx, distance in good_matches:
        if query_idx not in query_matched and db_idx not in db_matched:
            unique_matches.append([query_idx, db_idx])
            query_matched.add(query_idx)
            db_matched.add(db_idx)

    return (
        np.array(unique_matches, dtype=np.int32)
        if unique_matches
        else np.empty((0, 2), dtype=np.int32)
    )


def find_unmatched_keypoints(
    all_keypoints: np.ndarray,
    all_descriptors: np.ndarray,
    matched_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find keypoints in current frame that were not matched to database.

    These are candidates for creating new landmarks.

    Args:
        all_keypoints: (N, 2) all detected keypoints in current frame
        all_descriptors: (N, D) all descriptors in current frame
        matched_indices: (M,) indices of keypoints that were matched

    Returns:
        unmatched_keypoints: (K, 2) array of unmatched keypoints
        unmatched_descriptors: (K, D) array of unmatched descriptors

    """
    all_indices = np.arange(len(all_keypoints))

    unmatched_indices = all_indices[~np.isin(all_indices, matched_indices)]

    unmatched_keypoints = all_keypoints[unmatched_indices]
    unmatched_descriptors = all_descriptors[unmatched_indices]

    return unmatched_keypoints, unmatched_descriptors
