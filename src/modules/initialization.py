import cv2
import numpy as np

from modules.utils import create_empty_landmark_database
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def initialize_vo(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    timestamp1: float = 0.0,
    timestamp2: float = 0.0,
) -> tuple[VOState, LandmarkDatabase, bool]:
    """
    Bootstrap VO from first two frames using 2D-to-2D motion estimation.

    Steps:
    1. Detect features in both images
    2. Match features between images
    3. Compute relative pose using Essential matrix (5-point algorithm + RANSAC)
    4. Triangulate initial 3D points
    5. Create initial VOState and LandmarkDatabase

    Args:
        img1: First image
        img2: Second image
        K: 3x3 camera calibration matrix
        timestamp1: Timestamp of first image
        timestamp2: Timestamp of second image

    Returns:
        vo_state: Initial VOState
        landmark_db: Initial LandmarkDatabase
        success: True if initialization successful

    """
    # detect features
    kpts1, desc1 = detect_keypoints_and_descriptors(img1)
    kpts2, desc2 = detect_keypoints_and_descriptors(img2)

    if len(kpts1) < 50 or len(kpts2) < 50:
        print(f"Initialization failed: Too few keypoints ({len(kpts1)}, {len(kpts2)})")
        return (
            VOState(np.eye(3), np.zeros((3, 1)), 0, 0.0),
            create_empty_landmark_database(),
            False,
        )

    # match features
    matches = match_descriptors(desc1, desc2)
    if len(matches) < 20:
        print(f"Initialization failed: Too few matches ({len(matches)})")
        return (
            VOState(np.eye(3), np.zeros((3, 1)), 0, 0.0),
            create_empty_landmark_database(),
            False,
        )

    # align points for 5-point algo
    pts1 = kpts1[matches[:, 0]]
    pts2 = kpts2[matches[:, 1]]

    # get essential amtrix
    R, t, inlier_mask = estimate_pose_2d_to_2d(pts1, pts2, K)

    num_inliers = np.sum(inlier_mask)
    if num_inliers < 15:
        print(f"Initialization failed: Too few inliers ({num_inliers})")
        return (
            VOState(np.eye(3), np.zeros((3, 1)), 0, 0.0),
            create_empty_landmark_database(),
            False,
        )

    print(f"Initialized with {num_inliers} / {len(matches)} inliers.")

    # filter outliers
    pts1_inliers = pts1[inlier_mask.ravel() == 1]
    pts2_inliers = pts2[inlier_mask.ravel() == 1]

    # take descriptors from newest image for db
    desc_inliers = desc2[matches[:, 1]][inlier_mask.ravel() == 1]

    # triangulate initial points
    R1, t1 = np.eye(3), np.zeros((3, 1))
    points_3d = triangulate_points(pts1_inliers, pts2_inliers, R1, t1, R, t, K)

    # create db object
    landmark_db = create_initial_landmark_database(points_3d, desc_inliers)

    # current state is newest state
    vo_state = VOState(
        R=R,
        t=t,
        frame_id=1,
        timestamp=timestamp2,
    )

    return vo_state, landmark_db, True


def detect_keypoints_and_descriptors(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect keypoints using Shi-Tomasi (GFTT) and compute ORB descriptors.
    We use GFTT because it provides better 'trackable' features for KLT later,
    but we compute descriptors here for the robust initialization matching.
    """
    # track using shi tomasi
    feature_params = {
        "maxCorners": 2000,
        "qualityLevel": 0.01,
        "minDistance": 7,
        "blockSize": 7,
    }
    pts = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)

    if pts is None:
        return np.empty((0, 2)), np.empty((0, 32))

    # convert to keypoint objects for descriptor computation
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]

    # compute orb descirptors
    orb = cv2.ORB_create()
    kps, descriptors = orb.compute(img, kps)

    if descriptors is None:
        return np.empty((0, 2)), np.empty((0, 32))

    # convert to numpy
    keypoints = np.array([kp.pt for kp in kps], dtype=np.float32)

    return keypoints, descriptors


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.8,
) -> np.ndarray:
    """
    Match descriptors between two sets using ratio test and brute force hamming distance.

    Args:
        desc1: (N1, D) descriptors from first image
        desc2: (N2, D) descriptors from second image
        ratio_threshold: Lowe's ratio test threshold (typically 0.8)

    Returns:
        matches: (M, 2) array where each row is [idx1, idx2]

    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), dtype=np.int32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append([m.queryIdx, m.trainIdx])

    return np.array(good_matches, dtype=np.int32)


def estimate_pose_2d_to_2d(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate relative pose from 2D-2D correspondences using Essential matrix.

    Uses 5-point algorithm with RANSAC for robustness.

    Args:
        kpts1: (N, 2) keypoints in first image
        kpts2: (N, 2) keypoints in second image
        K: 3x3 camera calibration matrix

    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector (unit scale)
        inlier_mask: (N,) boolean array indicating inliers

    """
    # principal point
    pp = (K[0, 2], K[1, 2])
    focal = K[0, 0]  # fx approx fy

    # find essential matrix
    E, mask = cv2.findEssentialMat(
        kpts1, kpts2, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    if E is None or E.shape != (3, 3):
        return np.eye(3), np.zeros((3, 1)), np.zeros(len(kpts1), dtype=bool)

    # recover pose from essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, kpts1, kpts2, focal=focal, pp=pp, mask=mask)

    return R, t, mask_pose.astype(bool)


def triangulate_points(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points from 2D correspondences and camera poses.

    Uses linear triangulation (DLT) or midpoint method.

    Args:
        kpts1: (N, 2) keypoints in first image
        kpts2: (N, 2) keypoints in second image
        R1, t1: First camera pose (typically identity)
        R2, t2: Second camera pose
        K: 3x3 camera calibration matrix

    Returns:
        points_3d: (N, 3) array of triangulated 3D points

    """
    # projection matrices P = K * [R|t]
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    # triangulate (4xN)
    pts4d = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T)

    # convert to 3D (N, 3)
    return (pts4d[:3] / pts4d[3]).T


def create_initial_landmark_database(
    landmarks_3d: np.ndarray,
    descriptors: np.ndarray,
) -> LandmarkDatabase:
    """
    Create initial landmark database from triangulated points.

    Args:
        landmarks_3d: (N, 3) array of 3D landmark positions
        descriptors: (N, D) array of descriptors

    Returns:
        landmark_db: Initial LandmarkDatabase

    """
    N = len(landmarks_3d)

    # create sequential IDs
    track_ids = np.arange(N, dtype=np.int32)

    # initialize observation count
    num_observations = np.full(N, 2, dtype=np.int32)

    return LandmarkDatabase(
        landmarks_3d=landmarks_3d,
        descriptors=descriptors,
        track_ids=track_ids,
        num_observations=num_observations,
    )
