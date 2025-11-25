import numpy as np

from src.state.landmark_database import LandmarkDatabase
from src.state.vo_state import VOState

def extract_and_match_features(img: np.ndarray,
                               landmark_db: LandmarkDatabase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    pass


def match_against_database(query_descriptors: np.ndarray,
                          db_descriptors: np.ndarray,
                          ratio_threshold: float = 0.8) -> list[tuple[int, int]]:
    """
    Match query descriptors against database descriptors.
    
    Uses distance ratio test to reject ambiguous matches.
    
    Args:
        query_descriptors: (N, D) descriptors from new image
        db_descriptors: (M, D) descriptors from landmark database
        ratio_threshold: Lowe's ratio test threshold
        
    Returns:
        matches: List of (query_idx, db_idx) tuples
    """
    pass


def find_unmatched_keypoints(all_keypoints: np.ndarray,
                            all_descriptors: np.ndarray,
                            matched_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    pass