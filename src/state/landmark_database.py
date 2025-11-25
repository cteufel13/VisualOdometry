import numpy as np
from dataclasses import dataclass

@dataclass
class LandmarkDatabase:
    """Database of 3D landmarks with their descriptors"""
    landmarks_3d: np.ndarray      # (N, 3) array of 3D positions [X, Y, Z]
    descriptors: np.ndarray       # (N, D) array of feature descriptors
    track_ids: np.ndarray         # (N,) array of unique landmark IDs
    num_observations: np.ndarray  # (N,) array tracking number of observations per landmark