import numpy as np
from dataclasses import dataclass

@dataclass
class VOState:
    """Current state of the visual odometry system"""
    R: np.ndarray          # 3x3 rotation matrix (current camera orientation)
    t: np.ndarray          # 3x1 translation vector (current camera position)
    frame_id: int          # current frame number
    timestamp: float       # timestamp of current frame