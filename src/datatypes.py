"""Passive data structures for the VO pipeline."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Frame:
    """
    A single timeframe containing image data and tracking info.

    Attributes:
        frame_id: Unique sequential identifier.
        image: Grayscale image data.
        K: Intrinsic camera matrix (3x3).
        T_cw: World-to-Camera pose matrix (4x4).
        px: Array of 2D feature coordinates (N, 2).
        point_ids: Array of associated MapPoint IDs (N,). -1 indicates no association.

    """

    frame_id: int
    image: np.ndarray
    K: np.ndarray
    T_cw: np.ndarray = field(default_factory=lambda: np.eye(4))
    px: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    point_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))


@dataclass
class MapPoint:
    """
    A 3D point in the world.

    Attributes:
        point_id: Unique identifier.
        position: 3D world coordinates [x, y, z].
        count: Number of times this point has been observed (track length).

    """

    point_id: int
    position: np.ndarray
    count: int = 0


@dataclass
class State:
    """
    Holds the global persistent state of the VO pipeline.

    Attributes:
        points: Dictionary mapping point_id to MapPoint objects.
        next_pt_id: Counter for assigning unique IDs to new MapPoints.
        last_frame: The previous processed frame (Reference).
        curr_frame: The current frame being processed.

    """

    points: dict[int, MapPoint] = field(default_factory=dict)
    next_pt_id: int = 0
    last_frame: Frame | None = None
    curr_frame: Frame | None = None
