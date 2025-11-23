"""Visual Odometry Orchestrator."""

import numpy as np
from datatypes import State, Frame


class VisualOdometry:
    """Main pipeline class that manages state and calls functional modules."""

    def __init__(self, K: np.ndarray) -> None:
        """
        Initialize the VO pipeline.

        Args:
            K: Camera intrinsic matrix.

        """
        self.K = K
        self.state = State()
        self.initialized = False

    def process(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Process a single image frame.

        Args:
            image: Grayscale image.
            frame_id: Integer frame index.

        Returns:
            4x4 Pose Matrix (World -> Camera).

        """
        # 1. Create Frame
        self.state.curr_frame = Frame(frame_id, image, self.K)

        # 2. Bootstrap (if needed)
        if not self.initialized:
            return self._bootstrap()

        # 3. Track Features (Frontend)
        # TODO: Call tracking.track_klt(last_img, curr_img, last_px)
        # TODO: Update self.state.curr_frame.px and point_ids

        # 4. Estimate Pose (Geometry)
        # TODO: Call geometry.estimate_pose_pnp() using valid landmarks
        # TODO: Update self.state.curr_frame.T_cw

        # 5. Map Maintenance (Backend)
        # TODO: Call backend.create_new_map_points()
        # TODO: Call tracking.detect_features_grid() if features dropped below threshold

        # 6. Update State
        self.state.last_frame = self.state.curr_frame
        return self.state.curr_frame.T_cw

    def _bootstrap(self) -> np.ndarray:
        """Handle the initialization of the first two frames."""
        # TODO: Implement 5-point algorithm bootstrap logic
        return np.eye(4)
