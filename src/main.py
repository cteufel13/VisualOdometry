import numpy as np

from src.modules.initialization import initialize_vo
from src.modules.utils import create_camera_matrix, create_empty_landmark_database
from src.state.landmark_database import LandmarkDatabase
from src.state.vo_state import VOState


def process_first_two_frames(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    timestamp1: float = 0.0,
    timestamp2: float = 0.0,
) -> tuple[VOState, LandmarkDatabase, bool]:
    """
    Initialize the VO pipeline from first two frames.

    This is a convenience wrapper around initialize_vo.

    Args:
        img1: First image
        img2: Second image
        K: 3x3 camera calibration matrix
        timestamp1: Timestamp of first image
        timestamp2: Timestamp of second image

    Returns:
        vo_state: Initial VOState
        landmark_db: Initial LandmarkDatabase
        success: True if successful

    """
    return initialize_vo(img1, img2, K, timestamp1, timestamp2)


def process_frame(
    img: np.ndarray,
    vo_state: VOState,
    vo_state_prev: VOState,
    landmark_db: LandmarkDatabase,
    K: np.ndarray,
    timestamp: float,
    next_track_id: int,
) -> tuple[VOState, LandmarkDatabase, bool, int]:
    """
    Process a new frame in the VO pipeline.

    Main iteration:
    1. Extract features and match against landmark database
    2. Estimate camera pose using PnP-RANSAC
    3. Update VO state
    4. Update landmark database (triangulate new, filter old)

    Args:
        img: New input image
        vo_state: Current VO state
        vo_state_prev: Previous VO state (for triangulation)
        landmark_db: Current landmark database
        K: 3x3 camera calibration matrix
        timestamp: Timestamp of current frame
        next_track_id: Next available track ID for new landmarks

    Returns:
        updated_vo_state: Updated VOState
        updated_landmark_db: Updated LandmarkDatabase
        success: True if frame processed successfully
        next_track_id: Updated next track ID

    """
    pass


def run_vo_pipeline(
    images: list[np.ndarray],
    timestamps: list[float],
    K: np.ndarray,
) -> tuple[list[VOState], LandmarkDatabase]:
    """
    Run complete VO pipeline on a sequence of images.

    Args:
        images: List of images
        timestamps: List of timestamps
        K: 3x3 camera calibration matrix

    Returns:
        trajectory: List of VOState objects (one per frame)
        final_landmark_db: Final landmark database

    """
    if len(images) < 2:
        print("Error: Need at least 2 images")
        return [], None

    # Initialize from first two frames
    vo_state, landmark_db, success = process_first_two_frames(
        images[0],
        images[1],
        K,
        timestamps[0],
        timestamps[1],
    )

    if not success:
        print("Initialization failed")
        return [], None

    trajectory = [vo_state]
    next_track_id = len(landmark_db.track_ids)

    # Process remaining frames
    for i in range(2, len(images)):
        vo_state_prev = trajectory[-1]

        vo_state, landmark_db, success, next_track_id = process_frame(
            images[i],
            vo_state,
            vo_state_prev,
            landmark_db,
            K,
            timestamps[i],
            next_track_id,
        )

        if not success:
            print(f"Frame {i} failed")
            break

        trajectory.append(vo_state)

        if i % 10 == 0:
            print(
                f"Processed frame {i}/{len(images)}, "
                f"{len(landmark_db.landmarks_3d)} landmarks",
            )

    return trajectory, landmark_db


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create camera matrix
    K = create_camera_matrix(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157)
    print(f"Camera matrix K:\n{K}\n")

    # Example: Create empty database
    empty_db = create_empty_landmark_database()
    print(f"Empty database created with {len(empty_db.landmarks_3d)} landmarks\n")

    # TODO: Load images and run pipeline
    # images = [load_image(f"frame_{i:06d}.png") for i in range(100)]
    # timestamps = [i * 0.033 for i in range(100)]
    # trajectory, final_db = run_vo_pipeline(images, timestamps, K)

    print("Monocular VO functional skeleton ready!")
    print("\nMain functions to implement:")
    print("  - detect_keypoints_and_descriptors()")
    print("  - match_descriptors()")
    print("  - estimate_pose_2d_to_2d()")
    print("  - triangulate_points()")
    print("  - estimate_pose_pnp()")
    print("  - pnp_ransac()")
    print("  - p3p()")
    print("  - update_landmark_database()")
