from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import tyro

from modules.dataset_loader import (
    BaseDataset,
    KittiDataset,
    MalagaDataset,
    OwnDataset,
    ParkingDataset,
)
from modules.initialization import initialize_vo
from modules.utils import create_camera_matrix, create_empty_landmark_database
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def init_rerun() -> None:
    """Initialize Rerun logging."""
    rr.init("Visual Odometry", spawn=True)
    # log coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


def log_frame_rerun(
    image: np.ndarray,
    T_cw: np.ndarray,
    K: np.ndarray,
    frame_id: int,
    landmarks: np.ndarray | None = None,
) -> None:
    """
    Log data to Rerun.

    Args:
        image: Current image (H, W).
        T_cw: World-to-Camera pose (4x4).
        K: Intrinsic matrix (3x3).
        frame_id: Frame index.
        landmarks: Optional 3D points (N, 3).

    """
    rr.set_time_sequence("frame_idx", frame_id)

    # we need to convert T_cw (world -> camera) to T_wc (camera -> world).
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    # log camera in world frame
    rr.log("world/camera", rr.Transform3D(translation=t_wc, mat3x3=R_wc))

    # log pinhole model
    rr.log(
        "world/camera/image",
        rr.Pinhole(image_from_camera=K, width=image.shape[1], height=image.shape[0]),
    )

    # log image
    rr.log("world/camera/image", rr.Image(image))

    # log landmarks
    if landmarks is not None and len(landmarks) > 0:
        rr.log("world/landmarks", rr.Points3D(landmarks, colors=[255, 255, 255]))


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


@dataclass
class Args:
    """Command line arguments."""

    # Dataset selection (default is kitti)
    dataset: Literal["kitti", "malaga", "parking", "own"] = "kitti"

    # Path to data directory (defaults to ./data so you don't have to type it)
    path: Path = Path("data")

    # Sequence (only used for KITTI)
    sequence: str = "05"

    # Visualization mode
    use_rerun: bool = False


def main(args: Args) -> None:
    """Run main function."""
    # 1. Initialize Dataset
    print(f"Initializing {args.dataset} dataset from {args.path}...")

    loader: BaseDataset
    if args.dataset == "kitti":
        loader = KittiDataset(args.path, sequence=args.sequence)
    elif args.dataset == "malaga":
        loader = MalagaDataset(args.path)
    elif args.dataset == "parking":
        loader = ParkingDataset(args.path)
    elif args.dataset == "own":
        loader = OwnDataset(args.path)

    if not loader.image_files:
        print(f"Error: No images found in {args.path}")
        return

    print(f"Loaded {len(loader.image_files)} images.")
    print(f"Camera Matrix K:\n{loader.K}\n")

    # 2. Initialize Rerun if requested
    if args.use_rerun:
        init_rerun()

    # 3. Dummy Loop (Plays back images)
    print("Starting processing loop...")

    # Placeholder for the actual VO pipeline state
    # vo_state = VOState(...)

    for i, img_path in enumerate(loader.image_files):
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # --- TODO: INSERT VO PIPELINE HERE ---
        # T_cw = vo.process(img, i)

        # For now, we use a dummy identity pose just to make the visualizer work
        T_cw = np.eye(4)
        # -------------------------------------

        # Visualization
        if args.use_rerun:
            log_frame_rerun(img, T_cw, loader.K, i)
        else:
            # Standard OpenCV window
            cv2.imshow("VO Skeleton", img)
            cv2.setWindowTitle("VO Skeleton", f"Frame {i} - {args.dataset}")
            if cv2.waitKey(1) == 27:  # ESC to stop
                break

    cv2.destroyAllWindows()
    print("\nDone! Skeleton loop finished.")

    empty_db = create_empty_landmark_database()
    print(f"Empty database created with {len(empty_db.landmarks_3d)} landmarks\n")

    # what to implement next
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


if __name__ == "__main__":
    # use tyro for intuitive argument parsing and help
    args = tyro.cli(Args)
    main(args)
