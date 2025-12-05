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
from modules.feature_matching import extract_and_match_features
from modules.initialization import initialize_vo_from_two_frames
from modules.landmark_management import update_landmark_database
from modules.state_estimation import estimate_pose_pnp
from modules.utils import get_homogeneous_transform
from state.landmark_database import LandmarkDatabase
from state.vo_state import VOState


def init_rerun() -> None:
    """Initialize Rerun logging."""
    rr.init("Visual Odometry", spawn=True)
    # log coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


def log_frame_rerun(
    vo_state: VOState,
    K: np.ndarray,
    frame_id: int,
    lm_db: LandmarkDatabase,
) -> None:
    """
    Log data to Rerun.

    Args:
        vo_state: Current visual odometry state
        K: 3x3 camera intrinsic matrix
        frame_id: Frame index
        lm_db: Landmark database

    Raises:
        ValueError: If number of projected points doesn't match number of keypoints

    """
    image = vo_state.img
    T_cw = get_homogeneous_transform(vo_state)

    rr.set_time_sequence("frame_idx", frame_id)

    # Extract world -> camera transformation
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    # Convert to camera -> world for visualization
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    # Log camera pose in world frame
    rr.log("world/camera", rr.Transform3D(translation=t_wc, mat3x3=R_wc))

    # Log pinhole camera model
    rr.log(
        "world/camera/image",
        rr.Pinhole(image_from_camera=K, width=image.shape[1], height=image.shape[0]),
    )

    # Log the image
    rr.log("world/camera/image", rr.Image(image))

    # Separate matched and unmatched landmarks
    matched_track_ids = vo_state.matched_track_ids
    matched_mask = np.isin(lm_db.track_ids, matched_track_ids)

    # Log 3D landmarks in world frame
    if np.any(matched_mask):
        rr.log(
            "world/matched_landmarks",
            rr.Points3D(lm_db.landmarks_3d[matched_mask], colors=[0, 255, 0]),
        )
    if np.any(~matched_mask):
        rr.log(
            "world/unmatched_landmarks",
            rr.Points3D(lm_db.landmarks_3d[~matched_mask], colors=[255, 0, 0]),
        )

    # Project matched 3D landmarks to 2D image coordinates
    # cv2.projectPoints expects world->camera transformation (extrinsic parameters)
    if np.any(matched_mask):
        rvec_cw, _ = cv2.Rodrigues(R_cw)  # Convert rotation matrix to Rodrigues vector

        camera_projected_matched, _ = cv2.projectPoints(
            lm_db.landmarks_3d[matched_mask],  # 3D points in world coordinates
            rvec=rvec_cw,  # Rotation: world -> camera
            tvec=t_cw,  # Translation: world -> camera
            cameraMatrix=K,
            distCoeffs=np.zeros(4),
        )
        camera_projected_matched = camera_projected_matched.reshape(-1, 2)

        # Sanity check: number of projections should match number of matched keypoints
        if len(camera_projected_matched) != len(vo_state.matched_keypoints_2d):
            num_projected = len(camera_projected_matched)
            num_keypoints = len(vo_state.matched_keypoints_2d)
            error_msg = (
                f"Mismatch at frame {frame_id}: "
                f"{num_projected} projected points vs {num_keypoints} keypoints"
            )
            raise ValueError(error_msg)

        # Log reprojected 3D points (green) - where landmarks should appear
        rr.log(
            "world/camera/image/reprojected_matched_3d_points",
            rr.Points2D(
                camera_projected_matched,
                colors=[0, 255, 0],  # Green
                radii=3.0,
            ),
        )

        # Log actual detected 2D keypoints (blue) - where we actually detected them
        rr.log(
            "world/camera/image/matched_2d_keypoints",
            rr.Points2D(
                vo_state.matched_keypoints_2d,
                colors=[0, 0, 255],  # Blue
                radii=3.0,
            ),
        )

        # Calculate and log reprojection error statistics
        reprojection_errors = np.linalg.norm(
            camera_projected_matched - vo_state.matched_keypoints_2d, axis=1
        )
        mean_error = np.mean(reprojection_errors)
        max_error = np.max(reprojection_errors)

        rr.log("stats/mean_reprojection_error", rr.Scalars(mean_error))
        rr.log("stats/max_reprojection_error", rr.Scalars(max_error))
        rr.log("stats/num_matched_landmarks", rr.Scalars(len(matched_track_ids)))


def initialize_vo(
    images: list[np.ndarray],
    K: np.ndarray,
    *,
    cv2_viz: bool = False,
) -> tuple[list[np.ndarray], list[VOState], LandmarkDatabase, int]:
    """
    Bootstraps the VO pipeline from the first images, returning the remaining images,
    the initial two poses, the landmark database, and the next available track ID.

    Args:
        images: List of all images (will be modified by popping first frames)
        K: 3x3 camera calibration matrix
        cv2_viz: If True, use cv2 for visualization; otherwise use Rerun

    Returns:
        remaining_images: List of images not yet processed
        trajectory: List containing the first two VOState objects
        landmark_db: Initial landmark database
        next_track_id: Next available track ID for new landmarks

    """
    if len(images) < 2:
        error_msg = "Need at least 2 images for initialization"
        raise ValueError(error_msg)

    # Get first image
    first_image = images.pop(0)

    # Initialize VO with timeout
    max_initialization_attempts = min(10, len(images))
    initialization_successful = False

    second_vo_state = VOState(
        np.eye(3),
        np.zeros(3),
        0,
        img=first_image,
        matched_track_ids=[],
        matched_keypoints_2d=[],
    )

    landmark_db = None

    for _ in range(max_initialization_attempts):
        if len(images) == 0:
            error_msg = "Ran out of images before successful VO initialization"
            raise RuntimeError(error_msg)

        current_image = images.pop(0)
        second_vo_state, landmark_db, initialization_successful = (
            initialize_vo_from_two_frames(first_image, current_image, K)
        )

        if initialization_successful:
            break

    if not initialization_successful:
        error_msg = (
            f"Failed to initialize VO after {max_initialization_attempts} attempts"
        )
        raise RuntimeError(error_msg)

    # Create first VOState
    first_vo_state = VOState(
        np.eye(3),
        np.zeros(3),
        0,
        img=first_image,
        matched_track_ids=landmark_db.track_ids,
        matched_keypoints_2d=second_vo_state.matched_keypoints_2d,
    )

    trajectory = [first_vo_state, second_vo_state]

    # Log initial frames
    if cv2_viz:
        cv2.imshow("VO Skeleton", first_vo_state.img)
        cv2.waitKey(1)
        cv2.imshow("VO Skeleton", second_vo_state.img)
        cv2.waitKey(1)
    else:
        log_frame_rerun(first_vo_state, K, 0, landmark_db)
        log_frame_rerun(second_vo_state, K, 1, landmark_db)

    next_track_id = len(landmark_db.track_ids)

    return images, trajectory, landmark_db, next_track_id


def process_frame(
    img: np.ndarray,
    vo_state_prev: VOState,
    landmark_db: LandmarkDatabase,
    K: np.ndarray,
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
        vo_state_prev: Previous VO state (for triangulation)
        landmark_db: Current landmark database
        K: 3x3 camera calibration matrix
        next_track_id: Next available track ID for new landmarks

    Returns:
        updated_vo_state: Updated VOState
        updated_landmark_db: Updated LandmarkDatabase
        success: True if frame processed successfully
        next_track_id: Updated next track ID

    """
    points_2d, points_3d, landmark_indices = extract_and_match_features(
        img, landmark_db=landmark_db
    )
    new_R, new_t, inlier_mask, _ = estimate_pose_pnp(
        points_3d=points_3d, points_2d=points_2d, K=K
    )
    new_state = VOState(
        R=new_R,
        t=new_t,
        frame_id=vo_state_prev.frame_id + 1,
        img=img,
        matched_track_ids=landmark_db.track_ids[landmark_indices[inlier_mask]],
        matched_keypoints_2d=points_2d[inlier_mask],
    )

    # no new landmarks for first iteration
    # questions for next_track_id

    return new_state, landmark_db, True, 0


def run_vo_pipeline(
    images: list[np.ndarray],
    K: np.ndarray,
    *,
    cv2_viz: bool = False,
) -> tuple[list[VOState], LandmarkDatabase]:
    """
    Run complete VO pipeline on a sequence of images with incremental logging.

    Args:
        images: List of images
        K: 3x3 camera calibration matrix
        cv2_viz: If True, use cv2 for visualization; otherwise use Rerun

    Returns:
        trajectory: List of VOState objects (one per frame)
        final_landmark_db: Final landmark database

    """
    # Initialize VO from first frames
    remaining_images, trajectory, landmark_db, next_track_id = initialize_vo(
        images, K, cv2_viz=cv2_viz
    )

    current_vo_state = trajectory[-1]  # Get the last (second) state

    # Process remaining frames with incremental logging
    for i, image in enumerate(remaining_images):
        frame_idx = i + 2  # Account for the two initialization frames

        try:
            current_vo_state, landmark_db, success, next_track_id = process_frame(
                image,
                current_vo_state,
                landmark_db,
                K,
                next_track_id,
            )

            if not success:
                print(f"Frame {frame_idx} failed - stopping processing")
                break

            trajectory.append(current_vo_state)

            # Log immediately after successful processing
            if cv2_viz:
                cv2.imshow("VO Skeleton", current_vo_state.img)
                if cv2.waitKey(1) == 27:  # ESC to stop
                    print(f"Stopped by user at frame {frame_idx}")
                    break
            else:
                log_frame_rerun(current_vo_state, K, frame_idx, landmark_db)

            if frame_idx % 10 == 0:
                print(
                    f"Processed frame {frame_idx}/{len(remaining_images) + 2}, "
                    f"{len(landmark_db.landmarks_3d)} landmarks",
                )

        except (ValueError, cv2.error) as e:
            print(f"Processing error at frame {frame_idx}: {e}")
            print(
                f"Stopping pipeline. Successfully processed {len(trajectory)} frames."
            )
            break
        except KeyboardInterrupt:
            print(f"\nInterrupted by user at frame {frame_idx}")
            print(
                f"Stopping pipeline. Successfully processed {len(trajectory)} frames."
            )
            break

    return trajectory, landmark_db


@dataclass
class Args:
    """Command line arguments."""

    # Dataset selection (default is kitti)
    dataset: Literal["kitti", "malaga", "parking", "own"] = "kitti"

    # Path to data directory (defaults to ./data so you don't have to type it)
    path: Path = Path("data")

    # Sequence (only used for KITTI)
    sequence: str = "05"

    # Use cv2 for visualization instead of rerun
    cv2_viz: bool = False


def main(args: Args) -> None:
    """Run main function."""
    # Initialize Dataset
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

    # Initialize Rerun if requested
    if not args.cv2_viz:
        init_rerun()

    images = [
        cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        for img_path in loader.image_files
    ]
    K = loader.K

    # Run pipeline with incremental logging
    trajectory, lm_db = run_vo_pipeline(images, K, cv2_viz=args.cv2_viz)

    if args.cv2_viz:
        cv2.destroyAllWindows()

    print(f"\nDone! Processed {len(trajectory)} frames successfully.")
    print(f"Final landmark database contains {len(lm_db.landmarks_3d)} landmarks.")


if __name__ == "__main__":
    # use tyro for intuitive argument parsing and help
    args = tyro.cli(Args)
    main(args)
