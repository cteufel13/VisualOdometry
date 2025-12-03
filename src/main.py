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
from modules.initialization import initialize_vo
from modules.landmark_management import update_landmark_database
from modules.state_estimation import estimate_pose_pnp
from modules.utils import create_empty_landmark_database
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
    # matched_track_ids: np.ndarray,
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

    image = vo_state.img
    T_cw = vo_state.get_homogenous_tf()

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

    matched_track_ids = vo_state.matched_track_ids
    # log landmarks
    matched_mask = np.isin(lm_db.track_ids, matched_track_ids)
    rr.log(
        "world/matched_landmarks",
        rr.Points3D(lm_db.landmarks_3d[matched_mask], colors=[0, 255, 0]),
    )
    rr.log(
        "world/unmatched_landmarks",
        rr.Points3D(lm_db.landmarks_3d[~matched_mask], colors=[255, 0, 0]),
    )

    camera_projected_matched, _ = cv2.projectPoints(
        lm_db.landmarks_3d[matched_mask],
        rvec=R_wc,
        tvec=t_wc,
        cameraMatrix=K,
        distCoeffs=np.zeros(4),
    )
    assert len(camera_projected_matched) == len(
        vo_state.matched_keypoints_2d
    ), f"image: {frame_id},len 1 = {len(camera_projected_matched)}, len 2 = {len(
        vo_state.matched_keypoints_2d
    )} "

    camera_projected_matched = camera_projected_matched.reshape(-1, 2)

    # camera_projected_unmatched = cv2.projectPoints(
    #     lm_db.landmarks_3d[~matched_mask],
    #     rvec=R_wc,
    #     tvec=t_wc,
    #     cameraMatrix=K,
    #     distCoeffs=np.zeros(4),
    # )

    assert len(camera_projected_matched) == len(vo_state.matched_keypoints_2d)
    # Log as 2D points overlaid on image
    rr.log(
        "world/camera/image/feature1",
        rr.Points2D(
            camera_projected_matched,
            colors=[[0, 255, 0] for _ in range(len(camera_projected_matched))],
            radii=3.0,
        ),
    )
    rr.log(
        "world/camera/image/feature2",
        rr.Points2D(
            vo_state.matched_keypoints_2d,
            colors=[[0, 0, 255] for _ in range(len(camera_projected_matched))],
            radii=3.0,
        ),
    )


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
) -> tuple[list[VOState], LandmarkDatabase]:
    """
    Run complete VO pipeline on a sequence of images.

    Args:
        images: List of images
        K: 3x3 camera calibration matrix

    Returns:
        trajectory: List of VOState objects (one per frame)
        final_landmark_db: Final landmark database

    """
    if len(images) < 2:
        print("Error: Need at least 2 images")
        return [], None

    # Initialize from first two frames
    first_image = images.pop(0)

    # Initialize VO with timeout
    max_initialization_attempts = min(10, len(images))  # Prevent infinite loop
    initialization_successful = False

    second_vo_state = VOState(
        np.eye(3),
        np.zeros(3),
        0,
        img=first_image,
        matched_track_ids=[],
        matched_keypoints_2d=[],
    )

    for _ in range(max_initialization_attempts):
        if len(images) == 0:
            raise RuntimeError("Ran out of images before successful VO initialization")

        current_image = images.pop(0)
        second_vo_state, landmark_db, initialization_successful = initialize_vo(
            first_image, current_image, K
        )

        if initialization_successful:
            break

    if not initialization_successful:
        raise RuntimeError(
            f"Failed to initialize VO after {max_initialization_attempts} attempts"
        )

    # Initialize trajectory
    trajectory = [
        VOState(
            np.eye(3),
            np.zeros(3),
            0,
            img=first_image,
            matched_track_ids=landmark_db.track_ids,
            matched_keypoints_2d=second_vo_state.matched_keypoints_2d,
        ),
        second_vo_state,
    ]

    next_track_id = len(landmark_db.track_ids)

    current_vo_state = second_vo_state
    # Process remaining frames
    for i in range(2, len(images)):
        current_vo_state, landmark_db, success, next_track_id = process_frame(
            images[i],
            current_vo_state,
            landmark_db,
            K,
            next_track_id,
        )

        if not success:
            print(f"Frame {i} failed")
            break

        trajectory.append(current_vo_state)

        if i % 10 == 0:
            print(
                f"Processed frame {i}/{len(images)}, "
                f"{len(landmark_db.landmarks_3d)} landmarks",
            )

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
    trajectory, lm_db = run_vo_pipeline(images, K)

    for i, (vo_state) in enumerate(trajectory):
        if args.cv2_viz:
            cv2.imshow("VO Skeleton", vo_state.img)
            if cv2.waitKey(1) == 27:  # ESC to stop
                break
        else:
            log_frame_rerun(
                vo_state,
                loader.K,
                i,
                lm_db,
            )

    cv2.destroyAllWindows()
    print("\nDone! Skeleton loop finished.")


if __name__ == "__main__":
    # use tyro for intuitive argument parsing and help
    args = tyro.cli(Args)
    main(args)
