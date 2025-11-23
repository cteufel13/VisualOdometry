"""Entry point with Rerun visualization."""

import cv2
import numpy as np
import rerun as rr
from pathlib import Path

# Internal imports
from dataset_loader import KittiDataset, MalagaDataset, ParkingDataset, OwnDataset
from vo import VisualOdometry


def init_rerun() -> None:
    """Initialize Rerun logging."""
    rr.init("Visual Odometry", spawn=True)


def log_frame(
    image: np.ndarray, T_cw: np.ndarray, K: np.ndarray, frame_id: int
) -> None:
    """
    Log data to Rerun.

    Args:
        image: Current image.
        T_cw: World-to-Camera pose.
        K: Intrinsic matrix.
        frame_id: Frame index.

    """
    rr.set_time_sequence("frame_idx", frame_id)

    # Log Image
    rr.log("camera/image", rr.Image(image))

    # Log Camera Pose
    # Rerun expects Camera -> World (T_wc)
    # T_cw = [R|t] -> T_wc = [R.T | -R.T * t]
    R_wc = T_cw[:3, :3].T
    t_wc = -R_wc @ T_cw[:3, 3]

    rr.log(
        "camera",
        rr.Transform3D(translation=t_wc, mat3x3=R_wc, from_parent=True),
    )

    # Log Pinhole model
    rr.log(
        "camera",
        rr.Pinhole(image_from_camera=K, width=image.shape[1], height=image.shape[0]),
    )


def main() -> None:
    """Run the VO pipeline."""
    # Configuration
    ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
    base_path = Path("data")

    # Setup Dataset
    dataset = None
    if ds == 0:
        dataset = KittiDataset(base_path)
    elif ds == 1:
        dataset = MalagaDataset(base_path)
    elif ds == 2:
        dataset = ParkingDataset(base_path)
    elif ds == 3:
        dataset = OwnDataset(base_path)

    if dataset is None or not dataset.image_files:
        print("Dataset not found.")
        return

    # Initialize Rerun & Pipeline
    init_rerun()
    vo = VisualOdometry(dataset.K)

    print(f"Processing {len(dataset.image_files)} frames...")

    for i, img_path in enumerate(dataset.image_files):
        # Read image
        # Note: cv2.imread does not accept Path objects directly in all versions
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Run Pipeline
        T_cw = vo.process(img, i)

        # Visualization
        log_frame(img, T_cw, dataset.K, i)

        # Optional: Local opencv window for debug
        cv2.imshow("VO", img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
