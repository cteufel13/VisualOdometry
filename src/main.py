import cv2
import numpy as np
import io
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for a dataset loader."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.K = None
        self.ground_truth = None
        self.image_files = []

    @abstractmethod
    def load(self):
        """Loads dataset-specific files (K, poses, image paths)."""
        pass


class KittiDataset(BaseDataset):
    def __init__(self, base_path: Path, sequence: str = "05"):
        super().__init__(base_path)
        self.kitti_base = self.base_path / "kitti"
        self.sequence = sequence
        self.load()

    def load(self):
        # hardcoded K for KITTI
        self.K = np.array(
            [[7.18856e02, 0, 6.071928e02], [0, 7.18856e02, 1.852157e02], [0, 0, 1]]
        )

        # load ground truth poses
        pose_path = self.kitti_base / "poses" / f"{self.sequence}.txt"
        if pose_path.exists():
            poses = np.loadtxt(pose_path)
            self.ground_truth = poses[:, [3, 11]]

        # get image file names
        img_dir = self.kitti_base / self.sequence / "image_0"
        self.image_files = sorted(img_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} image paths from KITTI {self.sequence}")


class MalagaDataset(BaseDataset):
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.load()

    def load(self):
        # hardcoded K for Malaga
        self.K = np.array(
            [[621.18428, 0, 404.0076], [0, 621.18428, 309.05989], [0, 0, 1]]
        )

        # we dont have any ground truth
        self.ground_truth = None

        # get image file names
        img_dir = (
            self.base_path
            / "malaga"
            / "malaga-urban-dataset-extract-07_rectified_800x600_Images"
        )
        self.image_files = sorted(img_dir.glob("*.jpg"))
        print(f"Loaded {len(self.image_files)} image paths from Malaga")


class ParkingDataset(BaseDataset):
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.parking_base = self.base_path / "parking"
        self.load()

    def load(self):
        # load K from text file
        k_path = self.parking_base / "K.txt"
        self.K = self._load_parking_k(k_path)

        # load ground truth poses
        pose_path = self.parking_base / "poses.txt"
        if pose_path.exists():
            poses = np.loadtxt(pose_path)
            self.ground_truth = poses[:, [3, 11]]

        # Get image files
        img_dir = self.parking_base / "images"
        self.image_files = sorted(img_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} image paths from Parking")

    def _load_parking_k(self, k_path: Path) -> np.ndarray:
        """Handles the comma-and-space separated K file for Parking dataset."""
        with open(k_path, "r") as f:
            cleaned_text = f.read().replace(",", " ").strip()
        return np.loadtxt(io.StringIO(cleaned_text), dtype=np.float64)


class OwnDataset(BaseDataset):
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.own_base = self.base_path / "my_dataset"
        self.load()

    def load(self):
        # TODO: Load calibrated K matrix
        # k_path = self.own_base / 'K.txt'
        # self.K = np.loadtxt(k_path)
        print("WARNING: OwnDataset K matrix is not set up.")
        self.K = np.eye(3)  # placeholder

        self.ground_truth = None

        # TODO: glob pattern for custom images
        img_dir = self.own_base / "images"
        self.image_files = sorted(img_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} image paths from Own Dataset")


def main():
    ds = 2  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset

    base_path = Path(__file__).parent.parent / "data"

    # --- Dataset Factory ---
    # This is all the dataset-specific logic we need in main()
    # It's clean, readable, and scalable.
    if ds == 0:
        dataset = KittiDataset(base_path, sequence="05")
    elif ds == 1:
        dataset = MalagaDataset(base_path)
    elif ds == 2:
        dataset = ParkingDataset(base_path)
    elif ds == 3:
        dataset = OwnDataset(base_path)
    else:
        raise ValueError("Invalid dataset index")

    if not dataset.image_files:
        print(f"No image files found for dataset {ds}. Check paths.")
        return

    # frames 1 and 3 for KITTI (indices 0 and 2)
    bootstrap_frames = [0, 2]

    img0_path = dataset.image_files[bootstrap_frames[0]]
    img1_path = dataset.image_files[bootstrap_frames[1]]

    # pass the path string to cv2
    img0 = cv2.imread(str(img0_path), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)

    if img0 is None or img1 is None:
        print(f"Error: Could not read bootstrap images: {img0_path} or {img1_path}")
        return

    # TODO: Call Initialization function
    # [S_0, T_0_WC] = initialize(img0, img1, dataset.K)
    # trajectory = [T_0_WC]
    # S_prev = S_0
    # I_prev = img1

    print(f"Starting continuous operation from frame {bootstrap_frames[1] + 1}...")

    for i in range(bootstrap_frames[1] + 1, len(dataset.image_files)):
        image_path = dataset.image_files[i]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        # --- TODO: Call ProcessFrame ---
        # [S_curr, T_curr_WC] = processFrame(image, I_prev, S_prev, dataset.K)

        # --- TODO: Update state and visualize ---
        # trajectory.append(T_curr_WC)
        # S_prev = S_curr
        # I_prev = image

        cv2.imshow("Current Frame", image)
        if cv2.waitKey(10) == 27:  # Exit on 'ESC'
            print("User pressed ESC, exiting.")
            break

    cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    main()
