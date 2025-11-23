"""Dataset loaders for standard VO benchmarks."""

import numpy as np
import io
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for a dataset loader."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize the dataset loader.

        Args:
            base_path: The root directory of the dataset.

        """
        self.base_path = base_path
        self.K: np.ndarray | None = None
        self.ground_truth: np.ndarray | None = None
        self.image_files: list[Path] = []

    @abstractmethod
    def load(self) -> None:
        """Load dataset-specific files (K, poses, image paths)."""
        pass


class KittiDataset(BaseDataset):
    """Loader for the KITTI dataset."""

    def __init__(self, base_path: Path, sequence: str = "05") -> None:
        """
        Initialize KITTI loader.

        Args:
            base_path: Root data directory.
            sequence: KITTI sequence number string (e.g. "05").

        """
        super().__init__(base_path)
        self.kitti_base = self.base_path / "kitti"
        self.sequence = sequence
        self.load()

    def load(self) -> None:
        """Load K, Ground Truth, and Image paths for KITTI."""
        # hardcoded K for KITTI
        self.K = np.array(
            [[7.18856e02, 0, 6.071928e02], [0, 7.18856e02, 1.852157e02], [0, 0, 1]]
        )

        # load ground truth poses
        pose_path = self.kitti_base / "poses" / f"{self.sequence}.txt"
        if pose_path.exists():
            poses = np.loadtxt(pose_path)
            self.ground_truth = poses[:, [3, 11]]

        # Get image file names
        img_dir = self.kitti_base / self.sequence / "image_0"
        self.image_files = sorted(img_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} image paths from KITTI {self.sequence}")


class MalagaDataset(BaseDataset):
    """Loader for the Malaga dataset."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize Malaga loader.

        Args:
            base_path: Root data directory.

        """
        super().__init__(base_path)
        self.load()

    def load(self) -> None:
        """Load K, Ground Truth, and Image paths for Malaga."""
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
    """Loader for the Parking dataset."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize Parking loader.

        Args:
            base_path: Root data directory.

        """
        super().__init__(base_path)
        self.parking_base = self.base_path / "parking"
        self.load()

    def load(self) -> None:
        """Load K, Ground Truth, and Image paths for Parking."""
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
        """
        Parse the K matrix from the specific text format of the Parking dataset.

        Args:
            k_path: Path to the K.txt file.

        Returns:
            3x3 Intrinsic Matrix as a numpy array.

        """
        with open(k_path, "r") as f:
            cleaned_text = f.read().replace(",", " ").strip()
        return np.loadtxt(io.StringIO(cleaned_text), dtype=np.float64)


class OwnDataset(BaseDataset):
    """Loader for custom user datasets."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize Custom loader.

        Args:
            base_path: Root data directory.

        """
        super().__init__(base_path)
        self.own_base = self.base_path / "my_dataset"
        self.load()

    def load(self) -> None:
        """Load K and Image paths for the custom dataset."""
        print("WARNING: OwnDataset K matrix is not set up.")
        self.K = np.eye(3)  # placeholder

        self.ground_truth = None

        img_dir = self.own_base / "images"
        self.image_files = sorted(img_dir.glob("*.png"))
        print(f"Loaded {len(self.image_files)} image paths from Own Dataset")
