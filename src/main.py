from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import tyro

from config.config import get_config
from modules.dataset_loader import (
    BaseDataset,
    KittiDataset,
    MalagaDataset,
    OwnDataset,
    ParkingDataset,
)
from modules.vo import VisualOdometry


@dataclass
class Args:
    """Command line arguments."""

    # Dataset selection (default is kitti)
    dataset: Literal["kitti", "malaga", "parking", "own"] = "kitti"

    # Path to data directory (defaults to ./data so you don't have to type it)
    path: Path = Path("data")

    # Sequence (only used for KITTI)
    sequence: str = "05"


def main(args: Args) -> None:
    """Run main function."""
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

    config = get_config(args.dataset)
    vo = VisualOdometry(loader.K, config)
    print("Starting processing loop...")

    for i, img_path in enumerate(loader.image_files):
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        vo.process_frame(img)

    input("Press Enter to close the viewer and exit...")


if __name__ == "__main__":
    # use tyro for intuitive argument parsing and help
    args = tyro.cli(Args)
    main(args)
