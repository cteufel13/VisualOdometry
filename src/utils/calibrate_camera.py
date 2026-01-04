import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro


def calibrate_camera_charuco(
    image_folder: str,
    squares_x: int = 5,
    squares_y: int = 7,
    square_length: float = 0.03,
    marker_length: float = 0.015,
    aruco_dict_name: str = "DICT_5X5_100",
    output_file: str = "camera_calibration.npz",
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Calibrate camera using ChArUco board images

    Args:
        image_folder: folder containing calibration images
        squares_x: number of squares in X direction (must match printed board)
        squares_y: number of squares in Y direction (must match printed board)
        square_length: length of square side in meters (must match printed board)
        marker_length: length of ArUco marker side in meters (must match printed board)
        aruco_dict_name: ArUco dictionary name (must match printed board)
        output_file: path to save calibration results

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: distortion coefficients
        rvecs: rotation vectors for each image
        tvecs: translation vectors for each image

    """
    # Define the ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, aruco_dict
    )

    # Create CharucoDetector (OpenCV 4.7+)
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    # Lists to store detected corners and IDs from all images
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    # Get all image files
    image_files = sorted(glob.glob(f"{image_folder}/*.jpg")) + sorted(
        glob.glob(f"{image_folder}/*.png")
    )

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")

    # Calculate expected marker IDs for this board
    expected_ids = list(range(board.getIds().flatten().shape[0]))
    print(
        f"Board configuration: {squares_x}x{squares_y} squares, dict={aruco_dict_name}"
    )
    print(f"Expected marker IDs on board: {expected_ids}")
    print(f"Found {len(image_files)} images")
    print("Processing images...")

    for idx, image_file in enumerate(image_files):
        print(f"  [{idx + 1}/{len(image_files)}] {Path(image_file).name}", end="")

        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        # Detect ChArUco corners directly (OpenCV 4.7+)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(gray)
        )

        # If enough corners are found, add to calibration data
        if charuco_corners is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f" ✓ ({len(charuco_corners)} corners)")
        elif marker_corners is not None and len(marker_corners) > 0:
            print(
                f" ✗ (markers found but no corners - IDs: {marker_ids.flatten().tolist()})"
            )
        else:
            print(" ✗ (no markers detected)")

        # Debug visualization
        if debug:
            debug_img = img.copy()
            # Draw detected ArUco markers
            if marker_corners is not None and len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            # Draw detected ChArUco corners
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    debug_img, charuco_corners, charuco_ids, (0, 255, 0)
                )
            # Add status text
            n_markers = len(marker_corners) if marker_corners is not None else 0
            n_corners = len(charuco_corners) if charuco_corners is not None else 0
            cv2.putText(
                debug_img,
                f"Markers: {n_markers}, Corners: {n_corners}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # Resize for display if image is large
            h, w = debug_img.shape[:2]
            max_dim = 1200
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                debug_img = cv2.resize(debug_img, (int(w * scale), int(h * scale)))
            cv2.imshow("ChArUco Detection (Press any key)", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if len(all_charuco_corners) < 3:
        raise ValueError(
            f"Not enough valid images for calibration. Got {len(all_charuco_corners)}, need at least 3"
        )

    print(f"\nCalibrating with {len(all_charuco_corners)} images...")

    # Get 3D object points for each detected corner
    # Filter images with too few points (need at least 6 for calibration)
    obj_points = []
    img_points = []
    board_corners = board.getChessboardCorners()
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        if len(ids) < 6:
            continue
        obj_pts = board_corners[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 2).astype(np.float32)
        obj_points.append(obj_pts)
        img_points.append(img_pts)

    if len(obj_points) < 3:
        raise ValueError(
            f"Not enough valid images with 6+ corners. Got {len(obj_points)}"
        )

    # Calibrate camera using standard OpenCV function
    # Fix distortion coefficients to zero
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        dist_coeffs,
        flags=(
            cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_TANGENT_DIST
        ),
    )

    if not ret:
        raise ValueError("Calibration failed")

    # Calculate reprojection error
    total_error = 0
    total_points = 0

    for i in range(len(obj_points)):
        # Project 3D points to image plane
        imgpoints2, _ = cv2.projectPoints(
            obj_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(img_points[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(
            imgpoints2
        )
        total_error += error * len(img_points[i])
        total_points += len(img_points[i])

    mean_error = total_error / total_points

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"\nReprojection error: {mean_error:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())

    # Extract focal lengths and principal point
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print("\nIntrinsic Parameters:")
    print(f"  fx = {fx:.2f} pixels")
    print(f"  fy = {fy:.2f} pixels")
    print(f"  cx = {cx:.2f} pixels")
    print(f"  cy = {cy:.2f} pixels")
    print(f"  Image size: {image_size[0]} x {image_size[1]}")

    # Save results
    np.savetxt(output_file, camera_matrix)
    print(f"\nCalibration data saved to '{output_file}'")

    return camera_matrix, dist_coeffs, rvecs, tvecs


@dataclass
class Args:
    """Camera calibration using ChArUco board (Default: 7x5, 30mm squares, 15mm markers, DICT_5X5_100, A4 page)."""

    # Path to folder containing calibration images
    image_folder: Path

    # Number of squares in X direction (must match printed board)
    squares_x: int = 5

    # Number of squares in Y direction (must match printed board)
    squares_y: int = 7

    # Length of square side in meters (must match printed board)
    square_length: float = 0.03

    # Length of ArUco marker side in meters (must match printed board)
    marker_length: float = 0.015

    # ArUco dictionary name (must match printed board)
    aruco_dict: str = "DICT_5X5_100"

    # Output file path for calibration results
    output: Path = Path("./K.txt")

    # Show debug visualization of detected markers/corners
    debug: bool = False


def main(args: Args) -> None:
    """Run camera calibration."""
    if not args.image_folder.exists():
        print(f"Error: Image folder '{args.image_folder}' does not exist")
        return

    try:
        calibrate_camera_charuco(
            image_folder=str(args.image_folder),
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            aruco_dict_name=args.aruco_dict,
            output_file=str(args.output),
            debug=args.debug,
        )
    except Exception as e:
        print(f"\nError during calibration: {e}")
        return


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
