from pathlib import Path

import cv2


def video_to_frames(video_path, output_dir):
    # Create output directory if it doesnt exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    print(f"Processing video: {video_path.name}")

    vidcap = cv2.VideoCapture(str(video_path))

    if not vidcap.isOpened():
        print(f"Error: Could not open {video_path.name}")
        return

    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            save_path = output_dir / f"img_{count:05d}.png"

            cv2.imwrite(str(save_path), image)
            count += 1
        else:
            break

    cv2.destroyAllWindows()
    vidcap.release()
    print(f"Done! Extracted {count} frames.")


def run():
    script_path = Path(__file__).resolve()

    dataset_dir = script_path.parents[2] / "data" / "my_dataset"
    output_dir = dataset_dir / "images"

    print(f"Searching for videos in: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"Directory not found: {dataset_dir}")
        return

    # find video file with a valid extension
    valid_extensions = {".mov", ".mp4", ".avi", ".mkv"}

    video_file = next(
        (p for p in dataset_dir.iterdir() if p.suffix.lower() in valid_extensions), None
    )

    if video_file:
        video_to_frames(video_file, output_dir)
    else:
        print(f"No video found in {dataset_dir}")


if __name__ == "__main__":
    run()
