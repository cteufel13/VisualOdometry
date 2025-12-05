# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Monocular Visual Odometry (VO)** pipeline that estimates camera pose and 3D structure from a sequence of monocular images. The system uses feature-based SLAM techniques with PnP-RANSAC for pose estimation and maintains a landmark database for continuous tracking.

## Development Setup

This project uses **`uv`** for fast Python package management:

```bash
# Install dependencies and create virtual environment
uv sync

# Activate environment
source .venv/bin/activate

# Install pre-commit hooks (for contributors)
pre-commit install
```

## Running the Pipeline

The main entry point is `src/main.py`. Use `tyro` for argument parsing:

```bash
# View all available options
python src/main.py -h

# Run with default settings (KITTI sequence 05, Rerun visualization)
python src/main.py

# Run with cv2 visualization instead of Rerun
python src/main.py --cv2_viz

# Run on a different dataset
python src/main.py --dataset parking

# Run KITTI with a specific sequence
python src/main.py --dataset kitti --sequence 06
```

## Testing and Linting

```bash
# Run tests with coverage
pytest

# Run tests with specific markers
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Run only integration tests
pytest -m unit             # Run only unit tests

# Format code with ruff
ruff format

# Lint code with ruff (auto-fix where possible)
ruff check --fix

# Type check with mypy
mypy src
```

## Architecture Overview

### Core Pipeline Flow

The VO pipeline follows this structure:

1. **Initialization** (`modules/initialization.py`): Bootstrap from first two frames using 2D-to-2D motion estimation (Essential matrix + triangulation)
2. **Main Loop** (`main.py::process_frame`):
   - Extract and match features against landmark database
   - Estimate pose using PnP-RANSAC
   - Update landmark database (add new landmarks, filter old ones)
3. **Visualization**: Either Rerun (3D interactive) or cv2 (2D image display)

### Key Data Structures

**VOState** (`state/vo_state.py`): Represents camera state at a frame
- `R`: 3x3 rotation matrix (camera orientation)
- `t`: 3x1 translation vector (camera position)
- `img`: Current image
- `matched_track_ids`: IDs of landmarks matched in this frame
- `matched_keypoints_2d`: 2D keypoint locations in this frame

**LandmarkDatabase** (`state/landmark_database.py`): Persistent 3D map
- `landmarks_3d`: (N, 3) 3D positions in world coordinates
- `descriptors`: (N, D) feature descriptors for matching
- `track_ids`: (N,) unique identifiers for each landmark
- `last_seen_n_frames_ago`: (N,) observation tracking for filtering

### Module Organization

**`modules/initialization.py`**
- `initialize_vo_from_two_frames()`: Bootstrap VO from first two frames
- `estimate_pose_2d_to_2d()`: Essential matrix decomposition
- `triangulate_points()`: Linear triangulation using DLT

**`modules/feature_matching.py`**
- `detect_keypoints_and_descriptors()`: Grid-based feature detection for even distribution
- `match_descriptors()`: Lowe's ratio test + one-to-one correspondence enforcement
- `extract_and_match_features()`: Match current frame against landmark database
- `filter_existing_landmarks()`: Prevent duplicate landmarks by descriptor matching

**`modules/state_estimation.py`**
- `estimate_pose_pnp()`: PnP-RANSAC + nonlinear refinement
- `refine_pose_pnp()`: Levenberg-Marquardt optimization
- `compute_reprojection_error()`: Validate pose quality

**`modules/landmark_management.py`**
- `update_landmark_database()`: Main function for adding/filtering landmarks
- `validate_and_filter_new_landmarks()`: Consolidated quality filtering (depth, reprojection, scale)
- `update_observation_counts()`: Track when landmarks were last seen
- `filter_landmarks()`: Remove landmarks not seen for MAX_LAST_SEEN_FRAMES
- `filter_existing_landmarks()`: Prevent adding duplicate landmarks
- Adaptive keyframe selection based on current tracking quality (# matched landmarks)

**`modules/bundle_adjustment.py`**
- Uses **PyCeres** for non-linear optimization of camera poses and 3D points

**`modules/dataset_loader.py`**
- Loaders for KITTI, Malaga, Parking, and custom datasets
- Each loader provides camera intrinsics (K) and image paths

### Configuration

All tunable parameters are in `config/config.py`:

**Feature Detection**:
- `DETECT_TYPE`: Detector algorithm (FAST, SIFT, ORB)
- `DESCRIPT_TYPE`: Descriptor algorithm (SIFT, ORB)
- `GRID_ROWS`, `GRID_COLS`: Grid dimensions for spatial distribution (8×8 default)
- `FEATURES_PER_GRID_CELL`: Max features per cell (ensures even distribution)
- `FAST_THRESH`: FAST corner detection threshold

**Landmark Management**:
- `MIN_LANDMARKS_FOR_TRACKING`: Force keyframe if matched landmarks < this (default: 50)
- `TARGET_LANDMARKS`: Target number for adaptive keyframe selection (default: 200)
- `MAX_LAST_SEEN_FRAMES`: Remove landmarks not seen for N frames (default: 10)
- `MAX_REPROJECTION_ERROR_NEW_LANDMARKS`: Max reprojection error for new landmarks (pixels)
- `KEYFRAME_RATIO_THRESH`: Baseline/depth ratio for keyframe selection (default: 0.2)

**Geometry**:
- `RANSAC_PROB`, `RANSAC_THRESH_PIXELS`: RANSAC parameters
- `MIN_TRIANGULATION_DEPTH`: Minimum depth for valid triangulation (meters)
- `MAX_TRIANGULATION_DEPTH`: Maximum depth for valid landmarks (meters)

The system uses enums (`utils/enums.py`) for detector/descriptor type selection.

## Important Implementation Details

### Coordinate Frames
- **World frame**: First camera pose is identity (origin)
- **Camera frame**: Standard CV convention (right-hand, Y-down)
- Transformations are world→camera (R_cw, t_cw) for OpenCV compatibility

### Grid-Based Feature Detection
Uses **FAST detector with SIFT descriptors** in a grid-based approach:
1. Divides image into N×N grid (configurable via `GRID_ROWS`, `GRID_COLS`)
2. Detects FAST corners independently in each cell
3. Sorts by corner response strength and keeps top N per cell
4. Converts cell coordinates to global image coordinates
5. Computes SIFT descriptors for all selected keypoints

**Benefits**:
- Even spatial distribution across entire image
- Prevents feature clustering in high-texture areas
- Guarantees minimum feature coverage in low-texture regions
- Total features = `GRID_ROWS × GRID_COLS × FEATURES_PER_GRID_CELL`

### Feature Matching Strategy
- Two-stage matching: Lowe's ratio test + one-to-one enforcement
- One-to-one enforcement prevents multiple keypoints matching the same landmark
- Matches are sorted by distance before greedy unique selection
- Duplicate prevention: New features matched against database to avoid re-triangulating existing landmarks

### Triangulation and Depth Filtering

The pipeline uses **linear triangulation (DLT)** via SVD (`modules/initialization.py:148-212`), which minimizes algebraic error but doesn't enforce geometric constraints. This means triangulated points can end up behind the camera due to:

1. **Noise amplification**: Small pixel-level errors in 2D keypoints get amplified in 3D
2. **Outlier matches**: Some incorrect matches pass RANSAC within the pixel threshold
3. **Degenerate geometry**: Small baselines, distant points, or near-epipolar configurations
4. **Unconstrained optimization**: Linear DLT finds the least-squares solution without "points must be in front" constraint

**Why depth filtering is essential**: The depth checks at `landmark_management.py:396-398` remove physically impossible points (behind camera) and unreliable points (too close/far). These filters validate triangulation quality and prevent downstream errors in PnP pose estimation.

### Landmark Management & Quality Filtering

**Consolidated Validation** (`validate_and_filter_new_landmarks`):
1. **Depth filtering**: Rejects points behind camera or outside [MIN_TRIANGULATION_DEPTH, MAX_TRIANGULATION_DEPTH]
2. **Reprojection error**: Validates triangulation quality in both views (< 3 pixels)
3. **Scale consistency**: Currently disabled - can be enabled for scenes with consistent depth

**Adaptive Keyframe Selection**:
- **Force keyframe** if `num_matched_landmarks < MIN_LANDMARKS_FOR_TRACKING` (critical)
- **Relaxed threshold** if `num_matched_landmarks < TARGET_LANDMARKS` (0.5× threshold)
- **Standard threshold** otherwise (baseline_distance / median_depth > KEYFRAME_RATIO_THRESH)
- Uses **current frame tracking quality**, not total database size

**Key Insight**: Keyframe decisions based on how many landmarks are **actually matched in the current frame** (PnP inliers), not how many exist in the database. This ensures responsive keyframe creation when tracking degrades.

**Observation Tracking**:
- Landmarks not matched for MAX_LAST_SEEN_FRAMES are removed
- Prevents database bloat from out-of-view landmarks
- Counts reset to 0 when landmark successfully matched

### Scale Drift Prevention

Monocular VO is inherently scale-ambiguous and prone to drift. The pipeline includes multiple safeguards:

1. **Depth filtering during initialization**: Points outside [MIN_TRIANGULATION_DEPTH, MAX_TRIANGULATION_DEPTH] rejected before bundle adjustment
2. **Reprojection validation**: New landmarks must reproject accurately (< 3px error) in both views
3. **Duplicate prevention**: Features already in database are not re-triangulated
4. **Observation filtering**: Old landmarks removed to prevent accumulation of drift

**Optional Scale Consistency Check** (commented out by default):
- Compares new landmark depths vs existing landmark depths
- Uses Median Absolute Deviation (MAD) to distinguish scene transitions from bad triangulation
- Can be enabled for environments with consistent depth structure

**Future Improvements** (TODOs in code):
- Use existing landmark matches to validate scale during triangulation
- Directional scale validation (correlate scale changes with camera motion direction)

### PnP-RANSAC Flow
1. P3P-RANSAC finds inliers and initial pose (min 4 points required)
2. Levenberg-Marquardt refinement on inliers only
3. Reprojection error computed for validation
4. Fallback to identity pose if insufficient inliers (< 4)

**Important**: PnP estimates are only as good as the landmark quality. Poor triangulation → poor pose estimation → compounding drift.

### Visualization
- **Rerun**: 3D world with camera frustums, landmarks, and reprojection overlays
  - Green points = matched landmarks
  - Red points = unmatched landmarks
  - Blue points = detected 2D keypoints
  - Green overlays = reprojected 3D landmarks
- **cv2**: Simple 2D image display

## Dataset Structure

Datasets should be organized in `data/`:
```
data/
├── kitti/
│   └── 05/
│       └── image_0/
├── malaga/
├── parking/
└── my_dataset/
```

Each dataset loader handles camera calibration and image paths.

## Tuning Guide

### Insufficient Landmarks / Tracking Loss

**Symptoms**: PnP fails, not enough matched landmarks, tracking becomes unstable

**Solutions**:
1. Increase `FEATURES_PER_GRID_CELL` (10 → 20): More features per region
2. Decrease `MIN_LANDMARKS_FOR_TRACKING` (50 → 30): Force keyframes earlier
3. Increase `MAX_LAST_SEEN_FRAMES` (10 → 20): Keep landmarks longer
4. Adjust `FAST_THRESH` (10 → 5): Lower threshold detects more corners
5. Check `MAX_TRIANGULATION_DEPTH` is appropriate for scene (increase for outdoor/distant scenes)

### Too Many Features / Slow Performance

**Symptoms**: Processing is slow, too many descriptors being computed

**Solutions**:
1. Decrease `FEATURES_PER_GRID_CELL` (10 → 5): Fewer features per cell
2. Increase `GRID_ROWS`/`GRID_COLS` (8 → 6): Larger cells, fewer total features
3. Decrease `TARGET_LANDMARKS` (200 → 150): Less aggressive landmark creation

### Scale Drift / Inconsistent Depth

**Symptoms**: Scale changes dramatically between frames, unstable trajectory

**Solutions**:
1. Enable scale consistency check in `validate_and_filter_new_landmarks` (currently commented)
2. Decrease `MAX_TRIANGULATION_DEPTH` to reject far/uncertain points
3. Increase `MAX_REPROJECTION_ERROR_NEW_LANDMARKS` slightly if rejecting too many valid points
4. Ensure initialization has sufficient baseline (check keyframe ratio threshold)

### Scene Transitions (Close → Far)

**Symptoms**: No new landmarks added when moving from close to distant features

**Solutions**:
1. Increase `MAX_TRIANGULATION_DEPTH` to accommodate scene depth range
2. Keep scale consistency check **disabled** (current default)
3. Adjust `KEYFRAME_RATIO_THRESH` (0.2 → 0.15): More frequent keyframes

### High-Texture vs Low-Texture Scenes

**High-texture** (urban, indoor):
- Can use higher `FAST_THRESH` (10-20): Fewer but stronger corners
- Smaller `FEATURES_PER_GRID_CELL` (5-10): Plenty of features anyway

**Low-texture** (highway, sky):
- Lower `FAST_THRESH` (5-10): Detect weaker corners
- Higher `FEATURES_PER_GRID_CELL` (15-20): Need more features per region
- Consider increasing grid size for better coverage
