"""Configuration constants."""

import cv2

from utils.enums import DescriptorType, DetectorType

# Initialization
KEYFRAME_RATIO_THRESH = 0.2

# Matching
LOWES_RATIO = 0.8

# Tracking
GRID_ROWS = 8
GRID_COLS = 8
FEATURES_PER_GRID_CELL = 10  # Max features to keep per grid cell

# Detector / Descriptor
DETECT_TYPE = DetectorType.FAST
DESCRIPT_TYPE = DescriptorType.SIFT

# FAST
FAST_THRESH = 10

# SIFT
SIFT_NFEATURES = 1000  # Increased from 30 to ensure enough features
SIFT_NOCTAVES = 3
SIFT_CONTRASTTHRESH = 0.04
SIFT_EDGETHRESH = 10
SIFT_SIGMA = 1.6


# KLT
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}

# Geometry
RANSAC_PROB = 0.99
RANSAC_THRESH_PIXELS = 5.0

# Mapping
MIN_TRIANGULATION_DEPTH = 1.0
MAX_TRIANGULATION_DEPTH = 100.0

# Landmark Management
MAX_LAST_SEEN_FRAMES = 30
MAX_REPROJECTION_ERROR_NEW_LANDMARKS = 3.0  # pixels
MIN_LANDMARKS_FOR_TRACKING = 50  # Force keyframe if below this
TARGET_LANDMARKS = 200  # Ideal number of landmarks to maintain
SCALE_CONSISTENCY_SPREAD_THRESHOLD = 5
