"""Configuration constants."""

import cv2

from utils.enums import DescriptorType, DetectorType

# Initialization
KEYFRAME_RATIO_THRESH = 0.2

# Matching
LOWES_RATIO = 0.8

# Tracking
MIN_FEATURES = 2000
GRID_ROWS = 8
GRID_COLS = 8
MAX_POINT_LANDMARKS = 1000

# Detector / Descriptor
DETECT_TYPE = DetectorType.FAST
DESCRIPT_TYPE = DescriptorType.SIFT

# FAST
FAST_THRESH = 20

# SIFT
SIFT_NFEATURES = 30
SIFT_NOCTAVES = 3
SIFT_CONTRASTTHRESH = 0.04
SIFT_EDGETHRESH = 10
SIFT_SIGMA = 10


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
MIN_PARALLAX = 1.0
MAX_DEPTH = 500.0

# Landmark Management
MAX_LAST_SEEN_FRAMES = 20
