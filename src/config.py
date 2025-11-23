"""Configuration constants."""

import cv2

# Tracking
MIN_FEATURES = 2000
GRID_ROWS = 8
GRID_COLS = 8
FAST_THRESH = 20

# KLT
LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}

# Geometry
MIN_INLIERS = 15
RANSAC_PROB = 0.99
RANSAC_THRESH = 1.0

# Mapping
MIN_PARALLAX = 1.0
MAX_DEPTH = 500.0
