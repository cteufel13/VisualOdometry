# Robust Monocular Visual Odometry

This repository implements a Monocular Visual Odometry (VO) pipeline.

## Quick Start

### 1. Installation

We use `uv` for fast, modern Python package management.

```bash
# Install uv (if you haven't already)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Sync environment (installs dependencies from pyproject.toml)
uv sync

# Activate environment
source .venv/bin/activate
```

### 2\. Data Setup

Organize datasets in the `data/` directory structure:

```
data/
├── kitti/
│   └── 05/
│       └── image_0/
├── malaga/
├── parking/
└── my_dataset/
```

### 3\. Execution

Run the pipeline:

```bash
uv run src/main.py
```

## System Architecture


  * **`config.py`**: Centralized configuration parameters (constants only)
  * **`datatypes.py`**: Data classes (`Frame`, `MapPoint`, `State`) with no complex logic
  * **`tracking.py`**: 2D feature detection (grid-based) and optical flow tracking
  * **`geometry.py`**: 3D math, PnP RANSAC, and triangulation
  * **`backend.py`**: Map maintenance (culling, new point creation)
  * **`vo.py`**: Class that holds persistent state and calls the functional modules
  * **`dataset_loader.py`**: Loaders for different dataset formats.
  * **`main.py`**: Entry point, loop, and visualization
