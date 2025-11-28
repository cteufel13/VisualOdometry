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

# (if developing) Install pre-commit hooks
pre-commit install
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

We use `tyro` for argument parsing. You can view all available options and defaults using the help flag:

```bash
python src/main.py -h
```

**Common Examples:**

```bash
# Run with default settings (KITTI sequence 05, Rerun visualization)
python src/main.py

# Run with cv2 visualization instead of Rerun
python src/main.py --cv2_viz

# Run on a different dataset
python src/main.py --dataset parking

# Run KITTI with a specific sequence
python src/main.py --dataset kitti --sequence 06
```
