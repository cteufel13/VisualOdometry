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

# Install pre-commit hooks
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
