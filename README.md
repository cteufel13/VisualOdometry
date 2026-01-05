# Robust Monocular Visual Odometry

This repository implements a Monocular Visual Odometry (VO) pipeline.

## Quick Start

### 1. Installation

**Option A: Using uv (recommended)**

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

**Option B: Using conda**

Alternatively, you can use conda with the provided `environment.yml` file.

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate visual-odometry

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
# Run with default settings (KITTI sequence 05)
python src/main.py

# Run on a different dataset
python src/main.py --dataset parking

```

### 4\. Additional Details
All screencasts were run on a machine with an i7-14700KF CPU (8 performance cores @ 3.4 GHz, 12 efficiency cores @ 2.5 GHz), a RTX 4080 GPU and 64GB of RAM.

During runtime, our pipeline consumes 55 threads.
