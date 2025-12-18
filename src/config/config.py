from dataclasses import dataclass


@dataclass
class VOConfig:
    """Configuration data class for the visual odometry pipeline."""

    # feature management
    num_features: int = 2000

    # sift specific settings
    # contrast threshold: lower = more features but more noise
    sift_contrast_thresh: float = 0.05
    # edge threshold: higher = allow more edge like features
    sift_edge_thresh: float = 20

    # lowe matching ratio
    match_ratio_thresh: float = 0.7

    # initialization
    init_min_parallax: float = 10.0
    init_frame_step: int = 2  # frames to skip for init

    # triangulation
    min_ray_angle_deg: float = 0.5
    max_reproj_err: float = 5.0

    # pose estimation
    pnp_min_inliers: int = 10
    pnp_ransac_iter: int = 100

    # visualization
    viz_2d_candidates: bool = True
    viz_3d_landmarks: bool = True
    enable_window_ba: bool = True


def get_config(dataset: str) -> VOConfig:
    """
    Return the specific configuration for a dataset.

    Args:
        dataset: Name of the dataset (kitti, malaga, parking).

    Returns:
        The configuration object with dataset-specific overrides.

    """
    cfg = VOConfig()

    if dataset == "kitti":
        cfg.sift_contrast_thresh = 0.03
        cfg.init_min_parallax = 15.0
        cfg.num_features = 2000
        cfg.match_ratio_thresh = 0.70
        cfg.pnp_min_inliers = 10
        cfg.max_reproj_err = 4.0
        cfg.min_ray_angle_deg = 1
        cfg.sift_edge_thresh = 20

    elif dataset == "malaga":
        cfg.sift_contrast_thresh = 0.015
        cfg.num_features = 3000
        cfg.min_ray_angle_deg = 0.5
        cfg.max_reproj_err = 4.0

    elif dataset == "parking":
        cfg.sift_contrast_thresh = 0.02
        cfg.init_frame_step = 1
        cfg.match_ratio_thresh = 0.8

    return cfg
