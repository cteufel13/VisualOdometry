from dataclasses import dataclass


@dataclass
class VOConfig:
    """Configuration data class for the visual odometry pipeline."""

    # feature management
    num_features: int = 2000
    quality_level: float = 0.01
    min_distance: int = 10
    block_size: int = 7
    grid_rows: int = 4
    grid_cols: int = 6

    # tracking
    enable_fb_tracking: bool = False  # forward-backward check
    fb_error_thresh: float = 1.0  # max drift in pixels

    # initialization
    init_min_parallax: float = 20.0
    init_frame_step: int = 3  # frames to skip for init

    # triangulation
    min_ray_angle_deg: float = 1.0
    max_reproj_err: float = 2.0

    # pose estimation
    pnp_min_inliers: int = 10
    pnp_ransac_iter: int = 100

    # visualization
    viz_2d_candidates: bool = True
    viz_3d_landmarks: bool = True

    # optical flow
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
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
        cfg.quality_level = 0.001
        cfg.min_distance = 1
        cfg.block_size = 3
        cfg.init_min_parallax = 50.0
        cfg.init_frame_step = 3
        cfg.num_features = 3000
        cfg.max_reproj_err = 3.0
        cfg.pnp_ransac_iter = 100
        cfg.min_ray_angle_deg = 0.5
        cfg.pnp_min_inliers = 5
        cfg.lk_win_size = (21, 21)
        cfg.lk_max_level = 5

    elif dataset == "malaga":
        cfg.init_frame_step = 3
        cfg.min_distance = 1
        cfg.quality_level = 0.001
        cfg.block_size = 3
        cfg.num_features = 2000
        cfg.max_reproj_err = 9.0
        cfg.init_min_parallax = 30.0
        cfg.min_ray_angle_deg = 0.5
        cfg.pnp_min_inliers = 10
        cfg.lk_max_level = 5
        cfg.grid_rows = 6
        cfg.grid_cols = 4
        cfg.lk_win_size = (11, 11)
        cfg.min_distance = 10

    elif dataset == "parking":
        cfg.quality_level = 0.005
        cfg.init_frame_step = 4
        cfg.min_distance = 2
        cfg.block_size = 6
        cfg.num_features = 2500
        cfg.min_ray_angle_deg = 2.5
        cfg.max_reproj_err = 4.0
        cfg.min_ray_angle_deg = 2.0
        cfg.lk_win_size = (11, 11)
        cfg.block_size = 5

    return cfg
