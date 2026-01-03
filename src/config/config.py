from dataclasses import dataclass


@dataclass
class VOConfig:
    """Configuration data class for the VO pipeline."""

    # extractor selection
    extractor_type: str = "superpoint"  # superpoint or sift

    # global scaling
    global_scale: float = 50.0

    # feature extractor
    max_keypoints: int = 2048
    device: str = "cuda"

    # SIFT configs
    sift_n_features: int = 2048
    sift_contrast_threshold: float = 0.03
    sift_edge_threshold: float = 1.0
    sift_sigma: float = 1.6

    # initialization & keyframes
    min_parallax: float = 20.0
    min_inliers: int = 10
    init_ransac_prob: float = 0.999
    init_ransac_thresh: float = 1.0

    # triangulation & depth
    min_depth: float = 0.001
    max_reproj_err: float = 6.0
    reproj_err_relax: float = 1.5  # multiplier for relaxed check

    # pnp and tracking
    pnp_reproj_err: float = 4.0
    kf_min_tracked: int = 80

    # speed logic
    init_speed_min: float = 0.1
    init_speed_max: float = 5.0
    turn_thresh: float = 0.01  # rad
    move_thresh: float = 0.01
    turn_smoothing: float = 0.7
    trans_smoothing: float = 0.6
    baseline_lr: float = 0.01

    # smoothing limits
    scale_clamp_min: float = 0.5
    scale_clamp_max: float = 3.0


def get_config(dataset: str) -> VOConfig:
    """Get config based on dataset."""
    cfg = VOConfig()
    # tuning
    if dataset == "kitti":
        cfg.min_parallax = 40.0
        cfg.max_keypoints = 2048
        cfg.max_reproj_err = 5.0
        cfg.pnp_reproj_err = 1.0
        cfg.baseline_lr = 0.01
        cfg.turn_smoothing = 0.2
        cfg.trans_smoothing = 0.2

        # SIFT tuning
        if cfg.extractor_type == "sift":
            cfg.sift_n_features = 4000
            cfg.sift_contrast_threshold = 0.02
            cfg.sift_edge_threshold = 2.0
            cfg.max_reproj_err = 5.0
            cfg.pnp_reproj_err = 1.0
            cfg.turn_smoothing = 0.2
            cfg.trans_smoothing = 0.4

    elif dataset == "malaga":
        cfg.min_parallax = 30.0
        cfg.max_keypoints = 2048
        cfg.max_reproj_err = 5.0
        cfg.pnp_reproj_err = 2.0
        cfg.baseline_lr = 0.01
        cfg.turn_smoothing = 0.95
        cfg.trans_smoothing = 0.1

        if cfg.extractor_type == "sift":
            cfg.sift_n_features = 3000
            cfg.sift_contrast_threshold = 0.01
            cfg.sift_edge_threshold = 2.0
            cfg.min_parallax = 4.0

    elif dataset == "parking":
        cfg.min_parallax = 3.0
        cfg.max_reproj_err = 2.0
        cfg.pnp_reproj_err = 1.0

        if cfg.extractor_type == "sift":
            cfg.sift_n_features = 3000
            cfg.sift_contrast_threshold = 0.01
            cfg.sift_edge_threshold = 2.0
            cfg.min_parallax = 4.0

    return cfg
