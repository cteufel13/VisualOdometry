import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from config.config import VOConfig


class FeatureFrontend:
    def __init__(self, config: VOConfig):
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.extractor = (
            SuperPoint(max_num_keypoints=config.max_keypoints).eval().to(self.device)
        )
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.conf = config

    def process_image(self, img: np.ndarray) -> dict:
        """Extract features from an image."""
        # add channel dimension if missing
        if len(img.shape) == 2:
            img = img[..., None]

        # normalize and permute to (1, C, H, W)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(img_tensor)
        return feats

    def match_frames(self, feats0: dict, feats1: dict) -> np.ndarray:
        """Match features between two frames."""
        with torch.no_grad():
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
            matches = matches01["matches"][0]
        return matches.cpu().numpy()


def triangulate_points(
    T_cw1: np.ndarray,
    T_cw2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    config: VOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate 3D points and filter outliers."""
    if len(pts1) == 0:
        return np.empty((0, 3)), np.zeros(0, dtype=bool)

    # 1. triangulate
    P1 = K @ T_cw1[:3, :]
    P2 = K @ T_cw2[:3, :]
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # check point is in front of camera
    pts3d_c2 = (T_cw2[:3, :3] @ pts3d.T + T_cw2[:3, 3:4]).T
    mask_pos_depth = pts3d_c2[:, 2] > config.min_depth

    # allow slightly higher error during rotation
    R2, t2 = T_cw2[:3, :3], T_cw2[:3, 3]
    proj_pts2, _ = cv2.projectPoints(pts3d, R2, t2, K, None)
    err2 = np.linalg.norm(proj_pts2.reshape(-1, 2) - pts2, axis=1)

    # use relaxed error threshold from config
    mask_reproj = err2 < (config.max_reproj_err * config.reproj_err_relax)

    # combine masks
    final_mask = mask_pos_depth & mask_reproj

    return pts3d[final_mask], final_mask
