import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from config.config import VOConfig


class FeatureFrontend:
    def __init__(self, config: VOConfig):
        self.conf = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # init based on config type
        if self.conf.extractor_type == "superpoint":
            self.extractor = (
                SuperPoint(max_num_keypoints=config.max_keypoints)
                .eval()
                .to(self.device)
            )
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        elif self.conf.extractor_type == "sift":
            # inti SIFT with custom configs
            self.extractor = cv2.SIFT_create(
                nfeatures=config.sift_n_features,
                contrastThreshold=config.sift_contrast_threshold,
                edgeThreshold=config.sift_edge_threshold,
                sigma=config.sift_sigma,
            )
            # brute force matcher with l2 norm
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def process_image(self, img: np.ndarray) -> dict:
        """Extract features from an image."""
        if self.conf.extractor_type == "superpoint":
            # add channel dimension if missing
            if len(img.shape) == 2:
                img = img[..., None]

            # normalize and permute to (1, C, H, W)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feats = self.extractor.extract(img_tensor)
            return feats

        if self.conf.extractor_type == "sift":
            # grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            kps, des = self.extractor.detectAndCompute(gray, None)

            # handle keypoints
            if kps:
                pts = np.array([k.pt for k in kps], dtype=np.float32)
                des = np.array(des, dtype=np.float32)
            else:
                pts = np.empty((0, 2), dtype=np.float32)
                des = np.empty((0, 128), dtype=np.float32)

            # shape (1, N, 2)
            pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(self.device)
            des_tensor = torch.from_numpy(des).unsqueeze(0).to(self.device)

            return {
                "keypoints": pts_tensor,  # shape (1, N, 2)
                "descriptors": des_tensor,  # shape (1, N, D)
                "image_size": torch.tensor([(img.shape[1], img.shape[0])]).to(
                    self.device
                ),
            }
        return None

    def match_frames(self, feats0: dict, feats1: dict) -> np.ndarray:
        """Match features between two frames."""
        if self.conf.extractor_type == "superpoint":
            with torch.no_grad():
                matches01 = self.matcher({"image0": feats0, "image1": feats1})
                matches = matches01["matches"][0]
            return matches.cpu().numpy()

        if self.conf.extractor_type == "sift":
            # get escriptors
            des0 = feats0["descriptors"][0]
            des1 = feats1["descriptors"][0]

            # convert to numpy
            if isinstance(des0, torch.Tensor):
                des0 = des0.cpu().numpy()
            if isinstance(des1, torch.Tensor):
                des1 = des1.cpu().numpy()

            if des0 is None or des1 is None or len(des0) == 0 or len(des1) == 0:
                return np.empty((0, 2), dtype=int)

            # knn wiith lowe ratio test
            matches_knn = self.matcher.knnMatch(des0, des1, k=2)

            good_matches = []
            ratio_thresh = 0.75
            for m_n in matches_knn:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append([m.queryIdx, m.trainIdx])

            return np.array(good_matches, dtype=int)
        return None


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

    # triangulate
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
    mask_reproj = err2 < (config.max_reproj_err)

    # combine masks
    final_mask = mask_pos_depth & mask_reproj

    return pts3d[final_mask], final_mask
