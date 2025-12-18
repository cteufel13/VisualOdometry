import numpy as np
import cv2
import rerun as rr
from modules.backend import (
    detect_features,
    match_features,
    triangulate_and_filter,
)
from modules.optimizer import bundle_adjustment_window
from config.config import VOConfig


class VisualOdometry:
    """
    Robust Visual Odometry Pipeline.
    Includes 'Soft Reset' logic to continue trajectory after tracking loss.
    """

    def __init__(self, K: np.ndarray, config: VOConfig):
        self.K = K
        self.cfg = config
        self.frame_id = 0

        # state
        self.state = "INIT"
        self.T_wc = np.eye(4)
        self.T_wc_last_kf = np.eye(4)
        self.T_velocity = np.eye(4)

        # Soft Reset Memory
        self.last_known_T_wc = np.eye(4)  # Anchor for re-initialization

        # data
        self.map_points = {}
        self.prev_kps = np.empty((0, 2))
        self.prev_des = np.empty((0, 128))
        self.prev_idx_to_id = np.empty((0,), dtype=int)
        self.candidates = {}
        self.keyframes = []
        self.promoted_cids = {}

        # IDs
        self.next_pt_id = 0
        self.next_cand_id = 0

        # visualization
        self.trajectory_points = []

        # rerun setup
        rr.init("Visual Odometry", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    def get_T_cw(self) -> np.ndarray:
        R_wc = self.T_wc[:3, :3]
        t_wc = self.T_wc[:3, 3]
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_wc.T
        T_cw[:3, 3] = -R_wc.T @ t_wc
        return T_cw

    def process_frame(self, img: np.ndarray):
        """Main Loop"""

        # --- Preprocessing ---
        mask = np.full(img.shape, 255, dtype=np.uint8)
        mask[:100, :] = 0

        # 1. Detect
        curr_kps, curr_des = detect_features(img, self.cfg, mask=mask)
        rr.log("diagnostics/features/detected", rr.Scalars(len(curr_kps)))

        # Bootstrap for Frame 0
        if self.frame_id == 0:
            self.prev_kps = curr_kps
            self.prev_des = curr_des
            self.prev_idx_to_id = np.full(len(curr_kps), -1, dtype=int)
            for i in range(len(curr_kps)):
                self._create_candidate(i, curr_kps[i])
            self.frame_id += 1
            return

        # 2. Match
        match_idx_prev, match_idx_curr = match_features(
            self.prev_des, curr_des, self.cfg
        )
        rr.log("diagnostics/features/matches", rr.Scalars(len(match_idx_prev)))

        active_lm_ids = []
        active_lm_pts2d = []
        active_lm_pts3d = []
        active_cand_ids = []
        active_cand_pts2d = []
        flow_lines = []

        for i in range(len(match_idx_prev)):
            p_idx = match_idx_prev[i]
            c_idx = match_idx_curr[i]

            obj_id = self.prev_idx_to_id[p_idx]
            curr_pt = curr_kps[c_idx]
            prev_pt = self.prev_kps[p_idx]

            flow_lines.append([prev_pt, curr_pt])

            if obj_id >= 0:
                if obj_id in self.map_points:
                    active_lm_ids.append(obj_id)
                    active_lm_pts2d.append(curr_pt)
                    active_lm_pts3d.append(self.map_points[obj_id])
            elif obj_id < -1:
                cand_id = -1 * (obj_id + 2)
                if cand_id in self.candidates:
                    active_cand_ids.append(cand_id)
                    active_cand_pts2d.append(curr_pt)

        active_lm_ids = np.array(active_lm_ids)
        active_lm_pts2d = np.array(active_lm_pts2d)
        active_lm_pts3d = np.array(active_lm_pts3d)

        # 3. State Machine
        self.promoted_cids = {}

        if self.state == "INIT":
            self._process_initialization(active_cand_ids, active_cand_pts2d)
        elif self.state == "TRACKING":
            tracking_ok = self._process_tracking(
                active_lm_pts3d, active_lm_pts2d, active_lm_ids
            )

            # ONLY triangulate if tracking survived.
            # If tracking failed, it wiped self.candidates, so triangulation would crash.
            if tracking_ok:
                self._process_triangulation(active_cand_ids, active_cand_pts2d)

        # 4. Update ID Map
        next_idx_to_id = np.full(len(curr_kps), -1, dtype=int)

        for i in range(len(match_idx_prev)):
            p_idx = match_idx_prev[i]
            c_idx = match_idx_curr[i]
            prev_id_tag = self.prev_idx_to_id[p_idx]

            if prev_id_tag < -1:
                cid = -1 * (prev_id_tag + 2)
                if cid in self.promoted_cids:
                    next_idx_to_id[c_idx] = self.promoted_cids[cid]
                else:
                    next_idx_to_id[c_idx] = prev_id_tag
            else:
                next_idx_to_id[c_idx] = prev_id_tag

        # New Candidates
        mask_matched = np.zeros(len(curr_kps), dtype=bool)
        mask_matched[match_idx_curr] = True
        new_indices = np.where(~mask_matched)[0]

        for idx in new_indices:
            tag = self._create_candidate(idx, curr_kps[idx])
            next_idx_to_id[idx] = tag

        # Swap
        self.prev_kps = curr_kps
        self.prev_des = curr_des
        self.prev_idx_to_id = next_idx_to_id

        # 5. Visualization
        self._log_state(img, flow_lines, active_lm_pts2d)
        self.frame_id += 1

    def _create_candidate(self, idx, pt):
        cid = self.next_cand_id
        self.next_cand_id += 1
        self.candidates[cid] = {
            "first_pose": self.T_wc.copy(),
            "first_pt": pt,
        }
        return -1 * cid - 2

    def _process_tracking(self, lm_pts3d, lm_pts2d, lm_ids):
        """PnP + Tracking"""
        rr.log("diagnostics/pnp/input_points", rr.Scalars(len(lm_pts3d)))

        if len(lm_pts3d) < self.cfg.pnp_min_inliers:
            print(f"Tracking Lost: Low points {len(lm_pts3d)}. SOFT RESET.")

            # --- SOFT RESET LOGIC ---
            # 1. Save where we died
            self.last_known_T_wc = self.T_wc.copy()
            # 2. Clear candidates (they are mixed history, bad for 5-pt)
            self.candidates.clear()
            # 3. Reset velocity
            self.T_velocity = np.eye(4)
            # 4. Enter Init Mode
            self.state = "INIT"
            return False

        # Predict Pose
        T_guess = self.T_velocity @ self.T_wc
        T_guess_cw = np.linalg.inv(T_guess)

        r_vec_guess, _ = cv2.Rodrigues(T_guess_cw[:3, :3])
        r_vec_guess = np.ascontiguousarray(r_vec_guess, dtype=np.float64)
        t_vec_guess = T_guess_cw[:3, 3].reshape(3, 1)
        t_vec_guess = np.ascontiguousarray(t_vec_guess, dtype=np.float64)

        lm_pts3d = np.ascontiguousarray(lm_pts3d, dtype=np.float64)
        lm_pts2d = np.ascontiguousarray(lm_pts2d, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            lm_pts3d,
            lm_pts2d,
            self.K,
            None,
            rvec=r_vec_guess,
            tvec=t_vec_guess,
            useExtrinsicGuess=True,
            iterationsCount=self.cfg.pnp_ransac_iter,
            reprojectionError=self.cfg.max_reproj_err,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        inlier_count = len(inliers) if inliers is not None else 0
        rr.log("diagnostics/pnp/inliers", rr.Scalars(inlier_count))

        if success and inlier_count > self.cfg.pnp_min_inliers:
            R_pnp, _ = cv2.Rodrigues(rvec)
            t_pnp = tvec.flatten()

            T_wc_prev = self.T_wc.copy()
            self.T_wc = np.eye(4)
            self.T_wc[:3, :3] = R_pnp.T
            self.T_wc[:3, 3] = -R_pnp.T @ t_pnp

            self.T_velocity = self.T_wc @ np.linalg.inv(T_wc_prev)

            # Keyframe Logic (Parallax)
            R_last = self.T_wc_last_kf[:3, :3]
            C_last = self.T_wc_last_kf[:3, 3]
            C_curr = self.T_wc[:3, 3]

            lm_inliers = lm_pts3d[inliers.flatten()]
            v_last = lm_inliers - C_last
            v_curr = lm_inliers - C_curr

            n_last = np.linalg.norm(v_last, axis=1, keepdims=True)
            n_curr = np.linalg.norm(v_curr, axis=1, keepdims=True)
            v_last = v_last / (n_last + 1e-8)
            v_curr = v_curr / (n_curr + 1e-8)

            dots = np.sum(v_last * v_curr, axis=1)
            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            median_parallax = np.median(angles) if len(angles) > 0 else 0.0

            rr.log("diagnostics/kf/parallax_deg", rr.Scalars(median_parallax))

            is_movement = median_parallax > 2.0
            is_starving = inlier_count < 1000

            # Strict Keyframe Addition to avoid Degeneracy
            if is_movement or (is_starving and median_parallax > 1.0):
                self._add_keyframe(
                    lm_ids[inliers.flatten()], lm_pts2d[inliers.flatten()]
                )
                self.T_wc_last_kf = self.T_wc.copy()
                self._optimize_window()

            return True
        else:
            self.T_wc = T_guess
            rr.log("diagnostics/pnp/status", rr.Scalars(0))  # 0 = coasting
            return False

    def _process_triangulation(self, cand_ids, cand_pts2d):
        """Triangulate surviving candidates"""
        T_cw_curr = self.get_T_cw()
        rr.log("diagnostics/triangulation/candidates", rr.Scalars(len(cand_ids)))

        new_pts_count = 0
        angles = []

        for i, cid in enumerate(cand_ids):
            c_data = self.candidates[cid]
            T_wc_first = c_data["first_pose"]
            pt_first = c_data["first_pt"]
            pt_curr = cand_pts2d[i]

            R_f = T_wc_first[:3, :3]
            t_f = T_wc_first[:3, 3]
            T_cw_first = np.eye(4)
            T_cw_first[:3, :3] = R_f.T
            T_cw_first[:3, 3] = -R_f.T @ t_f

            pts3d, mask, stats = triangulate_and_filter(
                T_cw_first,
                T_cw_curr,
                np.array([pt_first]),
                np.array([pt_curr]),
                self.K,
                self.cfg,
            )

            if mask[0]:
                new_id = self.next_pt_id
                self.next_pt_id += 1
                self.map_points[new_id] = pts3d[0]
                del self.candidates[cid]
                self.promoted_cids[cid] = new_id
                new_pts_count += 1
                angles.append(stats["avg_angle"])

        rr.log("diagnostics/triangulation/new_points", rr.Scalars(new_pts_count))
        if len(angles) > 0:
            rr.log("diagnostics/triangulation/avg_angle", rr.Scalars(np.mean(angles)))

    def _process_initialization(self, cand_ids, cand_pts2d):
        if len(cand_ids) < 50:
            return

        pts1, pts2, used_cids = [], [], []
        anchor_T_wc = self.candidates[cand_ids[0]]["first_pose"]

        for i, cid in enumerate(cand_ids):
            pts1.append(self.candidates[cid]["first_pt"])
            pts2.append(cand_pts2d[i])
            used_cids.append(cid)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        disp = np.linalg.norm(pts1 - pts2, axis=1)
        avg_disp = np.mean(disp)
        rr.log("diagnostics/init/avg_disp", rr.Scalars(avg_disp))

        if avg_disp < self.cfg.init_min_parallax:
            return

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        # T_rel is the motion from View 1 (Anchor) to View 2 (Current)
        # BUT: recoverPose gives R,t such that x2 = R*x1 + t
        # This means T_c2_c1 (motion of points 1->2).
        # We want the World Pose of 2.
        # T_w2 = T_w1 * T_12
        # T_12 = T_c2_c1^-1

        T_rel_points = np.eye(4)
        T_rel_points[:3, :3] = R
        T_rel_points[:3, 3] = t.flatten()

        T_c2_c1 = T_rel_points
        T_c1_c2 = np.linalg.inv(T_c2_c1)

        # Apply to Anchor
        # self.T_wc = anchor_T_wc @ T_c1_c2

        # Note: T_wc is the CAMERA POSE in WORLD Frame.
        # T_wc_curr = T_wc_prev * T_prev_curr
        # recoverPose gives R,t.
        # R, t correspond to T_c2_c1.
        # So T_c1_c2 = inv(T_c2_c1).
        # T_w_c2 = T_w_c1 * T_c1_c2.

        self.T_wc = anchor_T_wc @ T_c1_c2

        # Triangulate
        T_cw_curr = self.get_T_cw()

        # Anchor Camera Pose (inverse of Anchor World Pose)
        R_a = anchor_T_wc[:3, :3]
        t_a = anchor_T_wc[:3, 3]
        T_cw_anchor = np.eye(4)
        T_cw_anchor[:3, :3] = R_a.T
        T_cw_anchor[:3, 3] = -R_a.T @ t_a

        pts3d, mask_tri, _ = triangulate_and_filter(
            T_cw_anchor, T_cw_curr, pts1, pts2, self.K, self.cfg
        )

        cnt = 0
        for i, is_valid in enumerate(mask_tri):
            if is_valid:
                new_id = self.next_pt_id
                self.next_pt_id += 1
                self.map_points[new_id] = pts3d[cnt]
                self.promoted_cids[used_cids[i]] = new_id
                del self.candidates[used_cids[i]]
                cnt += 1

        print(f"Re-Initialized with {cnt} points (Continued Trajectory)")
        self.state = "TRACKING"
        self._add_keyframe([], [])
        self.T_wc_last_kf = self.T_wc.copy()

    def _add_keyframe(self, active_ids, active_pts):
        kf = {
            "T_cw": self.get_T_cw(),
            "ids": active_ids,
            "pts_2d": active_pts,
        }
        self.keyframes.append(kf)
        if len(self.keyframes) > 7:
            self.keyframes.pop(0)

    def _optimize_window(self):
        if not self.cfg.enable_window_ba or len(self.keyframes) < 3:
            return

        rr.log("diagnostics/ba/map_size_pre", rr.Scalars(len(self.map_points)))

        self.keyframes, self.map_points, pruned = bundle_adjustment_window(
            self.keyframes, self.map_points, self.K
        )

        rr.log("diagnostics/ba/pruned", rr.Scalars(pruned))
        rr.log("diagnostics/ba/map_size_post", rr.Scalars(len(self.map_points)))

    def _log_state(self, img, flow_lines, active_pts):
        rr.set_time_sequence("frame", self.frame_id)

        rr.log(
            "world/camera",
            rr.Transform3D(translation=self.T_wc[:3, 3], mat3x3=self.T_wc[:3, :3]),
        )

        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=self.K, width=img.shape[1], height=img.shape[0]
            ),
        )
        rr.log("world/camera/image", rr.Image(img))

        if len(flow_lines) > 0:
            rr.log(
                "world/camera/image/flow",
                rr.LineStrips2D(flow_lines, radii=0.5, colors=[0, 255, 255]),
            )

        if len(active_pts) > 0:
            rr.log(
                "world/camera/image/landmarks",
                rr.Points2D(active_pts, colors=[0, 255, 0], radii=3),
            )

        self.trajectory_points.append(self.T_wc[:3, 3])
        if len(self.trajectory_points) > 1:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([self.trajectory_points], colors=[[0, 255, 255]]),
            )

        if len(self.map_points) > 0:
            pts = np.array(list(self.map_points.values()))
            rr.log("world/landmarks", rr.Points3D(pts, colors=[0, 255, 0]))
