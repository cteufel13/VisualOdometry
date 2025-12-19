import numpy as np
import cv2
import rerun as rr
from modules.backend import (
    detect_new_features,
    track_optical_flow,
    triangulate_and_filter,
)
from modules.optimizer import bundle_adjustment_window
from config.config import VOConfig


class VisualOdometry:
    """
    Cleaned Visual Odometry Pipeline.
    Philosophy: Robust Tracking, Conservative Mapping.
    """

    def __init__(self, K: np.ndarray, config: VOConfig):
        self.K = K
        self.cfg = config
        self.frame_id = 0
        self.prev_img = None

        # state
        self.state = "INIT"  # INIT, TRACKING
        self.T_wc = np.eye(4)  # Current Pose
        self.T_wc_last_kf = np.eye(4)  # pose of last keyframe
        self.T_velocity = np.eye(4)
        # map {id: [x,y,z]}
        self.map_points = {}

        # active tracking arrays
        self.lm_pts3d = np.empty((0, 3))  # 3D coords of tracked points
        self.lm_pts2d = np.empty((0, 2))  # 2D coords in current frame
        # NEW: Store previous location of tracked points for 2D recovery
        self.prev_lm_pts2d = np.empty((0, 2))
        self.lm_ids = np.empty((0,), dtype=int)  # Global IDs

        # candidates
        self.cand_curr_2d = np.empty((0, 2))
        self.cand_first_2d = np.empty((0, 2))
        self.cand_first_pose_wc = []  # Pose where candidate was first seen

        # Keyframe History for BA
        self.keyframes = []

        # ID generation
        self.next_pt_id = 0

        # Visualization
        self.trajectory_points = []

        self.probation_counter = 0

        # Rerun setup
        rr.init("Visual Odometry", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    def get_T_cw(self) -> np.ndarray:
        """Helper: World -> Camera"""
        R_wc = self.T_wc[:3, :3]
        t_wc = self.T_wc[:3, 3]
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_wc.T
        T_cw[:3, 3] = -R_wc.T @ t_wc
        return T_cw

    def process_frame(self, img: np.ndarray):
        """Main Loop"""

        # # Apply CLAHE to extract features from blurry/dark regions
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # img_enhanced = clahe.apply(img)

        # Use img_enhanced for EVERYTHING (Tracking + Feature Detection)
        # img = img_enhanced
        if self.prev_img is None:
            self.prev_img = img
            # Bootstrap: Find features, don't map yet
            self.cand_curr_2d = detect_new_features(img, None, self.cfg)
            self.cand_first_2d = self.cand_curr_2d.copy()
            curr_pose = self.T_wc.copy()
            self.cand_first_pose_wc = [curr_pose for _ in range(len(self.cand_curr_2d))]
            self.frame_id += 1
            return

        self._track_features(img)

        # state machine
        if self.state == "INIT":
            self._process_initialization()
        elif self.state == "TRACKING":
            self._process_tracking()
            # replenish features in empty regions so we don't go blind during turns
            self._replenish_features(img)

        # visualization
        self._log_state(img)

        # Update previous frame
        self.prev_img = img
        self.frame_id += 1

    def _track_features(self, img: np.ndarray):
        """
        KLT Tracking with Motion Compensation.
        Crucial: Calculates flow guess to help KLT during fast turns.
        """
        # motion compensation
        flow_guess = None
        if len(self.lm_pts3d) > 0 and self.state == "TRACKING":
            # apply constant velocity model
            T_pred = self.T_velocity @ self.T_wc
            R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
            r_vec, _ = cv2.Rodrigues(R_pred.T)
            t_vec = -R_pred.T @ t_pred

            # Project 3D points to getting expected 2D locations
            pts2d_pred, _ = cv2.projectPoints(self.lm_pts3d, r_vec, t_vec, self.K, None)
            pts2d_pred = pts2d_pred.reshape(-1, 2)
            flow_guess = pts2d_pred - self.lm_pts2d

        # track landmarks
        valid_lm = np.zeros(len(self.lm_pts2d), dtype=bool)
        if len(self.lm_pts2d) > 0:
            # Save the state before tracking updates it, so we have (Frame k-1, Frame k) pairs
            self.prev_lm_pts2d = self.lm_pts2d.copy()
            p2, valid_lm = track_optical_flow(
                self.prev_img, img, self.lm_pts2d, self.cfg, flow_guess=flow_guess
            )

            # flow vector logging
            if np.any(valid_lm):
                flow_vecs = p2[valid_lm] - self.lm_pts2d[valid_lm]
                flow_mags = np.linalg.norm(flow_vecs, axis=1)

                rr.log("diagnostics/flow/avg_mag", rr.Scalars(np.mean(flow_mags)))
                rr.log(
                    "world/camera/image/flow_vectors",
                    rr.Arrows2D(
                        origins=self.lm_pts2d[valid_lm],
                        vectors=flow_vecs,
                        colors=[0, 255, 255],
                    ),
                )

            # Filter current points
            self.lm_pts2d = p2[valid_lm]
            self.lm_pts3d = self.lm_pts3d[valid_lm]
            self.lm_ids = self.lm_ids[valid_lm]

            # Ensure prev matches before filtering
            if len(self.prev_lm_pts2d) == len(valid_lm):
                self.prev_lm_pts2d = self.prev_lm_pts2d[valid_lm]
            else:
                # Desync fallback
                self.prev_lm_pts2d = self.lm_pts2d.copy()

        # C. Track Candidates (2D - waiting for init)
        if len(self.cand_curr_2d) > 0:
            # We use the average flow of landmarks to guess candidate flow if possible
            cand_guess = None
            if flow_guess is not None and len(flow_guess) > 0:
                avg_flow = (
                    np.mean(flow_guess[valid_lm], axis=0)
                    if np.any(valid_lm)
                    else np.array([0.0, 0.0])
                )
                cand_guess = np.tile(avg_flow, (len(self.cand_curr_2d), 1))

            c2, valid_cand = track_optical_flow(
                self.prev_img, img, self.cand_curr_2d, self.cfg, flow_guess=cand_guess
            )

            self.cand_curr_2d = c2[valid_cand]
            self.cand_first_2d = self.cand_first_2d[valid_cand]
            # List comprehension to filter the pose list
            self.cand_first_pose_wc = [
                self.cand_first_pose_wc[i] for i, v in enumerate(valid_cand) if v
            ]

    def _process_initialization(self):
        """Standard 5-Point Algorithm Initialization"""
        # Check parallax
        if len(self.cand_curr_2d) < 50:
            return  # Wait for more points

        disp = np.linalg.norm(self.cand_curr_2d - self.cand_first_2d, axis=1)
        if np.mean(disp) < self.cfg.init_min_parallax:
            return  # Not enough movement yet

        # 1. Essential Matrix
        E, mask = cv2.findEssentialMat(
            self.cand_first_2d,
            self.cand_curr_2d,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            return

        # 2. Recover Pose
        _, R, t, _ = cv2.recoverPose(E, self.cand_first_2d, self.cand_curr_2d, self.K)

        # 3. Construct Pose (Camera 2 in World)
        # Note: recoverPose gives R, t from 1 to 2.
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()

        # Assume Frame 0 was at Identity. Frame k is at T_motion_inv
        T_motion = np.linalg.inv(T_rel)
        self.T_wc = T_motion

        # 4. Triangulate Initial Map
        T_cw_init = self.get_T_cw()
        pts3d, mask_tri, _ = triangulate_and_filter(
            np.eye(4),
            T_cw_init,  # T_cw1, T_cw2
            self.cand_first_2d,
            self.cand_curr_2d,
            self.K,
            self.cfg,
        )

        # 5. Store
        self.lm_pts3d = pts3d[mask_tri]
        self.lm_pts2d = self.cand_curr_2d[mask_tri]

        # Initialize prev_pts logic for next frame
        self.prev_lm_pts2d = self.lm_pts2d.copy()

        # Generate IDs
        num_new = len(self.lm_pts3d)
        self.lm_ids = np.arange(self.next_pt_id, self.next_pt_id + num_new)
        self.next_pt_id += num_new

        # 6. Create First Keyframe
        self._add_keyframe()

        self.state = "TRACKING"
        print(f"initialized with {num_new} points.")
        self.T_wc_last_kf = self.T_wc.copy()

    def _process_tracking(self):
        """
        Robust Tracking Logic: Parallax-Gated Keyframe Selection.
        Prevents "Bush Explosions" by ensuring geometric baseline before mapping.
        """
        if len(self.lm_pts3d) < 6:
            print("CRITICAL: Lost Tracking. Resetting.")
            self.state = "INIT"
            return

        T_wc_prev = self.T_wc.copy()

        # --- 1. Pose Estimation (PnP) ---
        T_guess = self.T_velocity @ self.T_wc
        R_guess = T_guess[:3, :3]
        t_guess = T_guess[:3, 3]

        R_guess_cw = R_guess.T
        t_guess_cw = -R_guess.T @ t_guess
        r_vec, _ = cv2.Rodrigues(R_guess_cw)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.lm_pts3d,
            self.lm_pts2d,
            self.K,
            None,
            rvec=r_vec,
            tvec=t_guess_cw,
            useExtrinsicGuess=True,
            iterationsCount=self.cfg.pnp_ransac_iter,
            reprojectionError=self.cfg.max_reproj_err,
        )

        tracking_reliable = False

        if success and len(inliers) > self.cfg.pnp_min_inliers:
            # --- Path A: Tracking Good ---
            R_pnp, _ = cv2.Rodrigues(rvec)
            t_pnp = tvec.flatten()

            self.T_wc = np.eye(4)
            self.T_wc[:3, :3] = R_pnp.T
            self.T_wc[:3, 3] = -R_pnp.T @ t_pnp

            # Update Velocity
            self.T_velocity = self.T_wc @ np.linalg.inv(T_wc_prev)

            # Filter Outliers
            inliers = inliers.flatten()
            self.lm_pts3d = self.lm_pts3d[inliers]
            self.lm_pts2d = self.lm_pts2d[inliers]
            self.lm_ids = self.lm_ids[inliers]

            # Sync prev_pts
            if len(self.prev_lm_pts2d) == len(inliers):
                self.prev_lm_pts2d = self.prev_lm_pts2d[inliers]
            else:
                self.prev_lm_pts2d = self.lm_pts2d.copy()

            tracking_reliable = True

            if self.probation_counter > 0:
                print(f"Tracking recovering... Probation: {self.probation_counter}")
                self.probation_counter -= 1

        else:
            # --- Path B: Coasting ---
            print("PnP Failed. Coasting (No Mapping).")
            self.T_wc = T_guess
            tracking_reliable = False
            self.probation_counter = 5

        # --- 2. Geometric Parallax Gate ---
        median_parallax = 0.0

        if len(self.keyframes) > 0 and len(self.lm_pts3d) > 0:
            last_kf = self.keyframes[-1]
            R_ref = last_kf["T_cw"][:3, :3]
            t_ref = last_kf["T_cw"][:3, 3]
            C_ref = -R_ref.T @ t_ref
            C_curr = self.T_wc[:3, 3]

            vecs_ref = self.lm_pts3d - C_ref
            vecs_curr = self.lm_pts3d - C_curr
            norms_ref = np.linalg.norm(vecs_ref, axis=1, keepdims=True)
            norms_curr = np.linalg.norm(vecs_curr, axis=1, keepdims=True)

            valid = (norms_ref > 0.01) & (norms_curr > 0.01)
            valid = valid.flatten()

            if np.sum(valid) > 0:
                v_ref = vecs_ref[valid] / norms_ref[valid]
                v_curr = vecs_curr[valid] / norms_curr[valid]
                dots = np.sum(v_ref * v_curr, axis=1)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.degrees(np.arccos(dots))
                median_parallax = np.median(angles)

        rr.log("diagnostics/kf/median_parallax", rr.Scalars(median_parallax))

        # --- 3. Keyframe Decision & Survival Logic ---

        has_parallax = median_parallax > 2.0
        low_features = len(self.lm_pts3d) < (self.cfg.num_features * 0.3)
        force_map = low_features and (median_parallax > 0.5)

        need_kf = has_parallax or force_map

        # --- SURVIVAL OVERRIDE ---
        is_stable = self.probation_counter == 0

        # If we have fewer than 200 points, we are about to die.
        # Force mapping even if we are in probation.
        is_starving = len(self.lm_pts3d) < 200

        # Determine if we are allowed to triangulate/optimize
        should_map = (is_stable or is_starving) and tracking_reliable

        if need_kf and should_map:
            self._add_keyframe()
            self._triangulate_new_points()

            # If we forced mapping due to starvation, skip BA this time
            # (just add points to survive, don't stress the optimizer yet)
            if not is_starving:
                self._optimize_window()
            else:
                print("Survival Mode: Mapping forced, BA skipped.")

            self.T_wc_last_kf = self.T_wc.copy()

            # Reset probation if we successfully mapped in survival mode
            if is_starving:
                self.probation_counter = 0

        elif need_kf:
            # We need a KF (moved enough), but we aren't allowed to map
            # (Probation or Unreliable PnP).
            # We MUST add the keyframe anyway to update the tracking reference!
            self._add_keyframe()
            print(
                f"Skipping Map Update (Probation: {self.probation_counter}, Reliable: {tracking_reliable})"
            )

    def _add_keyframe(self):
        """Stores current state as keyframe"""
        kf = {
            "T_cw": self.get_T_cw(),
            "ids": self.lm_ids.copy(),
            "pts_2d": self.lm_pts2d.copy(),
        }
        self.keyframes.append(kf)
        if len(self.keyframes) > 7:  # Keep window small
            self.keyframes.pop(0)

    def _triangulate_new_points(self):
        """
        Uses candidates + current frame to find new 3D points.
        Crucially, we rely on the backend filters (chirality, epipolar)
        """
        T_cw_curr = self.get_T_cw()

        new_pts3d = []
        new_pts2d = []
        keep_mask = np.ones(len(self.cand_curr_2d), dtype=bool)

        for i in range(len(self.cand_curr_2d)):
            T_wc_first = self.cand_first_pose_wc[i]

            # Check baseline length
            baseline = np.linalg.norm(self.T_wc[:3, 3] - T_wc_first[:3, 3])

            # Don't even try if baseline is tiny (reduces compute)
            if baseline < 0.15:
                continue

            # Invert T_wc_first -> T_cw_first
            R_f = T_wc_first[:3, :3]
            t_f = T_wc_first[:3, 3]
            T_cw_first = np.eye(4)
            T_cw_first[:3, :3] = R_f.T
            T_cw_first[:3, 3] = -R_f.T @ t_f

            # Triangulate
            pt3d, mask, stats = triangulate_and_filter(
                T_cw_first,
                T_cw_curr,
                self.cand_first_2d[i : i + 1],
                self.cand_curr_2d[i : i + 1],
                self.K,
                self.cfg,
            )
            rr.log("diagnostics/backend/avg_angle", rr.Scalars(stats["avg_angle"]))

            if mask[0]:
                new_pts3d.append(pt3d[0])
                new_pts2d.append(self.cand_curr_2d[i])
                keep_mask[i] = False  # Remove from candidates (promoted)

        # Update Main Lists
        if len(new_pts3d) > 0:
            num_new = len(new_pts3d)
            new_ids = np.arange(self.next_pt_id, self.next_pt_id + num_new)
            self.next_pt_id += num_new

            self.lm_pts3d = np.vstack((self.lm_pts3d, np.array(new_pts3d)))
            self.lm_pts2d = np.vstack((self.lm_pts2d, np.array(new_pts2d)))
            self.lm_ids = np.concatenate((self.lm_ids, new_ids))
            # NEW: Init prev points for these new ones to avoid shape mismatch next frame
            # FIX: Robust prev update
            new_pts2d_arr = np.array(new_pts2d)
            if len(self.prev_lm_pts2d) == len(self.lm_pts2d) - num_new:
                self.prev_lm_pts2d = np.vstack((self.prev_lm_pts2d, new_pts2d_arr))
            else:
                # If sizes don't match logic, rebuild prev from current (minus new)
                # This is tricky, easier to just reset prev to current for next frame
                self.prev_lm_pts2d = self.lm_pts2d.copy()

            print(f"Triangulated {num_new} new points.")

        # Clean candidates
        self.cand_curr_2d = self.cand_curr_2d[keep_mask]
        self.cand_first_2d = self.cand_first_2d[keep_mask]
        self.cand_first_pose_wc = [
            self.cand_first_pose_wc[k] for k in range(len(keep_mask)) if keep_mask[k]
        ]

    def _optimize_window(self):
        """Wrapper for Bundle Adjustment with ADAPTIVE FEEDBACK LOOP."""
        if not self.cfg.enable_window_ba or len(self.keyframes) < 3:
            return

        # Prepare Map Dictionary
        for i, pt_id in enumerate(self.lm_ids):
            self.map_points[pt_id] = self.lm_pts3d[i]

        total_points_before = len(self.map_points)

        try:
            # UNPACK 3 VALUES
            self.keyframes, self.map_points, num_pruned = bundle_adjustment_window(
                self.keyframes, self.map_points, self.K, fixed_window_size=True
            )

            # feedback loop
            if total_points_before > 0:
                prune_ratio = num_pruned / total_points_before
            else:
                prune_ratio = 0.0

            if prune_ratio > 0.20 and num_pruned > 50:
                print(
                    f"WARNING: BA instability detected. Pruned {num_pruned} pts ({prune_ratio * 100:.1f}%)."
                )
                self.probation_counter = max(self.probation_counter, 3)

            # Update Current State
            latest_kf = self.keyframes[-1]
            T_cw_opt = latest_kf["T_cw"]
            R_opt = T_cw_opt[:3, :3]
            t_opt = T_cw_opt[:3, 3]

            self.T_wc[:3, :3] = R_opt.T
            self.T_wc[:3, 3] = -R_opt.T @ t_opt

            # Update 3D points in array from Map
            # (Only update those that survived)
            for i, pt_id in enumerate(self.lm_ids):
                if pt_id in self.map_points:
                    self.lm_pts3d[i] = self.map_points[pt_id]

            # sync arrays
            keep_mask = []
            for pt_id in self.lm_ids:
                keep_mask.append(pt_id in self.map_points)
            keep_mask = np.array(keep_mask, dtype=bool)

            # SAFETY CHECK: Ensure arrays are aligned before filtering
            if len(self.lm_pts3d) != len(keep_mask):
                print(
                    f"CRITICAL DESYNC: pts3d={len(self.lm_pts3d)}, ids={len(self.lm_ids)}"
                )
                # Emergency Force-Sync: Truncate to smaller size to prevent crash
                min_len = min(len(self.lm_pts3d), len(keep_mask))
                self.lm_pts3d = self.lm_pts3d[:min_len]
                keep_mask = keep_mask[:min_len]

            if np.sum(keep_mask) < len(self.lm_ids):
                # print(f"Syncing: Dropping {len(self.lm_ids) - np.sum(keep_mask)} points")
                self.lm_pts3d = self.lm_pts3d[keep_mask]
                self.lm_pts2d = self.lm_pts2d[keep_mask]

                # FIX: Sync prev_pts if it exists and matches size
                if len(self.prev_lm_pts2d) == len(keep_mask):
                    self.prev_lm_pts2d = self.prev_lm_pts2d[keep_mask]
                elif len(self.prev_lm_pts2d) > 0:
                    # If sizes mismatch, just reset it to current (safe fallback)
                    self.prev_lm_pts2d = self.lm_pts2d.copy()

                self.lm_ids = self.lm_ids[keep_mask]

        except Exception as e:
            print(f"BA Failed: {e}")

    def _replenish_features(self, img: np.ndarray):
        """Finds new candidates in image"""
        if len(self.cand_curr_2d) + len(self.lm_pts2d) < self.cfg.num_features:
            # Mask out existing points
            mask = np.full(img.shape, 255, dtype=np.uint8)

            for pt in self.lm_pts2d:
                cv2.circle(mask, tuple(pt.astype(int)), self.cfg.min_distance, 0, -1)
            for pt in self.cand_curr_2d:
                cv2.circle(mask, tuple(pt.astype(int)), self.cfg.min_distance, 0, -1)

            new_pts = detect_new_features(img, mask, self.cfg)

            if len(new_pts) > 0:
                self.cand_curr_2d = (
                    np.vstack((self.cand_curr_2d, new_pts))
                    if len(self.cand_curr_2d)
                    else new_pts
                )
                self.cand_first_2d = (
                    np.vstack((self.cand_first_2d, new_pts))
                    if len(self.cand_first_2d)
                    else new_pts
                )
                # Store Pose
                curr_pose = self.T_wc.copy()
                self.cand_first_pose_wc.extend([curr_pose] * len(new_pts))

    def _log_state(self, img: np.ndarray):
        """Visualization Logic"""
        rr.set_time_sequence("frame", self.frame_id)

        # Camera
        rr.log(
            "world/camera",
            rr.Transform3D(translation=self.T_wc[:3, 3], mat3x3=self.T_wc[:3, :3]),
        )

        # Image
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=self.K, width=img.shape[1], height=img.shape[0]
            ),
        )
        rr.log("world/camera/image", rr.Image(img))

        # Trajectory
        self.trajectory_points.append(self.T_wc[:3, 3])
        if len(self.trajectory_points) > 1:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([self.trajectory_points], colors=[[0, 255, 255]]),
            )

        # Points
        if self.cfg.viz_3d_landmarks and len(self.lm_pts3d) > 0:
            rr.log("world/landmarks", rr.Points3D(self.lm_pts3d, colors=[0, 255, 0]))

            # Reprojection 2D
            T_cw = self.get_T_cw()
            reproj, _ = cv2.projectPoints(
                self.lm_pts3d, T_cw[:3, :3], T_cw[:3, 3], self.K, None
            )
            rr.log(
                "world/camera/image/landmarks",
                rr.Points2D(reproj.reshape(-1, 2), colors=[0, 255, 0], radii=2),
            )

        if self.cfg.viz_2d_candidates and len(self.cand_curr_2d) > 0:
            rr.log(
                "world/camera/image/candidates",
                rr.Points2D(self.cand_curr_2d, colors=[255, 0, 0], radii=2),
            )
