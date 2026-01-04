import cv2
import numpy as np
import rerun as rr

from config.config import VOConfig
from modules.frontend import FeatureFrontend, triangulate_points


class VisualOdometry:
    def __init__(self, K: np.ndarray, config: VOConfig):
        self.K = K
        self.cfg = config
        self.frontend = FeatureFrontend(config)

        # state
        self.frame_id = 0
        self.map_points = {}  # {pt_id: [x,y,z]}
        self.next_pt_id = 0
        self.T_wc = np.eye(4)  # world -> camera pose

        # speed scaling logic
        self.last_pos = np.zeros(3)
        self.baseline_speed = 1.0  # moving baseline of speed
        self.is_turning = False

        # keyframe management
        self.keyframe = None
        self.trajectory = []
        self.initialized = False

        # visualization
        rr.init("LightGlue VO", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    def _prune_map(self) -> None:
        """Remove old points to keep dictionary size constant."""
        # max map points to store
        max_points = 20000

        # smaller ids are older points
        threshold_id = self.next_pt_id - max_points

        # collect keys first
        to_remove = [pid for pid in self.map_points if pid < threshold_id]

        for pid in to_remove:
            del self.map_points[pid]

    def process_frame(self, img: np.ndarray) -> None:
        """Process incoming image frame."""
        curr_feats = self.frontend.process_image(img)
        num_kps = curr_feats["keypoints"].shape[1]
        curr_ids = np.full(num_kps, -1, dtype=int)

        # first frame
        if self.keyframe is None:
            self.keyframe = {"feats": curr_feats, "ids": curr_ids, "T_wc": np.eye(4)}
            self._visualize(curr_feats, curr_ids, None, None, img, 0.0)
            self.frame_id += 1
            print(f"Frame {self.frame_id}: Starting...")
            return

        # match
        prev_kf_feats = self.keyframe["feats"]
        matches = self.frontend.match_frames(prev_kf_feats, curr_feats)
        ref_indices, curr_indices = matches[:, 0], matches[:, 1]

        current_speed_for_plot = 0.0

        # initializationi
        if not self.initialized:
            uv_ref = prev_kf_feats["keypoints"][0].cpu().numpy()[ref_indices]
            uv_curr = curr_feats["keypoints"][0].cpu().numpy()[curr_indices]

            median_flow = 0.0
            if len(uv_ref) > 0:
                median_flow = np.median(np.linalg.norm(uv_ref - uv_curr, axis=1))

            if median_flow < self.cfg.min_median_flow:
                print(
                    f"Frame {self.frame_id}: Waiting for motion (median_flow {median_flow:.1f})..."
                )
                self._visualize(curr_feats, curr_ids, matches, prev_kf_feats, img, 0.0)
                self.frame_id += 1
                return

            E, mask = cv2.findEssentialMat(
                uv_ref,
                uv_curr,
                self.K,
                method=cv2.RANSAC,
                prob=self.cfg.init_ransac_prob,
                threshold=self.cfg.init_ransac_thresh,
            )
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, uv_ref, uv_curr, self.K)

                T_cw = np.eye(4)
                T_cw[:3, :3] = R
                T_cw[:3, 3] = t.flatten() * self.cfg.global_scale
                self.T_wc = np.linalg.inv(T_cw)

                # initialize baseline speed
                current_pos = self.T_wc[:3, 3]
                self.last_pos = np.zeros(3)
                init_dist = np.linalg.norm(current_pos - self.last_pos)

                # sanity clamp for initialization
                self.baseline_speed = np.clip(
                    init_dist, self.cfg.init_speed_min, self.cfg.init_speed_max
                )
                current_speed_for_plot = self.baseline_speed

                self.initialized = True
                print(
                    f"Frame {self.frame_id}: *** INIT (Speed {self.baseline_speed:.2f}) ***"
                )

                self._create_keyframe(curr_feats, curr_ids, ref_indices, curr_indices)

        # tracking
        else:
            kf_ids = self.keyframe["ids"][ref_indices]
            valid_mask = np.array(
                [(pid != -1 and pid in self.map_points) for pid in kf_ids]
            )

            if np.sum(valid_mask) > self.cfg.min_inliers:
                pnp_3d = np.array([self.map_points[pid] for pid in kf_ids[valid_mask]])

            if np.sum(valid_mask) > self.cfg.min_inliers:
                pnp_3d = np.array([self.map_points[pid] for pid in kf_ids[valid_mask]])
                pnp_2d = (
                    curr_feats["keypoints"][0].cpu().numpy()[curr_indices[valid_mask]]
                )

                success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                    pnp_3d,
                    pnp_2d,
                    self.K,
                    None,
                    reprojectionError=self.cfg.pnp_reproj_err,
                )

                if success:
                    R, _ = cv2.Rodrigues(R_vec)
                    T_cw = np.eye(4)
                    T_cw[:3, :3] = R
                    T_cw[:3, 3] = t_vec.flatten()

                    T_wc_raw = np.linalg.inv(T_cw)
                    raw_pos = T_wc_raw[:3, 3]

                    # speed control
                    delta = raw_pos - self.last_pos
                    raw_speed = np.linalg.norm(delta)

                    # calculate angular velocity
                    R_prev_wc = self.T_wc[:3, :3]
                    R_rel = R @ R_prev_wc
                    rel_r_vec, _ = cv2.Rodrigues(R_rel)
                    rot_magnitude = np.linalg.norm(rel_r_vec)

                    IS_TURNING = rot_magnitude > self.cfg.turn_thresh
                    IS_MOVING = raw_speed > self.cfg.move_thresh

                    scale_factor = 1.0

                    if IS_MOVING:
                        if IS_TURNING:
                            print(f"TURNING {rot_magnitude}")
                            # state: turning
                            self.is_turning = True
                            target_speed = (
                                self.cfg.turn_smoothing * self.baseline_speed
                            ) + ((1 - self.cfg.turn_smoothing) * raw_speed)
                            scale_factor = target_speed / raw_speed
                        else:
                            # state: straight
                            self.is_turning = False
                            target_speed = (
                                self.cfg.trans_smoothing * self.baseline_speed
                            ) + ((1 - self.cfg.trans_smoothing) * raw_speed)
                            scale_factor = target_speed / raw_speed

                            self.baseline_speed = (
                                (1 - self.cfg.baseline_lr) * self.baseline_speed
                            ) + (self.cfg.baseline_lr * raw_speed)

                        # apply scale
                        scale_factor = np.clip(
                            scale_factor,
                            self.cfg.scale_clamp_min,
                            self.cfg.scale_clamp_max,
                        )

                        corrected_delta = delta * scale_factor
                        self.T_wc[:3, 3] = self.last_pos + corrected_delta
                        self.T_wc[:3, :3] = T_wc_raw[:3, :3]

                        current_speed_for_plot = np.linalg.norm(corrected_delta)
                    else:
                        self.T_wc = T_wc_raw
                        current_speed_for_plot = 0.0

                    self.last_pos = self.T_wc[:3, 3].copy()

                    if inliers is not None:
                        tracked_indices = curr_indices[valid_mask]
                        tracked_ids = kf_ids[valid_mask]
                        for i in inliers.flatten():
                            curr_ids[tracked_indices[i]] = tracked_ids[i]

                    num_tracked = np.sum(curr_ids != -1)

                    # kf decision
                    uv_ref = prev_kf_feats["keypoints"][0].cpu().numpy()[ref_indices]
                    uv_curr = curr_feats["keypoints"][0].cpu().numpy()[curr_indices]
                    median_flow = np.median(np.linalg.norm(uv_ref - uv_curr, axis=1))

                    is_keyframe = False
                    reason = ""
                    if median_flow > self.cfg.min_median_flow:
                        is_keyframe = True
                        reason = "Median Flow"
                    elif num_tracked < self.cfg.kf_min_tracked:
                        is_keyframe = True
                        reason = "Low Tracking"

                    if is_keyframe:
                        print(
                            f"Frame {self.frame_id}: New KF ({reason}) | Tracked: {num_tracked}"
                        )
                        self._create_keyframe(
                            curr_feats, curr_ids, ref_indices, curr_indices
                        )
                    else:
                        print(
                            f"Frame {self.frame_id}: Tracking... (Median Flow {median_flow:.1f})"
                        )

                else:
                    print(f"Frame {self.frame_id}: PnP Failed! Resetting...")
                    self._reset_system()
            else:
                print(f"Frame {self.frame_id}: Lost Track (<10 matches). Resetting...")
                self._reset_system()

        self._visualize(
            curr_feats, curr_ids, matches, prev_kf_feats, img, current_speed_for_plot
        )
        self.frame_id += 1

    def _create_keyframe(
        self,
        curr_feats: dict,
        curr_ids: np.ndarray,
        ref_indices: np.ndarray,
        curr_indices: np.ndarray,
    ) -> None:
        """Triangulate new points and create keyframe."""
        T_cw_ref = np.linalg.inv(self.keyframe["T_wc"])
        T_cw_curr = np.linalg.inv(self.T_wc)
        no_id_mask = curr_ids[curr_indices] == -1

        if np.sum(no_id_mask) > 0:
            uv_ref_tri = (
                self.keyframe["feats"]["keypoints"][0]
                .cpu()
                .numpy()[ref_indices[no_id_mask]]
            )
            uv_curr_tri = (
                curr_feats["keypoints"][0].cpu().numpy()[curr_indices[no_id_mask]]
            )
            new_pts3d, valid_tri = triangulate_points(
                T_cw_ref, T_cw_curr, uv_ref_tri, uv_curr_tri, self.K, self.cfg
            )

            idx_to_update = curr_indices[no_id_mask]
            count = 0
            for i, is_valid in enumerate(valid_tri):
                if is_valid:
                    self.map_points[self.next_pt_id] = new_pts3d[count]
                    curr_ids[idx_to_update[i]] = self.next_pt_id
                    self.next_pt_id += 1
                    count += 1

        self.keyframe = {"feats": curr_feats, "ids": curr_ids, "T_wc": self.T_wc.copy()}

        self._prune_map()

    def _reset_system(self) -> None:
        """Reset VO state on failure."""
        self.initialized = False
        self.map_points = {}
        self.keyframe = None
        self.trajectory = []

        # reset logic
        self.last_pos = np.zeros(3)
        self.baseline_speed = 1.0

    def _visualize(
        self,
        curr_feats: dict,
        curr_ids: np.ndarray,
        matches: np.ndarray,
        prev_feats: dict,
        img: np.ndarray,
        speed: float,
    ) -> None:
        """Send data to Rerun for visualization."""
        rr.set_time("frame", sequence=self.frame_id)
        rr.log("world/camera/image", rr.Image(img))

        # plot speed
        rr.log("metrics/speed", rr.Scalars(speed))

        # plot baseline (target)
        rr.log("metrics/baseline", rr.Scalars(self.baseline_speed))

        # camera
        rr.log(
            "world/camera",
            rr.Transform3D(translation=self.T_wc[:3, 3], mat3x3=self.T_wc[:3, :3]),
        )

        # trajectory
        self.trajectory.append(self.T_wc[:3, 3].copy())

        # color trajectory based on state
        rr.log("world/traj", rr.LineStrips3D([self.trajectory], colors=[[255, 255, 0]]))

        kps = curr_feats["keypoints"][0].cpu().numpy()
        mask_good = curr_ids != -1
        if np.any(mask_good):
            rr.log(
                "world/camera/image/landmarks",
                rr.Points2D(kps[mask_good], colors=[0, 255, 0], radii=2),
            )
        if np.any(~mask_good):
            rr.log(
                "world/camera/image/candidates",
                rr.Points2D(kps[~mask_good], colors=[255, 0, 0], radii=2),
            )

        if len(self.map_points) > 0:
            pts = []
            colors = []
            active_ids = set(curr_ids[curr_ids != -1])
            for pid, pt in self.map_points.items():
                pts.append(pt)
                if pid in active_ids:
                    colors.append([0, 255, 0])
                else:
                    colors.append([200, 200, 200])
            rr.log(
                "world/map",
                rr.Points3D(np.array(pts), colors=np.array(colors), radii=0.05),
            )

        if matches is not None and prev_feats is not None:
            origins = prev_feats["keypoints"][0].cpu().numpy()[matches[:, 0]]
            vectors = kps[matches[:, 1]] - origins
            rr.log(
                "world/camera/image/flow",
                rr.Arrows2D(
                    origins=origins[::10], vectors=vectors[::10], colors=[0, 255, 255]
                ),
            )
