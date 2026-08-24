import heapq
import math
import os
from collections import deque
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from nav.config import ASTAR_DEFAULTS, DEFAULT_REACH_DISTANCE_M


Point = Tuple[int, int]
WorldPoint = Tuple[float, float]
PointToWorld = Callable[[Point], Optional[WorldPoint]]


class AStarBaseline:
    """Minimap A* baseline that outputs egocentric actions.

    Tuning-parameter defaults live in ``nav.config.ASTAR_DEFAULTS``; the
    constructor exposes them as individual kwargs so the runner can override
    any subset per-run.
    """

    _MARKER_RADIUS_CANONICAL_PX = 6.0
    _MARKER_CLEAR_RADIUS_FACTOR = 8.0 / 3.0

    def __init__(
        self,
        grid_cell_m: float = ASTAR_DEFAULTS.grid_cell_m,
        obstacle_threshold: int = ASTAR_DEFAULTS.obstacle_threshold,
        obstacle_bright_threshold: int = ASTAR_DEFAULTS.obstacle_bright_threshold,
        contrast_background_px: int = ASTAR_DEFAULTS.contrast_background_px,
        contrast_threshold: int = ASTAR_DEFAULTS.contrast_threshold,
        contrast_min_area_px: int = ASTAR_DEFAULTS.contrast_min_area_px,
        min_free_ratio: float = ASTAR_DEFAULTS.min_free_ratio,
        obstacle_clearance_m: float = ASTAR_DEFAULTS.obstacle_clearance_m,
        path_smoothing: bool = ASTAR_DEFAULTS.path_smoothing,
        path_corner_smoothing_m: float = ASTAR_DEFAULTS.path_corner_smoothing_m,
        stanley_gain: float = ASTAR_DEFAULTS.stanley_gain,
        stanley_softening_m: float = ASTAR_DEFAULTS.stanley_softening_m,
        stanley_max_correction_deg: float = ASTAR_DEFAULTS.stanley_max_correction_deg,
        stanley_forward_tolerance_deg: float = ASTAR_DEFAULTS.stanley_forward_tolerance_deg,
        turn_tolerance_deg: float = ASTAR_DEFAULTS.turn_tolerance_deg,
        forward_priority_tolerance_deg: float = ASTAR_DEFAULTS.forward_priority_tolerance_deg,
        drive_turn_tolerance_deg: float = ASTAR_DEFAULTS.drive_turn_tolerance_deg,
        waypoint_distance_m: float = ASTAR_DEFAULTS.waypoint_distance_m,
        waypoint_reach_m: float = ASTAR_DEFAULTS.waypoint_reach_m,
        path_replan_distance_m: float = ASTAR_DEFAULTS.path_replan_distance_m,
        dynamic_replan_lookahead_m: float = ASTAR_DEFAULTS.dynamic_replan_lookahead_m,
        dynamic_replan_confirm_steps: int = ASTAR_DEFAULTS.dynamic_replan_confirm_steps,
        terminal_approach_m: float = ASTAR_DEFAULTS.terminal_approach_m,
        lookahead_m: float = ASTAR_DEFAULTS.lookahead_m,
        front_cone_deg: float = ASTAR_DEFAULTS.front_cone_deg,
        hysteresis_reset_deg: float = ASTAR_DEFAULTS.hysteresis_reset_deg,
        hysteresis_lock_deg: float = ASTAR_DEFAULTS.hysteresis_lock_deg,
        stuck_distance_m: float = ASTAR_DEFAULTS.stuck_distance_m,
        stuck_steps: int = ASTAR_DEFAULTS.stuck_steps,
        recovery_turn_steps: int = ASTAR_DEFAULTS.recovery_turn_steps,
        stuck_block_ahead_m: float = ASTAR_DEFAULTS.stuck_block_ahead_m,
        stuck_block_radius_m: float = ASTAR_DEFAULTS.stuck_block_radius_m,
        stuck_block_ttl_steps: int = ASTAR_DEFAULTS.stuck_block_ttl_steps,
        proxy_stop_distance_m: float = ASTAR_DEFAULTS.proxy_stop_distance_m,
        pixel_scale: float = 1.0,
        minimap_has_baked_markers: bool = False,
        debug_viz: bool = False,
        debug_dir: Optional[str] = None,
    ):
        self.pixel_scale = float(pixel_scale)
        if self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive.")

        def scaled_int(value: float, minimum: int = 1) -> int:
            return max(minimum, int(round(float(value) * self.pixel_scale)))

        self.grid_cell_m = self._positive(grid_cell_m, "grid_cell_m")
        self.obstacle_threshold = int(obstacle_threshold)
        self.obstacle_bright_threshold = int(obstacle_bright_threshold)
        if self.obstacle_bright_threshold <= self.obstacle_threshold:
            raise ValueError(
                "obstacle_bright_threshold must exceed obstacle_threshold."
            )
        # medianBlur needs an odd kernel; area scales with the pixel count.
        self.contrast_background_px = max(3, scaled_int(contrast_background_px) | 1)
        self.contrast_threshold = int(contrast_threshold)
        self.contrast_min_area_px = max(
            1, int(round(float(contrast_min_area_px) * self.pixel_scale ** 2))
        )
        self.min_free_ratio = float(min_free_ratio)
        self.obstacle_clearance_m = self._nonnegative(
            obstacle_clearance_m, "obstacle_clearance_m"
        )
        self.path_smoothing = bool(path_smoothing)
        self.path_corner_smoothing_m = self._nonnegative(
            path_corner_smoothing_m, "path_corner_smoothing_m"
        )
        self.stanley_gain = self._nonnegative(stanley_gain, "stanley_gain")
        self.stanley_softening_m = self._positive(
            stanley_softening_m,
            "stanley_softening_m",
        )
        self.stanley_max_correction_deg = self._positive(
            stanley_max_correction_deg,
            "stanley_max_correction_deg",
        )
        self.stanley_forward_tolerance_deg = self._nonnegative(
            stanley_forward_tolerance_deg,
            "stanley_forward_tolerance_deg",
        )
        self.minimap_has_baked_markers = bool(minimap_has_baked_markers)
        self.marker_clear_px = scaled_int(
            self._MARKER_RADIUS_CANONICAL_PX * self._MARKER_CLEAR_RADIUS_FACTOR
        )
        self.turn_tolerance_deg = float(turn_tolerance_deg)
        self.forward_priority_tolerance_deg = float(forward_priority_tolerance_deg)
        self.drive_turn_tolerance_deg = float(drive_turn_tolerance_deg)
        self.waypoint_distance_m = float(waypoint_distance_m)
        self.waypoint_reach_m = float(waypoint_reach_m)
        self.path_replan_distance_m = float(path_replan_distance_m)
        self.dynamic_replan_lookahead_m = self._positive(
            dynamic_replan_lookahead_m,
            "dynamic_replan_lookahead_m",
        )
        self.dynamic_replan_confirm_steps = int(dynamic_replan_confirm_steps)
        if self.dynamic_replan_confirm_steps <= 0:
            raise ValueError("dynamic_replan_confirm_steps must be positive.")
        self.terminal_approach_m = float(terminal_approach_m)
        self.lookahead_m = float(lookahead_m)
        self.front_cone_deg = float(front_cone_deg)
        self.hysteresis_reset_deg = float(hysteresis_reset_deg)
        self.hysteresis_lock_deg = float(hysteresis_lock_deg)
        self.stuck_distance_m = self._nonnegative(
            stuck_distance_m, "stuck_distance_m"
        )
        self.stuck_steps = int(stuck_steps)
        self.recovery_turn_steps = int(recovery_turn_steps)
        self.stuck_block_ahead_m = self._nonnegative(
            stuck_block_ahead_m, "stuck_block_ahead_m"
        )
        self.stuck_block_radius_m = self._positive(
            stuck_block_radius_m, "stuck_block_radius_m"
        )
        self.stuck_block_ttl_steps = int(stuck_block_ttl_steps)
        self.proxy_stop_distance_m = self._positive(
            proxy_stop_distance_m, "proxy_stop_distance_m"
        )
        self.debug_viz = bool(debug_viz)
        self.debug_dir = debug_dir
        self.grid_step_x_px = 1
        self.grid_step_y_px = 1
        self.obstacle_clearance_x_px = 0
        self.obstacle_clearance_y_px = 0
        self.stuck_block_radius_x_px = 1
        self.stuck_block_radius_y_px = 1
        self.world_per_pixel = np.eye(2, dtype=np.float64)
        self.last_path: List[Point] = []
        self.tracking_path: List[Point] = []
        self.path_segment_index = 0
        self.follow_waypoints: List[Point] = []
        self.follow_waypoint_index = 0
        self.last_turn_sign = 0
        self.last_action_was_move = False
        self.prev_world_xz: Optional[Tuple[float, float]] = None
        self.stuck_count = 0
        self.recovery_count = 0
        self.recovery_turn_sign = 1
        self.last_effective_target_xy: Optional[Point] = None
        self.last_debug = {}
        self.last_tracking_metrics = {}
        self.last_plan_status = "not_planned"
        self.plan_revision = 0
        self.dynamic_blocked_count = 0
        self.virtual_obstacles: List[Tuple[int, int, int, int, int]] = []

    @staticmethod
    def _positive(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        return value

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        return value

    def _update_pixel_geometry(
        self,
        reference_xy: Point,
        point_to_world: PointToWorld,
    ) -> None:
        """Resolve physical planner dimensions into runtime minimap pixels."""
        x, y = int(reference_xy[0]), int(reference_xy[1])
        origin = point_to_world((x, y))
        x_neighbor = point_to_world((x + 1, y))
        y_neighbor = point_to_world((x, y + 1))
        if origin is None or x_neighbor is None or y_neighbor is None:
            raise ValueError("A* could not resolve minimap meters-per-pixel.")

        world_per_pixel = np.array(
            [
                [x_neighbor[0] - origin[0], y_neighbor[0] - origin[0]],
                [x_neighbor[1] - origin[1], y_neighbor[1] - origin[1]],
            ],
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(world_per_pixel))) <= 1e-12:
            raise ValueError("A* minimap-to-world projection is degenerate.")

        meters_per_pixel_x = float(np.linalg.norm(world_per_pixel[:, 0]))
        meters_per_pixel_y = float(np.linalg.norm(world_per_pixel[:, 1]))
        if meters_per_pixel_x <= 1e-9 or meters_per_pixel_y <= 1e-9:
            raise ValueError("A* minimap pixel scale must be positive.")

        self.world_per_pixel = world_per_pixel
        self.grid_step_x_px = max(
            1, int(round(self.grid_cell_m / meters_per_pixel_x))
        )
        self.grid_step_y_px = max(
            1, int(round(self.grid_cell_m / meters_per_pixel_y))
        )
        self.obstacle_clearance_x_px = max(
            0, int(math.ceil(self.obstacle_clearance_m / meters_per_pixel_x))
        )
        self.obstacle_clearance_y_px = max(
            0, int(math.ceil(self.obstacle_clearance_m / meters_per_pixel_y))
        )
        self.stuck_block_radius_x_px = max(
            1, int(math.ceil(self.stuck_block_radius_m / meters_per_pixel_x))
        )
        self.stuck_block_radius_y_px = max(
            1, int(math.ceil(self.stuck_block_radius_m / meters_per_pixel_y))
        )

    def decide(
        self,
        minimap_rgb: Optional[np.ndarray],
        curr_xy: Point,
        target_xy: Point,
        agent_theta: float,
        reach_m: float = DEFAULT_REACH_DISTANCE_M,
        step: Optional[int] = None,
        curr_world_xz: Optional[WorldPoint] = None,
        target_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[str, str, List[Point]]:
        self.last_tracking_metrics = {}
        distance_m = self._world_distance(curr_world_xz, target_world_xz)
        if distance_m is None:
            raise ValueError(
                "A* requires current and target Unity world coordinates."
            )
        if reach_m <= 0:
            raise ValueError("reach_m must be positive.")
        if minimap_rgb is not None and point_to_world is None:
            raise ValueError(
                "A* requires a minimap-to-world projection for path tracking."
            )
        if point_to_world is not None:
            self._update_pixel_geometry(curr_xy, point_to_world)
        if distance_m <= reach_m:
            self.last_path = [curr_xy, target_xy]
            self.tracking_path = list(self.last_path)
            self.path_segment_index = 0
            self.last_action_was_move = False
            return (
                "stop",
                f"Astar reached target vicinity; dist={distance_m:.2f}m.",
                self.last_path,
            )

        self._decay_virtual_obstacles()
        stuck_reason = self._update_stuck_state(
            curr_xy,
            curr_world_xz,
            agent_theta,
            point_to_world=point_to_world,
        )
        if self.recovery_count > 0:
            action = self._recovery_action()
            self._record_action(action)
            return (
                action,
                f"Astar recovery({stuck_reason}) turns_left={self.recovery_count} "
                f"turn_sign={self.recovery_turn_sign}",
                self.last_path,
            )

        if minimap_rgb is None:
            action, reasoning = self._greedy_action(
                curr_xy,
                target_xy,
                agent_theta,
                curr_world_xz=curr_world_xz,
                target_world_xz=target_world_xz,
            )
            self.last_path = [curr_xy, target_xy]
            self.tracking_path = list(self.last_path)
            self.path_segment_index = 0
            self._record_action(action)
            return action, f"Astar fallback without minimap: {reasoning}", self.last_path

        walkable = self._build_walkable_grid(minimap_rgb, curr_xy, target_xy)
        should_replan, replan_reason = self._should_replan(
            curr_xy,
            walkable=walkable,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        path = self.last_path
        if should_replan:
            self.plan_revision += 1
            path = self._plan_path(walkable, curr_xy, target_xy)
            self.follow_waypoints = self._build_follow_waypoints(
                path,
                curr_xy,
                target_xy,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            self.follow_waypoint_index = 0
            self.last_turn_sign = 0

        if not path:
            action, reasoning = self._greedy_action(
                curr_xy,
                target_xy,
                agent_theta,
                curr_world_xz=curr_world_xz,
                target_world_xz=target_world_xz,
            )
            fallback_path = [curr_xy, target_xy]
            self.last_path = []
            self.tracking_path = []
            self.path_segment_index = 0
            self.follow_waypoints = []
            self.follow_waypoint_index = 0
            self.last_effective_target_xy = None
            self._save_debug_visualization(
                minimap_rgb, curr_xy, target_xy, fallback_path, None, step
            )
            self._record_action(action)
            return action, f"Astar found no path, fallback: {reasoning}", fallback_path

        self.last_path = path
        if self._using_unreachable_proxy() and self.last_effective_target_xy is not None:
            proxy_distance, proxy_unit = self._point_distance(
                curr_xy,
                self.last_effective_target_xy,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            if (
                proxy_distance <= reach_m
                and distance_m <= self.proxy_stop_distance_m
            ):
                action, relative = self._terminal_action_toward(
                    curr_xy,
                    target_xy,
                    agent_theta,
                    curr_world_xz=curr_world_xz,
                    target_world_xz=target_world_xz,
                )
                self._save_debug_visualization(
                    minimap_rgb, curr_xy, target_xy, path, target_xy, step
                )
                self._record_action(action)
                return (
                    action,
                    f"Astar terminal approach from reachable proxy "
                    f"{self.last_effective_target_xy}; "
                    f"proxy_dist={proxy_distance:.2f}{proxy_unit} "
                    f"target_dist={distance_m:.2f}m relative={relative:.1f}deg "
                    f"plan_status={self.last_plan_status}",
                    path,
                )

        if not self.follow_waypoints:
            self.follow_waypoints = self._build_follow_waypoints(
                path,
                curr_xy,
                target_xy,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            self.follow_waypoint_index = 0
        anchor_waypoint, anchor_idx = self._current_follow_waypoint(
            curr_xy,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        waypoint, waypoint_idx, waypoint_reason = self._lookahead_follow_waypoint(
            curr_xy,
            agent_theta,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        closest_idx, closest_dist, closest_unit = self._closest_path_index(
            path,
            curr_xy,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        terminal_approach = (
            self.last_plan_status == "target_reachable"
            and not self._using_unreachable_proxy()
            and distance_m <= self.terminal_approach_m
            and waypoint == target_xy
        )
        action_waypoint = waypoint
        heading_error = 0.0
        cross_track_m = 0.0
        segment_idx = self.path_segment_index
        if terminal_approach:
            waypoint_idx = len(self.follow_waypoints) - 1
            waypoint_reason = "terminal_align_then_forward"
            action, relative = self._terminal_action_toward(
                curr_xy,
                waypoint,
                agent_theta,
                curr_world_xz=curr_world_xz,
                target_world_xz=target_world_xz,
            )
            heading_error = relative
            controller = "terminal"
        else:
            (
                action,
                relative,
                heading_error,
                cross_track_m,
                segment_idx,
                action_waypoint,
            ) = self._stanley_action(
                path,
                curr_xy,
                agent_theta,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            controller = "stanley"
            waypoint_reason = (
                f"{waypoint_reason};stanley_segment_{segment_idx}_"
                f"of_{max(0, len(self.tracking_path) - 2)}"
            )
        self._save_debug_visualization(
            minimap_rgb, curr_xy, target_xy, path, action_waypoint, step
        )
        self._record_action(action)
        return (
            action,
            "Astar "
            f"{'replanned' if should_replan else 'tracking'}({replan_reason}) "
            f"stuck={self.stuck_count} "
            f"controller={controller} segment={segment_idx} "
            f"cte={cross_track_m:.2f}m heading_error={heading_error:.1f}deg "
            f"steering={relative:.1f}deg "
            f"plan_status={self.last_plan_status} "
            f"path_len={len(path)} closest_idx={closest_idx} "
            f"closest_dist={closest_dist:.2f}{closest_unit} "
            f"anchor_idx={anchor_idx}/{len(self.follow_waypoints) - 1} "
            f"anchor={anchor_waypoint} action_idx={waypoint_idx} "
            f"waypoint={waypoint} waypoint_reason={waypoint_reason} "
            f"relative={relative:.1f}deg last_turn={self.last_turn_sign}",
            path,
        )

    def _low_contrast_obstacles(self, gray: np.ndarray) -> np.ndarray:
        """Find structures whose gray sits inside the free band.

        Painted rails and partition tops render at almost exactly the floor gray
        (measured: rail top 125 vs floor 127), so the global band cannot separate
        them. What does separate them is *local* contrast: the rail carries a
        bright highlight edge and a dark shadow edge a few pixels apart. Pixels
        deviating from a median-filtered background are collected, then small
        connected components are dropped so floor texture speckle does not
        become an obstacle.
        """
        if self.contrast_threshold <= 0:
            return np.zeros(gray.shape, dtype=bool)

        background = cv2.medianBlur(gray, self.contrast_background_px)
        deviating = cv2.absdiff(gray, background) > self.contrast_threshold
        mask = deviating.astype(np.uint8)
        if self.contrast_min_area_px <= 1:
            return mask > 0

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.contrast_min_area_px
        return keep[labels]

    def _clear_discs(self, free: np.ndarray, points: Tuple[Point, ...]) -> np.ndarray:
        free_u8 = free.astype(np.uint8)
        for point in points:
            cv2.circle(
                free_u8,
                (int(point[0]), int(point[1])),
                self.marker_clear_px,
                1,
                -1,
            )
        return free_u8.astype(bool)

    def _forced_free_points(self, curr_xy: Point, target_xy: Point) -> Tuple[Point, ...]:
        """Points whose surroundings are forced free before grid pooling.

        The target disc is the goal-tolerance region: ~20% of benchmark targets
        are annotated at or just inside rack geometry, and ``_plan_path`` always
        terminates the route at the raw target pixel, so without this the final
        segment dives into a shelf.

        The agent disc is only a marker artifact mask. It spans roughly
        1.2 x 1.6 world meters, so applying it when the minimap carries no baked
        marker just deletes the obstacle the agent is currently touching and
        lets the planner route straight back into it.
        """
        if self.minimap_has_baked_markers:
            return (curr_xy, target_xy)
        return (target_xy,)

    def _build_walkable_grid(
        self, minimap_rgb: np.ndarray, curr_xy: Point, target_xy: Point
    ) -> np.ndarray:
        rgb = self._to_uint8_rgb(minimap_rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Obstacles are whatever deviates from the mid-gray floor in either
        # direction: dark racks/crates *and* bright partition walls, which are
        # brighter than the floor and would otherwise read as free space.
        threshold_free = (gray > self.obstacle_threshold) & (
            gray <= self.obstacle_bright_threshold
        )
        low_contrast = self._low_contrast_obstacles(gray)
        threshold_free = threshold_free & ~low_contrast
        free = threshold_free.copy()

        forced_free = self._forced_free_points(curr_xy, target_xy)
        free = self._clear_discs(free, forced_free)

        if self.obstacle_clearance_x_px > 0 or self.obstacle_clearance_y_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.obstacle_clearance_x_px * 2 + 1,
                    self.obstacle_clearance_y_px * 2 + 1,
                ),
            )
            obstacles = cv2.dilate((~free).astype(np.uint8), kernel, iterations=1)
            free = obstacles == 0
            free = self._clear_discs(free, forced_free)

        if self.virtual_obstacles:
            free_u8 = free.astype(np.uint8)
            for ox, oy, radius_x, radius_y, _ttl in self.virtual_obstacles:
                cv2.ellipse(
                    free_u8,
                    (int(ox), int(oy)),
                    (int(radius_x), int(radius_y)),
                    0,
                    0,
                    360,
                    0,
                    -1,
                )
            free = free_u8.astype(bool)

        h, w = free.shape
        gh = int(math.ceil(h / self.grid_step_y_px))
        gw = int(math.ceil(w / self.grid_step_x_px))
        walkable = np.zeros((gh, gw), dtype=bool)
        for gy in range(gh):
            y0 = gy * self.grid_step_y_px
            y1 = min(h, y0 + self.grid_step_y_px)
            for gx in range(gw):
                x0 = gx * self.grid_step_x_px
                x1 = min(w, x0 + self.grid_step_x_px)
                walkable[gy, gx] = free[y0:y1, x0:x1].mean() >= self.min_free_ratio

        self.last_debug = {
            "gray": gray,
            "threshold_free": threshold_free,
            "low_contrast_obstacles": low_contrast,
            "inflated_free": free,
            "walkable": walkable,
            "grid_step_px": (self.grid_step_x_px, self.grid_step_y_px),
            "obstacle_clearance_px": (
                self.obstacle_clearance_x_px,
                self.obstacle_clearance_y_px,
            ),
        }
        return walkable

    def _save_debug_visualization(
        self,
        minimap_rgb: np.ndarray,
        curr_xy: Point,
        target_xy: Point,
        path: List[Point],
        waypoint: Optional[Point],
        step: Optional[int],
    ) -> None:
        if not self.debug_viz or not self.debug_dir or step is None:
            return

        os.makedirs(self.debug_dir, exist_ok=True)
        prefix = os.path.join(self.debug_dir, f"{int(step):04d}")

        gray = self.last_debug.get("gray")
        if gray is not None:
            cv2.imwrite(f"{prefix}_gray.png", gray)

        threshold_free = self.last_debug.get("threshold_free")
        if threshold_free is not None:
            threshold_img = self._mask_to_bgr(threshold_free)
            self._draw_points(threshold_img, curr_xy, target_xy)
            cv2.imwrite(f"{prefix}_free_threshold.png", threshold_img)

        inflated_free = self.last_debug.get("inflated_free")
        if inflated_free is not None:
            inflated_img = self._mask_to_bgr(inflated_free)
            self._draw_points(inflated_img, curr_xy, target_xy)
            cv2.imwrite(f"{prefix}_free_inflated.png", inflated_img)

        walkable = self.last_debug.get("walkable")
        if walkable is not None:
            grid = cv2.resize(
                self._mask_to_bgr(walkable),
                (inflated_free.shape[1], inflated_free.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            self._draw_points(grid, curr_xy, target_xy)
            cv2.imwrite(f"{prefix}_walkable_grid.png", grid)

        overlay = cv2.cvtColor(self._to_uint8_rgb(minimap_rgb), cv2.COLOR_RGB2BGR)
        if len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                overlay,
                [pts],
                isClosed=False,
                color=(0, 255, 255),
                thickness=max(1, round(2 * self.pixel_scale)),
            )
        if waypoint is not None:
            cv2.circle(
                overlay,
                (int(waypoint[0]), int(waypoint[1])),
                max(2, round(5 * self.pixel_scale)),
                (255, 255, 0),
                -1,
            )
        self._draw_points(overlay, curr_xy, target_xy)
        cv2.imwrite(f"{prefix}_path.png", overlay)

    @staticmethod
    def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
        img = np.where(mask, 255, 0).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _draw_points(self, bgr: np.ndarray, curr_xy: Point, target_xy: Point) -> None:
        curr_radius = max(2, round(5 * self.pixel_scale))
        target_radius = max(2, round(6 * self.pixel_scale))
        cv2.circle(bgr, (int(curr_xy[0]), int(curr_xy[1])), curr_radius, (0, 0, 255), -1)
        cv2.circle(bgr, (int(target_xy[0]), int(target_xy[1])), target_radius, (0, 200, 0), -1)

    def _plan_path(
        self, walkable: np.ndarray, curr_xy: Point, target_xy: Point
    ) -> List[Point]:
        start = self._nearest_free_cell(walkable, self._point_to_cell(curr_xy))
        goal = self._nearest_free_cell(walkable, self._point_to_cell(target_xy))
        self.last_effective_target_xy = None
        if start is None or goal is None:
            return []

        came_from = {}
        cost_so_far = {start: 0.0}
        open_heap = [(self._heuristic(start, goal), 0.0, start)]
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current_cost > cost_so_far.get(current, float("inf")):
                continue
            if current == goal:
                self.last_plan_status = "target_reachable"
                self.last_effective_target_xy = target_xy
                raw_path = self._reconstruct_path(came_from, current, target_xy)
                return self._smooth_path(walkable, raw_path)

            cy, cx = current
            for dy, dx, move_cost in neighbors:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < walkable.shape[0] and 0 <= nx < walkable.shape[1]):
                    continue
                if not walkable[ny, nx]:
                    continue
                if dy != 0 and dx != 0 and (
                    not walkable[cy, nx]
                    or not walkable[ny, cx]
                ):
                    continue
                nxt = (ny, nx)
                new_cost = current_cost + move_cost
                if new_cost < cost_so_far.get(nxt, float("inf")):
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(nxt, goal)
                    heapq.heappush(open_heap, (priority, new_cost, nxt))
                    came_from[nxt] = current

        # If the target is isolated or inside an obstacle, do not drop straight
        # into greedy fallback. Plan to the reachable cell closest to the real
        # target so the agent approaches the accessible side of the blockage.
        if not cost_so_far:
            self.last_plan_status = "no_reachable_cells"
            return []

        best_reachable = min(cost_so_far, key=lambda cell: self._heuristic(cell, goal))
        if best_reachable == start:
            self.last_plan_status = "target_unreachable_start_only"
            return []

        proxy_xy = self._cell_to_point(best_reachable)
        self.last_effective_target_xy = proxy_xy
        self.last_plan_status = (
            f"target_unreachable_proxy={proxy_xy}_"
            f"grid_dist={self._heuristic(best_reachable, goal):.1f}"
        )
        raw_path = self._reconstruct_path(came_from, best_reachable, proxy_xy)
        return self._smooth_path(walkable, raw_path)

    def _using_unreachable_proxy(self) -> bool:
        return self.last_plan_status.startswith("target_unreachable_proxy=")

    def _nearest_free_cell(
        self, walkable: np.ndarray, cell: Tuple[int, int], max_radius: int = 10
    ) -> Optional[Tuple[int, int]]:
        cy = min(max(cell[0], 0), walkable.shape[0] - 1)
        cx = min(max(cell[1], 0), walkable.shape[1] - 1)
        if walkable[cy, cx]:
            return cy, cx

        seen = {(cy, cx)}
        q = deque([(cy, cx, 0)])
        while q:
            y, x, radius = q.popleft()
            if radius >= max_radius:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < walkable.shape[0] and 0 <= nx < walkable.shape[1]):
                    continue
                if (ny, nx) in seen:
                    continue
                if walkable[ny, nx]:
                    return ny, nx
                seen.add((ny, nx))
                q.append((ny, nx, radius + 1))
        return None

    def _reconstruct_path(
        self, came_from, current: Tuple[int, int], target_xy: Point
    ) -> List[Point]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        path = [self._cell_to_point(cell) for cell in cells]
        if path:
            path[-1] = (int(target_xy[0]), int(target_xy[1]))
        return path

    def _smooth_path(
        self,
        walkable: np.ndarray,
        raw_path: List[Point],
    ) -> List[Point]:
        if len(raw_path) < 3 or not self.path_smoothing:
            control_points = list(raw_path)
        else:
            control_points = [raw_path[0]]
            anchor_idx = 0
            while anchor_idx < len(raw_path) - 1:
                next_idx = anchor_idx + 1
                for candidate_idx in range(
                    len(raw_path) - 1,
                    anchor_idx,
                    -1,
                ):
                    if self._line_is_walkable(
                        walkable,
                        raw_path[anchor_idx],
                        raw_path[candidate_idx],
                    ):
                        next_idx = candidate_idx
                        break
                control_points.append(raw_path[next_idx])
                anchor_idx = next_idx

            string_pull_points = list(control_points)
            control_points = self._round_path_corners(
                walkable,
                string_pull_points,
            )
            self.last_debug["string_pull_control_points"] = string_pull_points

        smoothed_path = self._densify_path(control_points)
        self.tracking_path = list(control_points)
        self.path_segment_index = 0
        self.last_debug["raw_path_len"] = len(raw_path)
        self.last_debug["smoothed_control_points"] = list(control_points)
        self.last_debug["smoothed_path_len"] = len(smoothed_path)
        return smoothed_path

    def _round_path_corners(
        self,
        walkable: np.ndarray,
        control_points: List[Point],
    ) -> List[Point]:
        if len(control_points) < 3 or self.path_corner_smoothing_m <= 0:
            return list(control_points)

        rounded = [control_points[0]]
        for previous, corner, following in zip(
            control_points,
            control_points[1:-1],
            control_points[2:],
        ):
            incoming_m = self._pixel_segment_length_m(previous, corner)
            outgoing_m = self._pixel_segment_length_m(corner, following)
            trim_in_m = min(
                self.path_corner_smoothing_m,
                0.25 * incoming_m,
            )
            trim_out_m = min(
                self.path_corner_smoothing_m,
                0.25 * outgoing_m,
            )
            incoming_tangent = self._point_along_segment(
                corner,
                previous,
                trim_in_m,
                incoming_m,
            )
            outgoing_tangent = self._point_along_segment(
                corner,
                following,
                trim_out_m,
                outgoing_m,
            )
            if (
                incoming_tangent != outgoing_tangent
                and self._line_is_walkable(
                    walkable,
                    incoming_tangent,
                    outgoing_tangent,
                )
            ):
                for point in (incoming_tangent, outgoing_tangent):
                    if point != rounded[-1]:
                        rounded.append(point)
            elif corner != rounded[-1]:
                rounded.append(corner)

        if control_points[-1] != rounded[-1]:
            rounded.append(control_points[-1])
        if all(
            self._line_is_walkable(walkable, start, end)
            for start, end in zip(rounded, rounded[1:])
        ):
            return rounded
        return list(control_points)

    def _pixel_segment_length_m(self, start: Point, end: Point) -> float:
        pixel_delta = np.array(
            [float(end[0] - start[0]), float(end[1] - start[1])],
            dtype=np.float64,
        )
        return float(np.linalg.norm(self.world_per_pixel @ pixel_delta))

    @staticmethod
    def _point_along_segment(
        start: Point,
        end: Point,
        distance_m: float,
        segment_length_m: float,
    ) -> Point:
        if segment_length_m <= 1e-9:
            return start
        fraction = min(1.0, max(0.0, distance_m / segment_length_m))
        return (
            int(round(start[0] + fraction * (end[0] - start[0]))),
            int(round(start[1] + fraction * (end[1] - start[1]))),
        )

    def _line_is_walkable(
        self,
        walkable: np.ndarray,
        start_xy: Point,
        end_xy: Point,
    ) -> bool:
        start_y, start_x = self._point_to_cell(start_xy)
        end_y, end_x = self._point_to_cell(end_xy)
        height, width = walkable.shape
        if not (
            0 <= start_y < height
            and 0 <= start_x < width
            and 0 <= end_y < height
            and 0 <= end_x < width
        ):
            return False

        delta_y = end_y - start_y
        delta_x = end_x - start_x
        sample_count = 2 * max(abs(delta_y), abs(delta_x))
        if sample_count == 0:
            return bool(walkable[start_y, start_x])

        previous_cell: Optional[Tuple[int, int]] = None
        for sample_idx in range(sample_count + 1):
            fraction = sample_idx / sample_count
            cell_y = int(round(start_y + delta_y * fraction))
            cell_x = int(round(start_x + delta_x * fraction))
            if not walkable[cell_y, cell_x]:
                return False
            if previous_cell is not None and previous_cell != (cell_y, cell_x):
                prev_y, prev_x = previous_cell
                if cell_y != prev_y and cell_x != prev_x:
                    if not walkable[prev_y, cell_x] or not walkable[cell_y, prev_x]:
                        return False
            previous_cell = (cell_y, cell_x)
        return True

    def _densify_path(self, control_points: List[Point]) -> List[Point]:
        if not control_points:
            return []

        dense_path = [control_points[0]]
        for start, end in zip(control_points, control_points[1:]):
            span_x = abs(end[0] - start[0]) / self.grid_step_x_px
            span_y = abs(end[1] - start[1]) / self.grid_step_y_px
            sample_count = max(1, int(math.ceil(max(span_x, span_y))))
            for sample_idx in range(1, sample_count + 1):
                fraction = sample_idx / sample_count
                point = (
                    int(round(start[0] + (end[0] - start[0]) * fraction)),
                    int(round(start[1] + (end[1] - start[1]) * fraction)),
                )
                if point != dense_path[-1]:
                    dense_path.append(point)
        return dense_path

    @staticmethod
    def _world_distance(
        first: Optional[WorldPoint], second: Optional[WorldPoint]
    ) -> Optional[float]:
        if first is None or second is None:
            return None
        return math.hypot(second[0] - first[0], second[1] - first[1])

    def _point_distance(
        self,
        first_xy: Point,
        second_xy: Point,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[float, str]:
        first_world = curr_world_xz
        if first_world is None and point_to_world is not None:
            first_world = point_to_world(first_xy)
        second_world = point_to_world(second_xy) if point_to_world is not None else None
        distance_m = self._world_distance(first_world, second_world)
        if distance_m is None:
            raise ValueError(
                "A* path tracking requires valid Unity world coordinates."
            )
        return distance_m, "m"

    def _should_replan(
        self,
        curr_xy: Point,
        *,
        walkable: np.ndarray,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[bool, str]:
        if not self.last_path:
            self.dynamic_blocked_count = 0
            return True, "no_cached_path"

        closest_idx, closest_dist, unit = self._closest_path_index(
            self.last_path,
            curr_xy,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        if closest_dist > self.path_replan_distance_m:
            self.dynamic_blocked_count = 0
            return True, f"off_path_{closest_dist:.2f}{unit}"
        if closest_idx >= len(self.last_path) - 1:
            self.dynamic_blocked_count = 0
            return True, "path_exhausted"
        if not self.last_action_was_move:
            self.dynamic_blocked_count = 0
            return False, (
                f"cached_path_dist_{closest_dist:.2f}{unit}_dynamic_check_paused"
            )
        blockage = self._remaining_path_blockage(
            walkable,
            self.last_path,
            closest_idx,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        if blockage is not None:
            blocked_idx, blocked_xy, blocked_distance_m = blockage
            self.dynamic_blocked_count += 1
            blocked_reason = (
                f"dynamic_obstacle_idx_{blocked_idx}_at_{blocked_xy}_"
                f"dist_{blocked_distance_m:.2f}m"
            )
            if self.dynamic_blocked_count >= self.dynamic_replan_confirm_steps:
                self.dynamic_blocked_count = 0
                return True, blocked_reason
            return False, (
                f"{blocked_reason}_pending_"
                f"{self.dynamic_blocked_count}_of_"
                f"{self.dynamic_replan_confirm_steps}"
            )

        self.dynamic_blocked_count = 0
        return False, f"cached_path_dist_{closest_dist:.2f}{unit}"

    def _remaining_path_blockage(
        self,
        walkable: np.ndarray,
        path: List[Point],
        closest_idx: int,
        *,
        curr_world_xz: Optional[WorldPoint],
        point_to_world: Optional[PointToWorld],
    ) -> Optional[Tuple[int, Point, float]]:
        if curr_world_xz is None or point_to_world is None:
            return None

        distance_ahead_m = 0.0
        previous_point: Optional[Point] = None
        previous_world = (
            float(curr_world_xz[0]),
            float(curr_world_xz[1]),
        )
        for path_idx in range(closest_idx + 1, len(path)):
            point = path[path_idx]
            point_world = point_to_world(point)
            if point_world is None:
                return None
            segment_m = self._world_distance(previous_world, point_world)
            if segment_m is None:
                return None
            distance_ahead_m += segment_m
            if distance_ahead_m > self.dynamic_replan_lookahead_m:
                break

            cell_y, cell_x = self._point_to_cell(point)
            blocked = not (
                0 <= cell_y < walkable.shape[0]
                and 0 <= cell_x < walkable.shape[1]
                and walkable[cell_y, cell_x]
            )
            if not blocked and previous_point is not None:
                blocked = not self._line_is_walkable(
                    walkable,
                    previous_point,
                    point,
                )
            if blocked:
                return path_idx, point, distance_ahead_m
            previous_point = point
            previous_world = point_world
        return None

    def _build_follow_waypoints(
        self,
        path: List[Point],
        curr_xy: Point,
        target_xy: Point,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> List[Point]:
        if not path:
            return []

        waypoints: List[Point] = []
        distance_since_last = 0.0
        prev_world = curr_world_xz
        if point_to_world is None or prev_world is None:
            raise ValueError(
                "A* waypoint construction requires a minimap-to-world projection."
            )
        for point in path[1:]:
            point_world = point_to_world(point)
            segment_m = self._world_distance(prev_world, point_world)
            if segment_m is None:
                raise ValueError("Failed to project an A* waypoint into world space.")
            distance_since_last += segment_m
            prev_world = point_world
            if distance_since_last >= self.waypoint_distance_m:
                waypoints.append(point)
                distance_since_last = 0.0

        path_end = (int(path[-1][0]), int(path[-1][1]))
        if not waypoints or waypoints[-1] != path_end:
            waypoints.append(path_end)
        return waypoints

    def _current_follow_waypoint(
        self,
        curr_xy: Point,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[Point, int]:
        if not self.follow_waypoints:
            return curr_xy, 0

        nearest_idx = self.follow_waypoint_index
        nearest_dist = float("inf")
        for idx in range(self.follow_waypoint_index, len(self.follow_waypoints)):
            waypoint = self.follow_waypoints[idx]
            distance, _ = self._point_distance(
                curr_xy,
                waypoint,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            if distance < nearest_dist:
                nearest_idx = idx
                nearest_dist = distance
        self.follow_waypoint_index = max(self.follow_waypoint_index, nearest_idx)

        while self.follow_waypoint_index < len(self.follow_waypoints) - 1:
            waypoint = self.follow_waypoints[self.follow_waypoint_index]
            distance, _ = self._point_distance(
                curr_xy,
                waypoint,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            if distance > self.waypoint_reach_m:
                break
            self.follow_waypoint_index += 1

        return (
            self.follow_waypoints[self.follow_waypoint_index],
            self.follow_waypoint_index,
        )

    def _lookahead_follow_waypoint(
        self,
        curr_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[Point, int, str]:
        if not self.follow_waypoints:
            return curr_xy, 0, "no_follow_waypoints"

        start_idx = self.follow_waypoint_index
        candidate_idx = start_idx
        distance_ahead = 0.0
        prev_world = curr_world_xz
        if point_to_world is None or prev_world is None:
            raise ValueError(
                "A* lookahead requires a minimap-to-world projection."
            )
        for idx in range(start_idx, len(self.follow_waypoints)):
            point = self.follow_waypoints[idx]
            point_world = point_to_world(point)
            segment_m = self._world_distance(prev_world, point_world)
            if segment_m is None:
                raise ValueError("Failed to project an A* lookahead point.")
            distance_ahead += segment_m
            prev_world = point_world
            candidate_idx = idx
            if distance_ahead >= self.lookahead_m:
                break

        final_idx = len(self.follow_waypoints) - 1
        for idx in range(candidate_idx, final_idx + 1):
            point = self.follow_waypoints[idx]
            relative = self._relative_angle_to_point(
                curr_xy,
                point,
                agent_theta,
                curr_world_xz=curr_world_xz,
                waypoint_world_xz=(
                    point_to_world(point) if point_to_world is not None else None
                ),
            )
            if relative is None:
                continue
            if abs(relative) <= self.front_cone_deg:
                return (
                    point,
                    idx,
                    f"lookahead_from_{start_idx}_dist_{distance_ahead:.2f}m",
                )

        point = self.follow_waypoints[candidate_idx]
        return point, candidate_idx, f"lookahead_outside_front_cone_from_{start_idx}"

    def _closest_path_index(
        self,
        path: List[Point],
        curr_xy: Point,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[int, float, str]:
        best_idx = 0
        best_dist = float("inf")
        best_unit = "m"
        for idx, point in enumerate(path):
            dist, unit = self._point_distance(
                curr_xy,
                point,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
                best_unit = unit
        return best_idx, best_dist, best_unit

    def _stanley_action(
        self,
        path: List[Point],
        curr_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: WorldPoint,
        point_to_world: PointToWorld,
    ) -> Tuple[str, float, float, float, int, Point]:
        tracking_path = self.tracking_path if len(self.tracking_path) >= 2 else path
        segment = self._nearest_tracking_segment(
            tracking_path,
            curr_world_xz,
            point_to_world,
        )
        if segment is None:
            fallback_waypoint = path[-1]
            action, steering_error = self._action_toward(
                curr_xy,
                fallback_waypoint,
                agent_theta,
                curr_world_xz=curr_world_xz,
                waypoint_world_xz=point_to_world(fallback_waypoint),
            )
            self.last_tracking_metrics = {
                "controller": "waypoint_fallback",
                "segment": -1,
                "cross_track_m": 0.0,
                "heading_error_deg": float(steering_error),
                "steering_deg": float(steering_error),
            }
            return (
                action,
                steering_error,
                steering_error,
                0.0,
                -1,
                fallback_waypoint,
            )

        segment_idx, closest_world, segment_vector, segment_length = segment
        segment_dx, segment_dz = segment_vector
        path_yaw = math.degrees(math.atan2(segment_dx, segment_dz)) % 360.0
        heading_error = self._normalize_angle_deg(path_yaw - float(agent_theta))
        offset_x = float(curr_world_xz[0]) - closest_world[0]
        offset_z = float(curr_world_xz[1]) - closest_world[1]
        cross_track_m = (
            segment_dx * offset_z - segment_dz * offset_x
        ) / segment_length
        correction_deg = math.degrees(
            math.atan2(
                self.stanley_gain * cross_track_m,
                self.stanley_softening_m,
            )
        )
        correction_deg = max(
            -self.stanley_max_correction_deg,
            min(self.stanley_max_correction_deg, correction_deg),
        )
        steering_error = self._normalize_angle_deg(
            heading_error + correction_deg
        )
        action = self._action_from_steering_error(
            steering_error,
            forward_tolerance_deg=self.stanley_forward_tolerance_deg,
            drive_turn_tolerance_deg=self.drive_turn_tolerance_deg,
        )
        self.last_tracking_metrics = {
            "controller": "stanley",
            "segment": int(segment_idx),
            "cross_track_m": float(cross_track_m),
            "heading_error_deg": float(heading_error),
            "steering_deg": float(steering_error),
        }
        tracking_waypoint = tracking_path[min(segment_idx + 1, len(tracking_path) - 1)]
        return (
            action,
            steering_error,
            heading_error,
            cross_track_m,
            segment_idx,
            tracking_waypoint,
        )

    def _nearest_tracking_segment(
        self,
        tracking_path: List[Point],
        curr_world_xz: WorldPoint,
        point_to_world: PointToWorld,
    ) -> Optional[Tuple[int, WorldPoint, WorldPoint, float]]:
        if len(tracking_path) < 2:
            return None

        start_idx = min(self.path_segment_index, len(tracking_path) - 2)
        best_segment = None
        best_distance_sq = float("inf")
        curr_x = float(curr_world_xz[0])
        curr_z = float(curr_world_xz[1])
        for segment_idx in range(start_idx, len(tracking_path) - 1):
            start_world = point_to_world(tracking_path[segment_idx])
            end_world = point_to_world(tracking_path[segment_idx + 1])
            if start_world is None or end_world is None:
                continue
            segment_dx = float(end_world[0]) - float(start_world[0])
            segment_dz = float(end_world[1]) - float(start_world[1])
            segment_length_sq = segment_dx * segment_dx + segment_dz * segment_dz
            if segment_length_sq <= 1e-12:
                continue
            projection = (
                (curr_x - float(start_world[0])) * segment_dx
                + (curr_z - float(start_world[1])) * segment_dz
            ) / segment_length_sq
            projection = max(0.0, min(1.0, projection))
            closest_world = (
                float(start_world[0]) + projection * segment_dx,
                float(start_world[1]) + projection * segment_dz,
            )
            distance_sq = (
                (curr_x - closest_world[0]) ** 2
                + (curr_z - closest_world[1]) ** 2
            )
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_segment = (
                    segment_idx,
                    closest_world,
                    (segment_dx, segment_dz),
                    math.sqrt(segment_length_sq),
                )

        if best_segment is not None:
            self.path_segment_index = best_segment[0]
        return best_segment

    def _greedy_action(
        self,
        curr_xy: Point,
        target_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        target_world_xz: Optional[WorldPoint] = None,
    ):
        action, relative = self._action_toward(
            curr_xy,
            target_xy,
            agent_theta,
            curr_world_xz=curr_world_xz,
            waypoint_world_xz=target_world_xz,
        )
        return action, f"target_relative={relative:.1f}deg"

    def _terminal_action_toward(
        self,
        curr_xy: Point,
        target_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        target_world_xz: Optional[WorldPoint] = None,
    ) -> Tuple[str, float]:
        relative = self._relative_angle_to_point(
            curr_xy,
            target_xy,
            agent_theta,
            curr_world_xz=curr_world_xz,
            waypoint_world_xz=target_world_xz,
        )
        if relative is None:
            return "stop", 0.0
        if abs(relative) <= self.turn_tolerance_deg:
            self.last_turn_sign = 0
            return "astar forward", relative

        self.last_turn_sign = 1 if relative > 0 else -1
        if self.last_turn_sign > 0:
            return "astar turn right", relative
        return "astar turn left", relative

    def _action_toward(
        self,
        curr_xy: Point,
        waypoint_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        waypoint_world_xz: Optional[WorldPoint] = None,
        drive_turn_tolerance_deg: Optional[float] = None,
    ) -> Tuple[str, float]:
        relative = self._relative_angle_to_point(
            curr_xy,
            waypoint_xy,
            agent_theta,
            curr_world_xz=curr_world_xz,
            waypoint_world_xz=waypoint_world_xz,
        )
        if relative is None:
            return "stop", 0.0

        drive_turn_tolerance = (
            self.drive_turn_tolerance_deg
            if drive_turn_tolerance_deg is None
            else float(drive_turn_tolerance_deg)
        )
        action = self._action_from_steering_error(
            relative,
            forward_tolerance_deg=self.forward_priority_tolerance_deg,
            drive_turn_tolerance_deg=drive_turn_tolerance,
        )
        return action, relative

    def _action_from_steering_error(
        self,
        steering_error_deg: float,
        *,
        forward_tolerance_deg: float,
        drive_turn_tolerance_deg: float,
    ) -> str:
        if abs(steering_error_deg) <= forward_tolerance_deg:
            if abs(steering_error_deg) <= self.hysteresis_reset_deg:
                self.last_turn_sign = 0
            return "forward"

        turn_sign = 1 if steering_error_deg > 0 else -1
        if (
            self.last_turn_sign
            and turn_sign != self.last_turn_sign
            and abs(steering_error_deg) >= self.hysteresis_lock_deg
        ):
            turn_sign = self.last_turn_sign
        self.last_turn_sign = turn_sign

        if abs(steering_error_deg) <= drive_turn_tolerance_deg:
            if turn_sign > 0:
                return "astar forward turn right"
            return "astar forward turn left"
        if turn_sign > 0:
            return "astar turn right"
        return "astar turn left"

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        normalized = float(angle_deg)
        while normalized > 180.0:
            normalized -= 360.0
        while normalized < -180.0:
            normalized += 360.0
        return normalized

    def _relative_angle_to_point(
        self,
        curr_xy: Point,
        waypoint_xy: Point,
        agent_theta: float,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        waypoint_world_xz: Optional[WorldPoint] = None,
    ) -> Optional[float]:
        if curr_world_xz is not None and waypoint_world_xz is not None:
            dx = waypoint_world_xz[0] - curr_world_xz[0]
            dz = waypoint_world_xz[1] - curr_world_xz[1]
            if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                return None
            target_yaw = math.degrees(math.atan2(dx, dz)) % 360.0
            relative = target_yaw - float(agent_theta)
            while relative > 180.0:
                relative -= 360.0
            while relative < -180.0:
                relative += 360.0
            return relative

        dx = waypoint_xy[0] - curr_xy[0]
        dy = waypoint_xy[1] - curr_xy[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        target_bearing = math.degrees(math.atan2(dy, dx))
        agent_bearing = (float(agent_theta) - 180.0) % 360.0
        relative = target_bearing - agent_bearing
        while relative > 180.0:
            relative -= 360.0
        while relative < -180.0:
            relative += 360.0
        return relative

    def _update_stuck_state(
        self,
        curr_xy: Point,
        curr_world_xz: WorldPoint,
        agent_theta: float,
        *,
        point_to_world: Optional[PointToWorld] = None,
    ) -> str:
        reason = "last_action_not_move"
        if self.last_action_was_move:
            if self.prev_world_xz is not None:
                motion = math.hypot(
                    float(curr_world_xz[0]) - self.prev_world_xz[0],
                    float(curr_world_xz[1]) - self.prev_world_xz[1],
                )
                threshold = self.stuck_distance_m
                unit = "m"
            else:
                motion = float("inf")
                threshold = 0.0
                unit = "unknown"

            if motion <= threshold:
                self.stuck_count += 1
                reason = f"low_motion_{motion:.3f}{unit}"
            else:
                self.stuck_count = 0
                reason = f"moved_{motion:.3f}{unit}"

        self.prev_world_xz = (float(curr_world_xz[0]), float(curr_world_xz[1]))

        if self.stuck_count >= self.stuck_steps:
            if point_to_world is not None:
                self._add_stuck_obstacle(
                    curr_xy,
                    agent_theta,
                )
            self.recovery_count = self.recovery_turn_steps
            self.recovery_turn_sign = -self.last_turn_sign if self.last_turn_sign else 1
            self.stuck_count = 0
            self.last_turn_sign = 0
            self.last_path = []
            self.tracking_path = []
            self.path_segment_index = 0
            self.follow_waypoints = []
            self.follow_waypoint_index = 0
            reason = f"{reason}_start_recovery"

        return reason

    def _add_stuck_obstacle(
        self,
        curr_xy: Point,
        agent_theta: float,
    ) -> None:
        yaw = math.radians(float(agent_theta) % 360.0)
        world_offset = np.array(
            [
                math.sin(yaw) * self.stuck_block_ahead_m,
                math.cos(yaw) * self.stuck_block_ahead_m,
            ],
            dtype=np.float64,
        )
        pixel_offset = np.linalg.solve(self.world_per_pixel, world_offset)
        ox = int(round(float(curr_xy[0]) + float(pixel_offset[0])))
        oy = int(round(float(curr_xy[1]) + float(pixel_offset[1])))
        self.virtual_obstacles.append(
            (
                ox,
                oy,
                self.stuck_block_radius_x_px,
                self.stuck_block_radius_y_px,
                self.stuck_block_ttl_steps,
            )
        )
        self.virtual_obstacles = self.virtual_obstacles[-12:]

    def _decay_virtual_obstacles(self) -> None:
        if not self.virtual_obstacles:
            return
        kept = []
        for ox, oy, radius_x, radius_y, ttl in self.virtual_obstacles:
            if ttl > 1:
                kept.append((ox, oy, radius_x, radius_y, ttl - 1))
        self.virtual_obstacles = kept

    def _recovery_action(self) -> str:
        self.recovery_count = max(0, self.recovery_count - 1)
        if self.recovery_turn_sign > 0:
            return "astar turn right"
        return "astar turn left"

    def _record_action(self, action: str) -> None:
        self.last_action_was_move = "forward" in action or action in {
            "back",
            "strafe left",
            "strafe right",
        }

    def _point_to_cell(self, point: Point) -> Tuple[int, int]:
        return (
            int(point[1]) // self.grid_step_y_px,
            int(point[0]) // self.grid_step_x_px,
        )

    def _cell_to_point(self, cell: Tuple[int, int]) -> Point:
        y, x = cell
        return (
            int(x * self.grid_step_x_px + self.grid_step_x_px / 2),
            int(y * self.grid_step_y_px + self.grid_step_y_px / 2),
        )

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            max_val = float(arr.max()) if arr.size else 1.0
            if max_val <= 1.01:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
