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

    def __init__(
        self,
        grid_step: int = ASTAR_DEFAULTS.grid_step,
        obstacle_threshold: int = ASTAR_DEFAULTS.obstacle_threshold,
        min_free_ratio: float = ASTAR_DEFAULTS.min_free_ratio,
        obstacle_inflate_px: int = ASTAR_DEFAULTS.obstacle_inflate_px,
        marker_clear_px: int = ASTAR_DEFAULTS.marker_clear_px,
        target_change_replan_px: float = ASTAR_DEFAULTS.target_change_replan_px,
        turn_tolerance_deg: float = ASTAR_DEFAULTS.turn_tolerance_deg,
        forward_priority_tolerance_deg: float = ASTAR_DEFAULTS.forward_priority_tolerance_deg,
        drive_turn_tolerance_deg: float = ASTAR_DEFAULTS.drive_turn_tolerance_deg,
        waypoint_distance_m: float = ASTAR_DEFAULTS.waypoint_distance_m,
        waypoint_reach_m: float = ASTAR_DEFAULTS.waypoint_reach_m,
        path_replan_distance_m: float = ASTAR_DEFAULTS.path_replan_distance_m,
        terminal_approach_m: float = ASTAR_DEFAULTS.terminal_approach_m,
        lookahead_m: float = ASTAR_DEFAULTS.lookahead_m,
        front_cone_deg: float = ASTAR_DEFAULTS.front_cone_deg,
        hysteresis_reset_deg: float = ASTAR_DEFAULTS.hysteresis_reset_deg,
        hysteresis_lock_deg: float = ASTAR_DEFAULTS.hysteresis_lock_deg,
        stuck_world_epsilon: float = ASTAR_DEFAULTS.stuck_world_epsilon,
        stuck_px_epsilon: float = ASTAR_DEFAULTS.stuck_px_epsilon,
        stuck_steps: int = ASTAR_DEFAULTS.stuck_steps,
        recovery_turn_steps: int = ASTAR_DEFAULTS.recovery_turn_steps,
        stuck_block_ahead_px: float = ASTAR_DEFAULTS.stuck_block_ahead_px,
        stuck_block_radius_px: float = ASTAR_DEFAULTS.stuck_block_radius_px,
        stuck_block_ttl_steps: int = ASTAR_DEFAULTS.stuck_block_ttl_steps,
        proxy_stop_real_dist_m: float = ASTAR_DEFAULTS.proxy_stop_real_dist_m,
        pixel_scale: float = 1.0,
        debug_viz: bool = False,
        debug_dir: Optional[str] = None,
    ):
        self.pixel_scale = float(pixel_scale)
        if self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive.")

        def scaled_int(value: float, minimum: int = 1) -> int:
            return max(minimum, int(round(float(value) * self.pixel_scale)))

        self.grid_step = scaled_int(grid_step)
        self.obstacle_threshold = int(obstacle_threshold)
        self.min_free_ratio = float(min_free_ratio)
        self.obstacle_inflate_px = scaled_int(obstacle_inflate_px, minimum=0)
        self.marker_clear_px = scaled_int(marker_clear_px)
        self.target_change_replan_px = float(target_change_replan_px) * self.pixel_scale
        self.turn_tolerance_deg = float(turn_tolerance_deg)
        self.forward_priority_tolerance_deg = float(forward_priority_tolerance_deg)
        self.drive_turn_tolerance_deg = float(drive_turn_tolerance_deg)
        self.waypoint_distance_m = float(waypoint_distance_m)
        self.waypoint_reach_m = float(waypoint_reach_m)
        self.path_replan_distance_m = float(path_replan_distance_m)
        self.terminal_approach_m = float(terminal_approach_m)
        self.lookahead_m = float(lookahead_m)
        self.front_cone_deg = float(front_cone_deg)
        self.hysteresis_reset_deg = float(hysteresis_reset_deg)
        self.hysteresis_lock_deg = float(hysteresis_lock_deg)
        self.stuck_world_epsilon = float(stuck_world_epsilon)
        self.stuck_px_epsilon = float(stuck_px_epsilon) * self.pixel_scale
        self.stuck_steps = int(stuck_steps)
        self.recovery_turn_steps = int(recovery_turn_steps)
        self.stuck_block_ahead_px = float(stuck_block_ahead_px) * self.pixel_scale
        self.stuck_block_radius_px = float(stuck_block_radius_px) * self.pixel_scale
        self.stuck_block_ttl_steps = int(stuck_block_ttl_steps)
        self.proxy_stop_real_dist_m = float(proxy_stop_real_dist_m)
        self.debug_viz = bool(debug_viz)
        self.debug_dir = debug_dir
        self.last_path: List[Point] = []
        self.follow_waypoints: List[Point] = []
        self.follow_waypoint_index = 0
        self.last_turn_sign = 0
        self.last_action_was_move = False
        self.prev_curr_xy: Optional[Point] = None
        self.prev_world_xz: Optional[Tuple[float, float]] = None
        self.stuck_count = 0
        self.recovery_count = 0
        self.recovery_turn_sign = 1
        self.last_target_xy: Optional[Point] = None
        self.last_effective_target_xy: Optional[Point] = None
        self.last_debug = {}
        self.last_plan_status = "not_planned"
        self.virtual_obstacles: List[Tuple[int, int, float, int]] = []

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
        if distance_m <= reach_m:
            self.last_path = [curr_xy, target_xy]
            self.last_action_was_move = False
            return (
                "stop",
                f"Astar reached target vicinity; dist={distance_m:.2f}m.",
                self.last_path,
            )

        self._decay_virtual_obstacles()
        stuck_reason = self._update_stuck_state(curr_xy, curr_world_xz, agent_theta)
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
            self._record_action(action)
            return action, f"Astar fallback without minimap: {reasoning}", self.last_path

        walkable = self._build_walkable_grid(minimap_rgb, curr_xy, target_xy)
        should_replan, replan_reason = self._should_replan(
            curr_xy,
            target_xy,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        path = self.last_path
        if should_replan:
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
            self.follow_waypoints = []
            self.follow_waypoint_index = 0
            self.last_target_xy = None
            self.last_effective_target_xy = None
            self._save_debug_visualization(
                minimap_rgb, curr_xy, target_xy, fallback_path, None, step
            )
            self._record_action(action)
            return action, f"Astar found no path, fallback: {reasoning}", fallback_path

        self.last_path = path
        self.last_target_xy = target_xy
        if self._using_unreachable_proxy() and self.last_effective_target_xy is not None:
            proxy_distance, proxy_unit = self._point_distance(
                curr_xy,
                self.last_effective_target_xy,
                curr_world_xz=curr_world_xz,
                point_to_world=point_to_world,
            )
            if (
                proxy_distance <= reach_m
                and distance_m <= self.proxy_stop_real_dist_m
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
        else:
            action, relative = self._action_toward(
                curr_xy,
                waypoint,
                agent_theta,
                curr_world_xz=curr_world_xz,
                waypoint_world_xz=(
                    point_to_world(waypoint) if point_to_world is not None else None
                ),
            )
        self._save_debug_visualization(
            minimap_rgb, curr_xy, target_xy, path, waypoint, step
        )
        self._record_action(action)
        return (
            action,
            "Astar "
            f"{'replanned' if should_replan else 'tracking'}({replan_reason}) "
            f"stuck={self.stuck_count} "
            f"plan_status={self.last_plan_status} "
            f"path_len={len(path)} closest_idx={closest_idx} "
            f"closest_dist={closest_dist:.2f}{closest_unit} "
            f"anchor_idx={anchor_idx}/{len(self.follow_waypoints) - 1} "
            f"anchor={anchor_waypoint} action_idx={waypoint_idx} "
            f"waypoint={waypoint} waypoint_reason={waypoint_reason} "
            f"relative={relative:.1f}deg last_turn={self.last_turn_sign}",
            path,
        )

    def _build_walkable_grid(
        self, minimap_rgb: np.ndarray, curr_xy: Point, target_xy: Point
    ) -> np.ndarray:
        rgb = self._to_uint8_rgb(minimap_rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        threshold_free = gray > self.obstacle_threshold
        free = threshold_free.copy()

        # Keep the colored spawn/target markers from becoming obstacles.
        free_u8 = free.astype(np.uint8)
        for point in (curr_xy, target_xy):
            cv2.circle(free_u8, (int(point[0]), int(point[1])), self.marker_clear_px, 1, -1)
        free = free_u8.astype(bool)

        if self.obstacle_inflate_px > 0:
            kernel_size = self.obstacle_inflate_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            obstacles = cv2.dilate((~free).astype(np.uint8), kernel, iterations=1)
            free = obstacles == 0
            free_u8 = free.astype(np.uint8)
            for point in (curr_xy, target_xy):
                cv2.circle(free_u8, (int(point[0]), int(point[1])), self.marker_clear_px, 1, -1)
            free = free_u8.astype(bool)

        if self.virtual_obstacles:
            free_u8 = free.astype(np.uint8)
            for ox, oy, radius, _ttl in self.virtual_obstacles:
                cv2.circle(free_u8, (int(ox), int(oy)), int(radius), 0, -1)
            free = free_u8.astype(bool)

        h, w = free.shape
        gh = int(math.ceil(h / self.grid_step))
        gw = int(math.ceil(w / self.grid_step))
        walkable = np.zeros((gh, gw), dtype=bool)
        for gy in range(gh):
            y0 = gy * self.grid_step
            y1 = min(h, y0 + self.grid_step)
            for gx in range(gw):
                x0 = gx * self.grid_step
                x1 = min(w, x0 + self.grid_step)
                walkable[gy, gx] = free[y0:y1, x0:x1].mean() >= self.min_free_ratio

        self.last_debug = {
            "gray": gray,
            "threshold_free": threshold_free,
            "inflated_free": free,
            "walkable": walkable,
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
                return self._reconstruct_path(came_from, current, target_xy)

            cy, cx = current
            for dy, dx, move_cost in neighbors:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < walkable.shape[0] and 0 <= nx < walkable.shape[1]):
                    continue
                if not walkable[ny, nx]:
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
        return self._reconstruct_path(came_from, best_reachable, proxy_xy)

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
        target_xy: Point,
        *,
        curr_world_xz: Optional[WorldPoint] = None,
        point_to_world: Optional[PointToWorld] = None,
    ) -> Tuple[bool, str]:
        if not self.last_path:
            return True, "no_cached_path"
        if self.last_target_xy is None:
            return True, "no_cached_target"
        if math.hypot(
            target_xy[0] - self.last_target_xy[0],
            target_xy[1] - self.last_target_xy[1],
        ) > self.target_change_replan_px:
            return True, "target_changed"

        closest_idx, closest_dist, unit = self._closest_path_index(
            self.last_path,
            curr_xy,
            curr_world_xz=curr_world_xz,
            point_to_world=point_to_world,
        )
        if closest_dist > self.path_replan_distance_m:
            return True, f"off_path_{closest_dist:.2f}{unit}"
        if closest_idx >= len(self.last_path) - 1:
            return True, "path_exhausted"
        return False, f"cached_path_dist_{closest_dist:.2f}{unit}"

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

        if abs(relative) <= self.forward_priority_tolerance_deg:
            if abs(relative) <= self.hysteresis_reset_deg:
                self.last_turn_sign = 0
            return "forward", relative

        turn_sign = 1 if relative > 0 else -1
        if (
            self.last_turn_sign
            and turn_sign != self.last_turn_sign
            and abs(relative) >= self.hysteresis_lock_deg
        ):
            turn_sign = self.last_turn_sign
        self.last_turn_sign = turn_sign

        drive_turn_tolerance = (
            self.drive_turn_tolerance_deg
            if drive_turn_tolerance_deg is None
            else float(drive_turn_tolerance_deg)
        )
        if abs(relative) <= drive_turn_tolerance:
            if turn_sign > 0:
                return "astar forward turn right", relative
            return "astar forward turn left", relative
        if turn_sign > 0:
            return "astar turn right", relative
        return "astar turn left", relative

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
        curr_world_xz: Optional[Tuple[float, float]],
        agent_theta: float,
    ) -> str:
        reason = "last_action_not_move"
        if self.last_action_was_move:
            if curr_world_xz is not None and self.prev_world_xz is not None:
                motion = math.hypot(
                    float(curr_world_xz[0]) - self.prev_world_xz[0],
                    float(curr_world_xz[1]) - self.prev_world_xz[1],
                )
                threshold = self.stuck_world_epsilon
                unit = "world"
            elif self.prev_curr_xy is not None:
                motion = math.hypot(
                    float(curr_xy[0]) - self.prev_curr_xy[0],
                    float(curr_xy[1]) - self.prev_curr_xy[1],
                )
                threshold = self.stuck_px_epsilon
                unit = "px"
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

        self.prev_curr_xy = (int(curr_xy[0]), int(curr_xy[1]))
        if curr_world_xz is not None:
            self.prev_world_xz = (float(curr_world_xz[0]), float(curr_world_xz[1]))

        if self.stuck_count >= self.stuck_steps:
            self._add_stuck_obstacle(curr_xy, agent_theta)
            self.recovery_count = self.recovery_turn_steps
            self.recovery_turn_sign = -self.last_turn_sign if self.last_turn_sign else 1
            self.stuck_count = 0
            self.last_turn_sign = 0
            self.last_path = []
            self.follow_waypoints = []
            self.follow_waypoint_index = 0
            reason = f"{reason}_start_recovery"

        return reason

    def _add_stuck_obstacle(self, curr_xy: Point, agent_theta: float) -> None:
        bearing = math.radians((float(agent_theta) - 180.0) % 360.0)
        ox = int(round(curr_xy[0] + math.cos(bearing) * self.stuck_block_ahead_px))
        oy = int(round(curr_xy[1] + math.sin(bearing) * self.stuck_block_ahead_px))
        self.virtual_obstacles.append(
            (ox, oy, self.stuck_block_radius_px, self.stuck_block_ttl_steps)
        )
        self.virtual_obstacles = self.virtual_obstacles[-12:]

    def _decay_virtual_obstacles(self) -> None:
        if not self.virtual_obstacles:
            return
        kept = []
        for ox, oy, radius, ttl in self.virtual_obstacles:
            if ttl > 1:
                kept.append((ox, oy, radius, ttl - 1))
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
        return int(point[1]) // self.grid_step, int(point[0]) // self.grid_step

    def _cell_to_point(self, cell: Tuple[int, int]) -> Point:
        y, x = cell
        return (
            int(x * self.grid_step + self.grid_step / 2),
            int(y * self.grid_step + self.grid_step / 2),
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
