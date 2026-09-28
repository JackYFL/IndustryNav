"""Calibrated occupancy geometry for task sampling and privileged RL rewards.

This is an approximate *image-derived* map, not Unity's physics/NavMesh ground
truth. It supplies distances, never policy actions. Eight-connected edges may
not cut blocked corners. The fixed map avoids reward changes caused solely by
moving objects; their physical effects are handled by online collision costs.
"""

from __future__ import annotations

import heapq
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class NavigationMapCoverageError(ValueError):
    """A valid physical location is not represented in the approximate map."""


class NavigationMap:
    def __init__(self, walkable: np.ndarray, metadata: dict):
        self.walkable = np.asarray(walkable, dtype=bool)
        self.metadata = dict(metadata)
        if self.walkable.ndim != 2 or not self.walkable.any():
            raise ValueError("Navigation map must contain a nonempty free grid")
        self.origin = np.asarray(metadata["world_origin"], dtype=np.float64)
        self.basis = np.asarray(metadata["world_per_cell"], dtype=np.float64)
        if self.origin.shape != (2,) or self.basis.shape != (2, 2):
            raise ValueError("Invalid map affine dimensions")
        if not np.isfinite(self.basis).all() or abs(np.linalg.det(self.basis)) < 1e-9:
            raise ValueError("Degenerate navigation-map affine")
        self.inverse = np.linalg.inv(self.basis)
        self.height, self.width = self.walkable.shape
        self.free_cells = np.column_stack(np.nonzero(self.walkable)[::-1])
        self.free_world = self.cell_to_world(self.free_cells)
        # Do not merge diagonally touching rooms across blocked corners.
        self.component_count, self.components = cv2.connectedComponents(
            self.walkable.astype(np.uint8), connectivity=4,
        )
        self.adjacency = self._adjacency()
        self._fields: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def _adjacency(self) -> list[list[tuple[int, float]]]:
        result: list[list[tuple[int, float]]] = [[] for _ in range(self.walkable.size)]
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cost = float(np.linalg.norm(self.basis @ (dx, dy)))
            for x, y in self.free_cells:
                nx, ny = int(x + dx), int(y + dy)
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if not self.walkable[ny, nx]:
                    continue
                if dx and dy and not (self.walkable[y, nx] and self.walkable[ny, x]):
                    continue
                a, b = int(y * self.width + x), ny * self.width + nx
                result[a].append((b, cost))
                result[b].append((a, cost))
        return result

    def cell_to_world(self, cells):
        return np.asarray(cells, dtype=float) @ self.basis.T + self.origin

    def world_to_cell(self, world):
        return (np.asarray(world, dtype=float) - self.origin) @ self.inverse.T

    def distance_field(self, goal_world, *, reach_m: float = 0.0) -> np.ndarray:
        goal = np.asarray(goal_world, dtype=float)
        key = (*np.round(goal, 5), round(float(reach_m), 5))
        if key in self._fields:
            self._fields.move_to_end(key)
            return self._fields[key]
        offsets = np.linalg.norm(self.free_world - goal, axis=1)
        if reach_m > 0:
            selected = np.flatnonzero(offsets <= reach_m)
        else:
            selected = np.array([int(np.argmin(offsets))])
        if not len(selected) or float(offsets.min()) > max(reach_m, 1.0):
            raise NavigationMapCoverageError("Goal is not close to reachable free space")
        distances = np.full(self.walkable.size, np.inf, dtype=np.float64)
        queue = []
        for index in selected:
            x, y = self.free_cells[index]
            flat = int(y * self.width + x)
            cost = 0.0 if reach_m > 0 else float(offsets[index])
            distances[flat] = cost
            heapq.heappush(queue, (cost, flat))
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            for other, cost in self.adjacency[node]:
                candidate = distance + cost
                if candidate < distances[other]:
                    distances[other] = candidate
                    heapq.heappush(queue, (candidate, other))
        field = distances.reshape(self.walkable.shape)
        self._fields[key] = field
        if len(self._fields) > 16:
            self._fields.popitem(last=False)
        return field

    def distance_at(self, world, field: np.ndarray, *, max_snap_m: float = 1.0):
        """Continuous bilinear distance in free cells; bounded local fallback.

        The fallback only uses nearby cells, never a global snap across the
        scene. Unreachable/out-of-map states return None and earn no progress.
        """
        cell = self.world_to_cell(world)
        x, y = np.floor(cell).astype(int)
        if 0 <= x < self.width - 1 and 0 <= y < self.height - 1:
            values = field[y:y + 2, x:x + 2]
            if np.isfinite(values).all() and self.walkable[y:y + 2, x:x + 2].all():
                u, v = cell - (x, y)
                return float(values[0, 0] * (1-u) * (1-v)
                             + values[0, 1] * u * (1-v)
                             + values[1, 0] * (1-u) * v + values[1, 1] * u * v)
        rx, ry = np.rint(cell).astype(int)
        if not (0 <= rx < self.width and 0 <= ry < self.height):
            return None
        if self.walkable[ry, rx] and not np.isfinite(field[ry, rx]):
            return None
        # The physical clearance fringe is measured in meters, not one grid
        # cell. A single-cell search wrongly lost coverage after grid refinement.
        # Find the closest *free* cell first, including unreachable components;
        # never prefer a farther reachable cell through a wall to fabricate progress.
        cell_scale = float(np.linalg.svd(self.basis, compute_uv=False).min())
        radius = int(np.ceil(max_snap_m / cell_scale)) + 1
        candidates = []
        for ny in range(max(0, ry-radius), min(self.height, ry+radius+1)):
            for nx in range(max(0, rx-radius), min(self.width, rx+radius+1)):
                if not self.walkable[ny, nx]:
                    continue
                offset = float(np.linalg.norm(self.cell_to_world((nx, ny)) - world))
                if offset <= max_snap_m:
                    candidates.append((offset, float(field[ny, nx]) + offset))
        if not candidates:
            return None
        result = min(candidates)[1]
        return result if np.isfinite(result) else None

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "geometry.npz", walkable=self.walkable)
        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        with np.load(directory / "geometry.npz", allow_pickle=False) as data:
            grid = data["walkable"]
        return cls(grid, json.loads((directory / "metadata.json").read_text()))
