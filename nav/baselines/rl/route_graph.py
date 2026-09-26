"""Topological route-memory controller for hierarchical PointGoal RL."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np


class RouteGraphController:
    """Select local waypoints on a graph built from independent A* routes.

    The graph is a compact training artifact rather than a simulator map.  At
    evaluation time this controller consumes only scene identity, GPS position,
    and the final PointGoal coordinates.  The recurrent depth policy remains
    responsible for executing each local waypoint and avoiding dynamic objects.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        lookahead_m: float | None = None,
        terminal_distance_m: float | None = None,
        max_snap_distance_m: float | None = None,
    ) -> None:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        config = payload["config"]
        self.lookahead_m = float(
            config["lookahead_m"] if lookahead_m is None else lookahead_m
        )
        self.terminal_distance_m = float(
            self.lookahead_m
            if terminal_distance_m is None
            else terminal_distance_m
        )
        self.max_snap_distance_m = float(
            config["max_snap_distance_m"]
            if max_snap_distance_m is None
            else max_snap_distance_m
        )
        if self.lookahead_m <= 0.0 or self.terminal_distance_m < 0.0:
            raise ValueError("lookahead must be positive and terminal distance nonnegative")
        if self.max_snap_distance_m <= 0.0:
            raise ValueError("max snap distance must be positive")
        self.scenes: dict[int, tuple[np.ndarray, list[list[tuple[int, float]]]]] = {}
        for scene_key, graph in payload["scenes"].items():
            nodes = np.asarray(graph["nodes"], dtype=np.float32)
            adjacency: list[list[tuple[int, float]]] = [
                [] for _ in range(len(nodes))
            ]
            for first, second in graph["edges"]:
                weight = float(np.linalg.norm(nodes[first] - nodes[second]))
                if weight <= 0.0:
                    continue
                adjacency[first].append((second, weight))
                adjacency[second].append((first, weight))
            self.scenes[int(scene_key)] = nodes, adjacency
        self.reset()

    def reset(self) -> None:
        self._cache_key: tuple[int, int] | None = None
        self._distances: np.ndarray | None = None
        self._next_hop: np.ndarray | None = None

    @staticmethod
    def _nearest_node(nodes: np.ndarray, point: np.ndarray) -> tuple[int, float]:
        squared = np.square(nodes - point[None]).sum(axis=1)
        index = int(np.argmin(squared))
        return index, math.sqrt(float(squared[index]))

    @staticmethod
    def _routes_to_target(
        adjacency: list[list[tuple[int, float]]], target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full(len(adjacency), np.inf, dtype=np.float32)
        next_hop = np.full(len(adjacency), -1, dtype=np.int64)
        distances[target] = 0.0
        queue: list[tuple[float, int]] = [(0.0, target)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > float(distances[node]) + 1.0e-6:
                continue
            for neighbor, weight in adjacency[node]:
                candidate = distance + weight
                if candidate + 1.0e-6 < float(distances[neighbor]):
                    distances[neighbor] = candidate
                    next_hop[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        return distances, next_hop

    def predict_waypoint(
        self,
        *,
        scene_id: int,
        curr_world_x: float,
        curr_world_z: float,
        target_world_x: float,
        target_world_z: float,
    ) -> tuple[float, float]:
        current = np.asarray([curr_world_x, curr_world_z], dtype=np.float32)
        target = np.asarray([target_world_x, target_world_z], dtype=np.float32)
        if float(np.linalg.norm(target - current)) <= self.terminal_distance_m:
            return float(target[0]), float(target[1])
        graph = self.scenes.get(int(scene_id))
        if graph is None or not len(graph[0]):
            return float(target[0]), float(target[1])
        nodes, adjacency = graph
        current_node, current_snap = self._nearest_node(nodes, current)
        target_node, target_snap = self._nearest_node(nodes, target)
        if max(current_snap, target_snap) > self.max_snap_distance_m:
            return float(target[0]), float(target[1])
        cache_key = (int(scene_id), target_node)
        if self._cache_key != cache_key:
            self._distances, self._next_hop = self._routes_to_target(
                adjacency, target_node
            )
            self._cache_key = cache_key
        assert self._distances is not None and self._next_hop is not None
        if not math.isfinite(float(self._distances[current_node])):
            return float(target[0]), float(target[1])
        node = current_node
        waypoint = nodes[node]
        travelled = float(np.linalg.norm(waypoint - current))
        visited = {node}
        while travelled < self.lookahead_m and node != target_node:
            successor = int(self._next_hop[node])
            if successor < 0 or successor in visited:
                break
            travelled += float(np.linalg.norm(nodes[successor] - nodes[node]))
            node = successor
            visited.add(node)
            waypoint = nodes[node]
        if node == target_node and target_snap <= self.terminal_distance_m:
            waypoint = target
        return float(waypoint[0]), float(waypoint[1])


__all__ = ["RouteGraphController"]
