"""Task ordering helpers for on-policy navigation rollouts."""

from __future__ import annotations

import random
from typing import Iterable


def round_robin_scene_tasks(tasks: Iterable[dict], limit: int = 0) -> list[dict]:
    """Interleave scenes while retaining deterministic per-scene task order."""
    by_scene: dict[str, list[dict]] = {}
    for task in tasks:
        by_scene.setdefault(str(task["scene_name"]), []).append(task)
    ordered: list[dict] = []
    depth = 0
    while True:
        added = False
        for scene in sorted(
            by_scene, key=lambda name: int(name.removeprefix("scene"))
        ):
            if depth < len(by_scene[scene]):
                ordered.append(by_scene[scene][depth])
                added = True
                if limit > 0 and len(ordered) >= limit:
                    return ordered
        if not added:
            return ordered
        depth += 1


class RandomSceneBlockTaskSampler:
    """Sample shuffled point pairs while amortizing Unity scene relaunches.

    Tasks are drawn without replacement from a shuffled bag for the current
    scene.  The scene changes after a small episode block and scene order is
    reshuffled without immediate repeats.  This mixes short and long pairs
    throughout training without relaunching Unity after every episode.
    """

    def __init__(
        self,
        tasks: Iterable[dict],
        *,
        seed: int,
        scene_block_episodes: int = 4,
    ) -> None:
        if scene_block_episodes <= 0:
            raise ValueError("scene_block_episodes must be positive")
        self._by_scene: dict[str, list[dict]] = {}
        for task in tasks:
            self._by_scene.setdefault(str(task["scene_name"]), []).append(task)
        if not self._by_scene:
            raise ValueError("RandomSceneBlockTaskSampler requires tasks")
        self._rng = random.Random(int(seed))
        self._scene_block_episodes = int(scene_block_episodes)
        self._scene_bag: list[str] = []
        self._task_bags: dict[str, list[dict]] = {}
        self._last_task_id: dict[str, str] = {}
        self._current_scene: str | None = None
        self._episodes_in_block = 0

    @staticmethod
    def _task_id(task: dict) -> str:
        return str(task.get("episode_id", id(task)))

    def _refill_scene_bag(self) -> None:
        self._scene_bag = sorted(self._by_scene)
        self._rng.shuffle(self._scene_bag)
        if (
            self._current_scene is not None
            and len(self._scene_bag) > 1
            and self._scene_bag[-1] == self._current_scene
        ):
            self._scene_bag[0], self._scene_bag[-1] = (
                self._scene_bag[-1],
                self._scene_bag[0],
            )

    def _next_scene(self) -> str:
        if not self._scene_bag:
            self._refill_scene_bag()
        return self._scene_bag.pop()

    def _refill_task_bag(self, scene: str) -> None:
        bag = list(self._by_scene[scene])
        self._rng.shuffle(bag)
        previous = self._last_task_id.get(scene)
        if (
            previous is not None
            and len(bag) > 1
            and self._task_id(bag[-1]) == previous
        ):
            bag[0], bag[-1] = bag[-1], bag[0]
        self._task_bags[scene] = bag

    def next_task(self) -> dict:
        if (
            self._current_scene is None
            or self._episodes_in_block >= self._scene_block_episodes
        ):
            self._current_scene = self._next_scene()
            self._episodes_in_block = 0
        scene = self._current_scene
        bag = self._task_bags.get(scene)
        if not bag:
            self._refill_task_bag(scene)
            bag = self._task_bags[scene]
        task = bag.pop()
        self._last_task_id[scene] = self._task_id(task)
        self._episodes_in_block += 1
        return task

    def skip_current_scene(self) -> None:
        """Abandon the active scene after a Unity initialization failure.

        A malformed render or an unsupported scene must not strand every DDP
        rank at the next collective.  The next :meth:`next_task` call advances
        to another scene while retaining this scene's remaining task bag for a
        later visit.
        """
        if self._current_scene is not None:
            self._episodes_in_block = self._scene_block_episodes


__all__ = ["RandomSceneBlockTaskSampler", "round_robin_scene_tasks"]
