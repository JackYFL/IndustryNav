"""Regression tests for the reusable package-layer contracts."""

from __future__ import annotations

import unittest

from nav.baselines.bc.agent import BCNavController
from nav.baselines.bc.trainer import BCTrainer
from nav.baselines.rl.agent import PPOPointGoalController
from nav.baselines.rl.trainer import PPOTrainer
from nav.core.agent import NavigationAgent
from nav.envs.base import NavigationEnvironment
from nav.envs.unity_pointgoal import UnityPointGoalEnv
from nav.eval.base import BinaryRateMetric
from nav.eval.warning import WarningDetector as EvalWarningDetector
from nav.safety import CollisionDetector, WarningDetector
from nav.train.base import BaseTrainer


class ArchitectureLayerTest(unittest.TestCase):
    def test_concrete_agents_share_the_navigation_contract(self) -> None:
        self.assertTrue(issubclass(BCNavController, NavigationAgent))
        self.assertTrue(issubclass(PPOPointGoalController, NavigationAgent))

    def test_concrete_trainers_share_the_trainer_lifecycle(self) -> None:
        self.assertTrue(issubclass(BCTrainer, BaseTrainer))
        self.assertTrue(issubclass(PPOTrainer, BaseTrainer))

    def test_unity_adapter_uses_the_environment_contract(self) -> None:
        self.assertTrue(issubclass(UnityPointGoalEnv, NavigationEnvironment))

    def test_eval_reuses_the_online_warning_detector(self) -> None:
        self.assertIs(EvalWarningDetector, WarningDetector)

    def test_collision_and_rate_primitives_compose(self) -> None:
        detector = CollisionDetector()
        metric = BinaryRateMetric()
        metric.update(detector.detect(10.0, (0.0, 0.0), (0.2, 0.0)).triggered)
        metric.update(detector.detect(10.0, (0.0, 0.0), (1.0, 0.0)).triggered)
        result = metric.compute()
        self.assertEqual((result.total, result.triggered, result.rate), (2, 1, 0.5))


if __name__ == "__main__":
    unittest.main()
