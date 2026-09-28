"""Qwen-specific recovery checks; synthetic data, no API or Unity calls."""

import argparse
import csv
from functools import partial
import json
from pathlib import Path
import sqlite3
import unittest

from nav.harness.checkpoint import atomic_json
from nav.scripts.agent.reconstruct_gemini_runs import create_bundle, snapshot
from nav.scripts.agent.reconstruct_qwen_runs import (
    MODEL, checkpoint_store, job_evidence, read_counter, reconstruct_qwen,
)
from nav.scripts.agent.smoke_reconstructed_gemini import check_bundle, smoke
from nav.scripts.tests import test_reconstruct_gemini_runs as fixtures


class QwenReconstructionTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ReconstructionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.template = fixture.root, fixture.template
        self.source = fixture.source.parents[1] / 'qwen3.8-flash/seed0'
        self.source.parent.mkdir()
        fixture.source.rename(self.source)
        self.bundle = self.root / 'qwen_rebuilt/scene1/point1/qwen3.8-flash/seed0'
        log = self.source / 'run.log'
        log.write_text(log.read_text().replace('google/gemini-3.8-flash', MODEL).replace(
            '--max_steps 70', '--max_steps 70 --dynamic_step_budget --step_budget_min 50 '
            '--step_budget_max 160 --steps_per_path_meter 1.25 --step_budget_overhead 20 '
            '--llm_provider openrouter --max_tokens 500 --llm_min_request_interval_sec 0.0'))
        summary = dict(stop_reason='decision_error', max_steps=50, initial_step_budget=50,
                       step_budget_mode='dynamic', sim_steps_per_decision=2)
        with (self.source / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        self.job = self.root / 'qwen_job'
        self.job.mkdir()
        counter = self.job / 'requests.sqlite3'
        with sqlite3.connect(counter) as db:
            db.execute('CREATE TABLE request_counter (id INTEGER PRIMARY KEY, value INTEGER)')
            db.execute('INSERT INTO request_counter VALUES (1,5066)')
        atomic_json(self.job / 'status.json', dict(model=MODEL, stage='finished_with_errors', request_limit=7151,
            seed=0, history_size=5, max_tokens=500, reasoning_enabled=False, json_mode=True,
            request_interval_sec=5, request_attempts=3, tasks_with_results=96, requests_reserved=5066,
            counter_file=str(counter), invalid_tasks=[dict(scene='scene1', point='point1')]))
        self.evidence = job_evidence(self.job)

    def create(self):
        return create_bundle(self.source, self.bundle, self.template,
            reconstructor=partial(reconstruct_qwen, evidence=self.evidence), store_factory=checkpoint_store)

    def test_preserves_dynamic_budget_counter_and_original_files(self):
        before = snapshot(self.source)
        report = self.create()
        config, state = check_bundle(self.bundle)
        self.assertEqual(snapshot(self.source), before)
        self.assertEqual(report['total_budget'], 50)
        self.assertEqual(state['step_budget_mode'], 'dynamic')
        self.assertEqual(state['step_count'], 1)
        self.assertEqual(state['request_budget']['requests_reserved'], 5066)
        self.assertEqual(state['request_budget']['limit'], 7151)
        self.assertEqual(config['settings']['max_tokens'], 500)
        self.assertFalse(config['execution_policy']['api_resume_ready'])

    def test_no_model_smoke_keeps_dynamic_allowance_and_counter(self):
        self.create()
        def setup(args, logger):
            return argparse.Namespace(env=fixtures.SyntheticUnity(args), target_world=(10., -10.))
        report = smoke(self.bundle, self.root / 'smoke', unity=True, setup=setup,
                       base_port=5507, backend_name='synthetic_unity')
        self.assertEqual(report['original_step_budget'], 50)
        self.assertEqual(report['committed_steps_after'], 2)
        self.assertEqual(report['model_calls'], 0)
        self.assertEqual(read_counter(self.evidence['budget_binding'])['requests_reserved'], 5066)

    def test_missing_or_reset_counter_blocks_without_reinitializing(self):
        self.create()
        path = Path(self.evidence['budget_binding']['counter_file'])
        with sqlite3.connect(path) as db:
            db.execute('UPDATE request_counter SET value=0 WHERE id=1')
        with self.assertRaisesRegex(ValueError, 'reset'):
            check_bundle(self.bundle)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_bundle(self.bundle)
        self.assertFalse(path.exists())

    def test_empty_pending_reasoning_is_not_fabricated(self):
        path = self.source / 'agent_qa.txt'
        text = path.read_text()
        index = text.rfind('Reasoning: Clear floor.')
        path.write_text(text[:index] + text[index:].replace('Reasoning: Clear floor.', 'Reasoning: ', 1))
        with self.assertRaisesRegex(ValueError, 'empty reasoning'):
            self.create()
        self.assertFalse(self.bundle.exists())

    def test_summary_budget_mismatch_is_rejected(self):
        path = self.source / 'results.csv'
        path.write_text(path.read_text().replace(',50,50,', ',70,70,'))
        with self.assertRaisesRegex(ValueError, 'dynamic budget'):
            self.create()

    def test_job_model_mismatch_is_rejected(self):
        path = self.job / 'status.json'
        status = json.loads(path.read_text())
        status['model'] = 'another/model'
        atomic_json(path, status)
        with self.assertRaisesRegex(ValueError, 'job metadata'):
            job_evidence(self.job)


if __name__ == '__main__':
    unittest.main()
