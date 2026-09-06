"""Synthetic offline tests: no actual simulator or model calls."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import shlex
import tempfile
import unittest

import numpy as np
from PIL import Image

from nav.config import ACTIONS_CSV_FIELDS
from nav.harness.navigation_protocol import check_navigation_run_config, navigation_run_config
from nav.scripts.reconstruct_gemini_runs import (
    ACTION_SPACE, create_bundle, legacy_template, read_rows, reconstruct, snapshot,
)
from nav.scripts.smoke_reconstructed_gemini import check_bundle, smoke


class SyntheticUnity:
    def __init__(self, args, wrong_pose=False):
        self.pose = [args.init_world_x + bool(wrong_pose), args._resume_world_y, args.init_world_z,
                     0., args.init_curr_direction, 0.]
        self.closed = False
        self.action_count = 0
        self.signal = [0., 0., 0.]

    def get_steps(self, _):
        rgb = np.broadcast_to(np.linspace(0, 1, 512)[None, None, None, :], (1, 3, 512, 512))
        return argparse.Namespace(obs=[rgb, None, None, np.array([self.pose], dtype=np.float32)]), None

    def set_actions(self, _, action):
        self.signal = action.continuous[0]
        self.action_count += 1

    def step(self):
        move, _, look = self.signal
        self.pose[2] += float(move) * .05 * math.cos(math.radians(self.pose[4]))
        self.pose[0] += float(move) * .05 * math.sin(math.radians(self.pose[4]))
        self.pose[4] = (self.pose[4] + float(look)) % 360

    def close(self):
        self.closed = True


class ReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'outputs/scene1/point1/gemini-3.8-flash/seed0'
        self.source.mkdir(parents=True)
        self.template = legacy_template()
        client = self.root / 'test-client'
        client.touch()
        options = dict(baseline='llm', model_id='google/gemini-3.8-flash', file_name=str(client),
                       scene_id='0', scene_name='scene1', point_id='point1', seed_id='0',
                       vision_input='true', history_size='5', max_steps='70', reach_m='2.0',
                       ego_width='512', ego_height='512', minimap_width='862', minimap_height='512',
                       init_world_x='0', init_world_z='5', init_curr_direction='180',
                       target_x='200', target_y='200', dynamic_objects='moving')
        command = ['python', '-m', 'nav.scripts.run_benchmark_cell']
        for key, value in options.items():
            command += [f'--{key}', value]
        (self.source / 'run.log').write_text('# command: ' + shlex.join(command) + '\n'
            'Engine config: quality_level=3, screen=1724x1024\n'
            'human_speed=1.200m/s vehicle_speed=2.500m/s robot_speed=1.500m/s lighting=disabled\n')
        rows, qa = [], ''
        history = 'No previous movements yet.'
        (self.source / 'llm_fp').mkdir()
        for step, action, z in [(1, 'forward', 5.), (2, 'turn right', 3.5)]:
            row = dict(step=step, action=action, move=15 if action == 'forward' else 0,
                       strafe=0, look=22.5 if action == 'turn right' else 0,
                       init_px=100, init_py=100, init_world_x=0, init_world_z=5, init_direction=180,
                       curr_px=100, curr_py=100, curr_world_x=0, curr_world_y=.08, curr_world_z=z,
                       curr_direction_x=0, curr_direction_y=180, curr_direction_z=0,
                       target_px=200, target_py=200, target_world_x=10, target_world_z=-10,
                       marker_source='vector', distance_world=math.hypot(10, z+10))
            values = dict(curr_world_x='0.00', curr_world_z=f'{z:.2f}', target_world_x='10.00',
                          target_world_z='-10.00', curr_x='100', curr_y='100', target_x='200', target_y='200',
                          distance_m=f"{row['distance_world']:.2f}", theta='180.0', dynamic_objects='moving',
                          allowed_actions=str(list(ACTION_SPACE)), history=history, reach_m='2.00')
            prompt = self.template.format(**values).strip()
            qa += f'=== Step {step} ===\n[Q]\n{prompt}\n\n[A]\nAction: {action}\nReasoning: Clear floor.\n' + '='*60 + '\n\n'
            history = f"Step {step}: World X/Z (0.00, {z:.2f}) m, θ=180.0°, Action: '{action}', Distance to target: {values['distance_m']} m"
            rows.append(row)
            Image.new('RGB', (512,512), 'blue').save(self.source/'llm_fp'/f'{step-1}.png')
        with (self.source/'llm_actions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=ACTIONS_CSV_FIELDS)
            writer.writeheader();writer.writerows(rows)
        (self.source/'agent_qa.txt').write_text(qa)
        self.bundle=self.root/'rebuilt/scene1/point1/gemini-3.8-flash/seed0'

    def test_reconstruction_preserves_sources_and_unknowns(self):
        before=snapshot(self.source)
        report=create_bundle(self.source,self.bundle,self.template)
        config,state=check_bundle(self.bundle)
        self.assertEqual(snapshot(self.source),before)
        self.assertEqual(report['committed_steps'],1)
        self.assertEqual(state['pending_decision']['action'],'turn right')
        self.assertEqual(state['pending_decision']['observation'],'')
        self.assertEqual(state['history'][0]['step'],1)
        self.assertEqual(state['step_budget'],70)
        self.assertIsNone(state['historical_api_calls'])
        self.assertIsNone(config['settings']['max_tokens'])
        self.assertFalse(config['execution_policy']['api_resume_ready'])
        with self.assertRaises(ValueError):
            check_navigation_run_config(self.bundle,navigation_run_config(argparse.Namespace(),self.template))

    def test_pose_conflict_refuses_before_creating_output(self):
        rows=read_rows(self.source/'llm_actions.csv');rows[-1]['curr_world_z']='4.5'
        with (self.source/'llm_actions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=ACTIONS_CSV_FIELDS);writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'pose conflict'):
            create_bundle(self.source,self.bundle,self.template)
        self.assertFalse(self.bundle.exists())

    def test_prompt_and_action_mismatch_fail_closed(self):
        qa=self.source/'agent_qa.txt';original=qa.read_text()
        qa.write_text(original.replace('warehouse navigation agent','different agent'))
        with self.assertRaisesRegex(ValueError,'prompt differs'):
            reconstruct(self.source,self.template)
        qa.write_text(original.replace('Action: turn right','Action: forward'))
        with self.assertRaisesRegex(ValueError,'answer'):
            reconstruct(self.source,self.template)

    def test_existing_destination_and_modified_copy_are_protected(self):
        create_bundle(self.source,self.bundle,self.template)
        with self.assertRaisesRegex(ValueError,'new and separate'):
            create_bundle(self.source,self.bundle,self.template)
        (self.bundle/'llm_fp/1.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'modified'):
            check_bundle(self.bundle)

    def test_storage_only_smoke_has_no_simulator(self):
        create_bundle(self.source,self.bundle,self.template)
        def forbidden(*args):raise AssertionError('Simulator must not launch')
        report=smoke(self.bundle,self.root/'storage_smoke',setup=forbidden)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['model_calls'],0)
        self.assertEqual(report['backend'],'storage_only')

    def test_cached_smoke_commits_exactly_one_saved_action(self):
        create_bundle(self.source,self.bundle,self.template)
        environments=[]
        def setup(args,logger):
            env=SyntheticUnity(args);environments.append(env)
            return argparse.Namespace(env=env,target_world=(10.,-10.))
        out=self.root/'fake_smoke'
        report=smoke(self.bundle,out,unity=True,setup=setup,base_port=5507,backend_name='synthetic_unity')
        self.assertEqual(report['committed_steps_after'],2)
        self.assertEqual(report['post_action_pose']['yaw'],225.)
        self.assertEqual(report['model_calls'],0)
        self.assertEqual(len(read_rows(out/'llm_actions.csv')),2)
        self.assertEqual((out/'agent_qa.txt').read_bytes(),(self.source/'agent_qa.txt').read_bytes())
        self.assertEqual(json.loads((out/'checkpoint.json').read_text())['phase'],'ready')
        self.assertEqual(environments[0].action_count,1)
        self.assertTrue(environments[0].closed)

    def test_wrong_unity_pose_fails_before_action(self):
        create_bundle(self.source,self.bundle,self.template)
        environments=[]
        def setup(args,logger):
            env=SyntheticUnity(args,wrong_pose=True);environments.append(env)
            return argparse.Namespace(env=env,target_world=(10.,-10.))
        out=self.root/'wrong_pose'
        with self.assertRaisesRegex(ValueError,'did not restore'):
            smoke(self.bundle,out,unity=True,setup=setup,base_port=5507,backend_name='synthetic_unity')
        self.assertEqual(environments[0].action_count,0)
        self.assertTrue(environments[0].closed)
        self.assertEqual(json.loads((out/'smoke_report.json').read_text())['status'],'failed')


if __name__=='__main__':
    unittest.main()
