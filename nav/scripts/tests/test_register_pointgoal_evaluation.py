"""Keep audited original results separate from independent GIF replays."""
import json
from pathlib import Path
import tempfile
import unittest

from nav.scripts.gallery.export_llm_gallery import gallery_html
from nav.scripts.gallery.register_pointgoal_evaluation import register_evaluation
from nav.scripts.tests.test_ppo_result_audit import make_report


class RegisterEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gallery = self.root / 'gallery'
        self.gallery.mkdir()
        self.summary = self.root / 'summary.json'
        self.summary.write_text(json.dumps(make_report(71)))
        (self.gallery / 'manifest.json').write_text('[]\n')
        (self.gallery / 'index.html').write_text('original page')

    def register(self, **overrides):
        args = dict(run_id='mixed400-original', name='Mixed400 original',
                    recording_note='Not recorded; metrics only')
        args.update(overrides)
        return register_evaluation(self.summary, self.gallery, **args)

    def test_preserves_original_bytes_and_has_no_fake_gifs(self):
        manifest_before = (self.gallery / 'manifest.json').read_bytes()
        self.register()
        self.assertEqual((self.gallery / 'manifest.json').read_bytes(), manifest_before)
        self.assertEqual((self.gallery / 'reports/mixed400-original.json').read_bytes(), self.summary.read_bytes())
        page = (self.gallery / 'index.html').read_text()
        self.assertNotIn('Recorded evaluations', page)
        registry = json.loads((self.gallery / 'evaluation_runs.json').read_text())
        self.assertEqual(registry[0]['successes'], 71)
        self.assertEqual(registry[0]['recording_note'], 'Not recorded; metrics only')
        self.assertFalse((self.gallery / 'gifs').exists())
        backups = list(self.gallery.glob('before-evaluation-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'index.html').read_text(), 'original page')

    def test_idempotent_and_conflicting_run_is_rejected(self):
        self.register()
        self.register()
        self.assertEqual(len(json.loads((self.gallery / 'evaluation_runs.json').read_text())), 1)
        self.assertEqual(len(list(self.gallery.glob('before-evaluation-*'))), 1)
        self.summary.write_text(json.dumps(make_report(72)))
        with self.assertRaisesRegex(ValueError, 'different run'):
            self.register()

    def test_replay_cannot_borrow_original_score(self):
        with self.assertRaisesRegex(ValueError, 'same 96 tasks'):
            self.register(model='visual-replay')
        self.assertFalse((self.gallery / 'evaluation_runs.json').exists())

    def test_reject_incomplete_report(self):
        report = make_report(71)
        report['episodes'].pop()
        self.summary.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            self.register()

    def test_safe_archive_id_and_metadata_stays_off_page(self):
        self.register(name='<b>original</b>')
        self.assertNotIn('<b>original</b>', (self.gallery / 'index.html').read_text())
        with self.assertRaises(ValueError):
            self.register(run_id='../unsafe')

    def test_empty_gallery_unchanged_and_model_deep_link_supported(self):
        page = gallery_html([], 'Test')
        self.assertNotIn('<h2>Recorded evaluations</h2>', page)
        self.assertIn("new URLSearchParams(location.search).get('model')", page)
        self.assertIn('toFixed(2)', page)


if __name__ == '__main__':
    unittest.main()
