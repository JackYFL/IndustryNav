"""Offline tests for timeout cleanup of a grid cell's owned process group."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from nav.scripts.agent.run_benchmark_grid import run_cell_subprocess


class GridSubprocessTest(unittest.TestCase):
    def test_normal_exit_returns_completed_process(self):
        with patch("nav.scripts.agent.run_benchmark_grid.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            result = run_cell_subprocess(["test"], timeout=10, cwd="/tmp")
        self.assertEqual(result.returncode, 0)
        popen.assert_called_once_with(["test"], start_new_session=(os.name == "posix"), cwd="/tmp")
        popen.return_value.wait.assert_called_once_with(timeout=10)

    @unittest.skipUnless(os.name == "posix", "POSIX owned process groups")
    def test_timeout_kills_descendants_even_if_wrapper_exits_on_term(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired(["test"], 10), 0, 0]
        with patch("nav.scripts.agent.run_benchmark_grid.subprocess.Popen", return_value=process), \
                patch("nav.scripts.agent.run_benchmark_grid.owned_cell_process_groups", return_value={12345}), \
                patch("nav.scripts.agent.run_benchmark_grid.os.killpg") as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_cell_subprocess(["test"], timeout=10)
        self.assertEqual(killpg.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (12345, signal.SIGKILL))

    @unittest.skipUnless(os.name == "posix", "POSIX owned process groups")
    def test_real_timeout_leaves_no_running_cell_or_grandchild(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "owned_pids.json"
            code = (
                "import json,os,subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                "open(sys.argv[1],'w').write(json.dumps([os.getpid(),child.pid])); time.sleep(30)"
            )
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_cell_subprocess([sys.executable, "-c", code, str(marker)], timeout=1)
                pids = json.loads(marker.read_text())
                for pid in pids:
                    check = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                        capture_output=True, text=True, check=False)
                    # An adopted zombie has exited; it can no longer use API,
                    # mutate output, or hold the per-run lock.
                    self.assertTrue(not check.stdout.strip() or check.stdout.strip().startswith("Z"),
                        f"Owned process {pid} still running: {check.stdout}")
            finally:
                if marker.exists():
                    try:
                        os.killpg(json.loads(marker.read_text())[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    for pid in json.loads(marker.read_text())[1:]:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
