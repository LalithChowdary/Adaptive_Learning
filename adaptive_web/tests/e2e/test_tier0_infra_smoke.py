"""Tier 0: Infrastructure smoke tests (must pass before and after M3)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

import harness  # noqa: E402


class TestTier0InfraSmoke(unittest.TestCase):
    def test_tier0_01_check_counts_gate_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(E2E_DIR / "check_counts.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=f"check_counts.py failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("OK: All E2E tier counts", proc.stdout)

    def test_tier0_02_unittest_discover_collects_at_least_172_with_zero_import_errors(self) -> None:
        loader = unittest.TestLoader()
        suite = loader.discover(str(E2E_DIR), top_level_dir=str(E2E_DIR))
        self.assertEqual(loader.errors, [], msg=f"Loader import errors: {loader.errors}")
        self.assertGreaterEqual(suite.countTestCases(), 172)

    def test_tier0_03_local_llm_stub_loopback_delay_error_and_clean_teardown(self) -> None:
        before_threads = {t.name for t in threading.enumerate()}
        with harness.LocalLLMStubServer(delay_seconds=0.05) as stub:
            req = urllib.request.Request(
                f"{stub.base_url}/v1/chat/completions",
                data=json.dumps({"model": "stub", "messages": [{"role": "user", "content": "curate learning_path"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer sk-test"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
                content = json.loads(payload["choices"][0]["message"]["content"])
                self.assertIn("steps", content)

            stub.status_code = 429
            req_err = urllib.request.Request(
                f"{stub.base_url}/v1/messages",
                data=json.dumps({"model": "stub", "messages": []}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_err, timeout=5.0)
            self.assertEqual(ctx.exception.code, 429)
            self.assertEqual(len(stub.received_requests), 2)

        after_threads = {t.name for t in threading.enumerate()}
        leaked = [n for n in (after_threads - before_threads) if n.startswith("aw-llm-stub-")]
        self.assertEqual(leaked, [], msg=f"Orphan stub threads leaked: {leaked}")

    def test_tier0_04_v22_catalog_resolver_all_5_formats_and_reference_hashes(self) -> None:
        catalog = harness.load_v22_catalog()
        self.assertEqual(len(catalog), 2160)
        for fmt in harness.TASK_FORMATS:
            task_id = f"range_boundaries.easy.{fmt}.00"
            valid_ans, invalid_ans = harness.solve_v22_task(task_id)
            self.assertNotEqual(valid_ans, invalid_ans)
            if fmt in ("output_prediction", "fill_in_the_blank", "multiple_select"):
                self.assertIsInstance(valid_ans, list)
                self.assertGreater(len(valid_ans), 0)
            else:
                self.assertIsInstance(valid_ans, str)
                self.assertGreater(len(valid_ans), 0)

        hashes_before = harness.snapshot_reference_hashes()
        hashes_after = harness.snapshot_reference_hashes()
        self.assertEqual(hashes_before, hashes_after)
        self.assertGreaterEqual(len(hashes_before), 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
