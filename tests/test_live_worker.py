import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest


MODULE_PATH = Path(__file__).parents[1] / "blend_mol" / "live_worker.py"
SPEC = importlib.util.spec_from_file_location("blend_mol_live_worker", MODULE_PATH)
live_worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live_worker
SPEC.loader.exec_module(live_worker)
LatestOnlyWorker = live_worker.LatestOnlyWorker
adapt_chunk_iterations = live_worker.adapt_chunk_iterations
should_continue_relaxation = live_worker.should_continue_relaxation


class LatestOnlyWorkerTests(unittest.TestCase):
    def test_adaptive_chunk_tracks_budget_with_smoothing_and_clamps(self):
        self.assertEqual(adapt_chunk_iterations(2, 100.0, 50.0, 1.0), 1)
        self.assertEqual(adapt_chunk_iterations(2, 25.0, 50.0, 1.0), 4)
        self.assertEqual(adapt_chunk_iterations(2, 25.0, 50.0, 0.5), 3)
        self.assertEqual(adapt_chunk_iterations(10, 1.0, 50.0), 10)
        self.assertEqual(adapt_chunk_iterations(0, 0.0, 50.0), 1)

    def test_interactive_relaxation_continues_only_while_progressing(self):
        self.assertTrue(should_continue_relaxation(1, "iteration limit", 2, 25))
        self.assertFalse(should_continue_relaxation(0, "gradient tolerance", 2, 25))
        self.assertFalse(should_continue_relaxation(1, "energy tolerance", 2, 25))
        self.assertFalse(should_continue_relaxation(1, "line search stalled", 2, 25))
        self.assertFalse(should_continue_relaxation(1, "iteration limit", 25, 25))

    def test_settling_chain_resubmits_until_converged(self):
        worker = LatestOnlyWorker(
            lambda remaining: (
                remaining - 1,
                1 if remaining > 1 else 0,
                "iteration limit" if remaining > 1 else "gradient tolerance",
            )
        )
        worker.submit(4)
        applied = 0
        while True:
            result = self._await_result(worker)
            remaining, steps, termination = result.value
            applied += 1
            if not should_continue_relaxation(steps, termination, applied - 1, 25):
                break
            worker.submit(remaining)
        worker.stop(wait=True)
        self.assertEqual(applied, 4)
        self.assertEqual(remaining, 0)

    def test_returns_completed_latest_generation(self):
        worker = LatestOnlyWorker(lambda value: value * 2)
        generation = worker.submit(4)
        result = self._await_result(worker)
        self.assertEqual(result.generation, generation)
        self.assertEqual(result.value, 8)
        self.assertIsNone(result.error)
        worker.stop(wait=True)

    def test_replaces_pending_and_discards_stale_completion(self):
        release = threading.Event()
        started = threading.Event()

        def calculate(value):
            if value == 1:
                started.set()
                release.wait(1.0)
            return value

        worker = LatestOnlyWorker(calculate)
        worker.submit(1)
        self.assertTrue(started.wait(1.0))
        worker.submit(2)
        latest_generation = worker.submit(3)
        release.set()
        result = self._await_result(worker)
        self.assertEqual(result.generation, latest_generation)
        self.assertEqual(result.value, 3)
        worker.stop(wait=True)

    def test_stop_invalidates_running_result(self):
        release = threading.Event()
        started = threading.Event()

        def calculate(value):
            started.set()
            release.wait(1.0)
            return value

        worker = LatestOnlyWorker(calculate)
        worker.submit(1)
        self.assertTrue(started.wait(1.0))
        worker.stop()
        release.set()
        worker.stop(wait=True)
        self.assertIsNone(worker.poll_latest())

    @staticmethod
    def _await_result(worker):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            result = worker.poll_latest()
            if result is not None:
                return result
            time.sleep(0.005)
        raise AssertionError("worker result timed out")


if __name__ == "__main__":
    unittest.main()
