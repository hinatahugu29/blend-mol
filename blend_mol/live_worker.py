"""Latest-only background calculation worker for interactive UFF.

The worker deliberately knows nothing about Blender.  Callers submit immutable
molecule snapshots and consume completed results on Blender's main thread.
While a calculation is running, repeated drag updates replace the pending job
instead of building an increasingly stale queue.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Condition, Thread
from time import perf_counter
from typing import Callable, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def adapt_chunk_iterations(
    current: int, measured_ms: float, target_ms: float, smoothing: float = 0.7,
    minimum: int = 1, maximum: int = 10,
) -> int:
    """Fit the next interactive optimization chunk to a wall-clock budget.

    The multiplicative update mirrors Avogadro's interactive optimizer: equal
    speedups and slowdowns have symmetric effects in log space, while smoothing
    prevents one noisy measurement from making the UI cadence oscillate.
    """

    minimum = max(1, minimum)
    maximum = max(minimum, maximum)
    fallback = min(maximum, max(minimum, current))
    if current <= 0 or measured_ms <= 0.0 or target_ms <= 0.0 or smoothing <= 0.0:
        return fallback
    scale = (target_ms / measured_ms) ** min(smoothing, 1.0)
    proposed = current * scale
    return min(maximum, max(minimum, math.floor(proposed + 0.5)))


def should_continue_relaxation(
    accepted_steps: int, termination: str, completed_updates: int, maximum_updates: int,
) -> bool:
    """Whether an interactive result should immediately schedule another update."""
    return (
        accepted_steps > 0
        and termination == "iteration limit"
        and completed_updates < maximum_updates
    )


@dataclass(frozen=True)
class WorkerResult(Generic[OutputT]):
    generation: int
    value: OutputT | None
    elapsed_ms: float
    error: BaseException | None = None


class LatestOnlyWorker(Generic[InputT, OutputT]):
    """Run one job at a time and retain at most the newest pending request."""

    def __init__(self, calculate: Callable[[InputT], OutputT], name: str = "BlendMolLiveUFF"):
        self._calculate = calculate
        self._condition = Condition()
        self._pending: tuple[int, InputT] | None = None
        self._completed: WorkerResult[OutputT] | None = None
        self._generation = 0
        self._stopped = False
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def submit(self, payload: InputT) -> int:
        with self._condition:
            if self._stopped:
                raise RuntimeError("background worker has stopped")
            self._generation += 1
            self._pending = (self._generation, payload)
            self._condition.notify()
            return self._generation

    def poll_latest(self) -> WorkerResult[OutputT] | None:
        """Return only a completion matching the newest submitted generation."""
        with self._condition:
            completed = self._completed
            self._completed = None
            if completed is None or completed.generation != self._generation:
                return None
            return completed

    def stop(self, wait: bool = False) -> None:
        """Invalidate queued/running work; a running native call may finish silently."""
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._generation += 1
            self._pending = None
            self._completed = None
            self._condition.notify_all()
        if wait:
            self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                generation, payload = self._pending
                self._pending = None
            started = perf_counter()
            try:
                value = self._calculate(payload)
                completion = WorkerResult(generation, value, (perf_counter() - started) * 1000.0)
            except BaseException as exc:
                completion = WorkerResult(
                    generation, None, (perf_counter() - started) * 1000.0, exc
                )
            with self._condition:
                if not self._stopped and generation == self._generation:
                    self._completed = completion
