"""Wall-clock timing helpers for mkCELL pipeline stages."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_timings: Dict[str, float] = {}
_stack: list[tuple[str, float]] = []


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = time.perf_counter()
    _stack.append((name, start))
    try:
        yield
    finally:
        _, t0 = _stack.pop()
        elapsed = time.perf_counter() - t0
        _timings[name] = _timings.get(name, 0.0) + elapsed
        print(f"[timing] {name}={elapsed:.2f}s", flush=True)


def reset() -> None:
    _timings.clear()
    _stack.clear()


def record(name: str, elapsed: float) -> None:
    """Record an externally measured top-level phase."""
    elapsed = float(elapsed)
    _timings[name] = _timings.get(name, 0.0) + elapsed
    print(f"[timing] {name}={elapsed:.2f}s", flush=True)


def summary(label: Optional[str] = None) -> None:
    if not _timings:
        return
    total = sum(_timings.values())
    header = label or "timing summary"
    print(f"[timing] === {header} (total={total:.2f}s) ===", flush=True)
    for name, elapsed in sorted(_timings.items(), key=lambda kv: -kv[1]):
        print(f"[timing]   {name}: {elapsed:.2f}s", flush=True)
