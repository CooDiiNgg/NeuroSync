"""
Timing utilities for NeuroSync.
"""

import time
from functools import wraps
from typing import Callable, Optional
from contextlib import contextmanager

class Timer:
    """A simple timer class for measuring elapsed time."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        end_time = time.perf_counter()
        self.elapsed = end_time - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self) -> "Timer":
        return self.start()
    
    def __exit__(self, *args) -> None:
        self.stop()
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.4f} seconds")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(name=func.__name__).start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        print(f"[Timed] {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

@contextmanager
def timing_context(name: Optional[str] = None):
    """Context manager for timing a code block."""
    timer = Timer(name=name).start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        if name:
            print(f"[Timing Context] {name}: {elapsed:.4f} seconds")