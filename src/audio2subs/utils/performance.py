import logging
import time
from contextlib import ContextDecorator
from typing import Optional


class Timer(ContextDecorator):
    """A flexible timer for performance benchmarking.
    
    Can be used as a context manager or a decorator.
    
    Example:
        with Timer("Heavy Task", logger, duration=60.0):
            ...
            
        @Timer("Another Task", logger)
            def do_something():
                ...
    """

    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
        duration: Optional[float] = None,
        level: int = logging.INFO,
    ):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.duration = duration
        self.level = level
        self.start_perf = 0.0

    def __enter__(self):
        self.start_perf = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_perf
        
        msg = f"[{self.name}] took {elapsed:.2f}s"
        
        if self.duration and self.duration > 0:
            rtf = elapsed / self.duration
            speed = 1.0 / rtf if rtf > 0 else float('inf')
            msg += f" (RTF: {rtf:.3f}x | Speed: {speed:.1f}x)"
            
        self.logger.log(self.level, msg)
