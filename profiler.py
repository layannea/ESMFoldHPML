import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import torch


class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()
        self.timings: Dict[str, List[float]] = {}

    def reset(self):
        self.timings.clear

    @contextmanager
    def profile(self, name: str):
        if not self.enabled:
            yield
            return

        if self.use_cuda:
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000.0  # Convert to milliseconds

            # Store timing
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed_ms)
            print(name, elapsed_ms)

