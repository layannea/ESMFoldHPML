import time
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import torch


class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()

        # Event groupings for organized reporting
        self.relevant_events = {
            "INPUT_EMBEDDING": [
                "input_prep",
                "esm2_encoding",
                "esm_preprocessing",
                "pairwise_positional_embedding",
            ],
            "BIAS_AND_MLP": [
                "bias_pair_to_sequence",
                "seq_layernorm",
                "seq_mlp",
                "sequence_to_pair",
                "pair_mlp",
            ],
            "TRIANGULAR_MULTIPLICATION": [
                "tri_mul_outgoing",
                "tri_mul_incoming",
            ],
            "TRIANGULAR_ATTENTION": [
                "seq_attention",
                "seq_attention_dropout",
                "tri_att_starting",
                "tri_att_ending",
            ],
            "STRUCTURE_MODULE": [
                "recycle_norm_s",
                "recycle_norm_z",
                "recycle_disto_embed",
                "trunk2sm_projections",
                "structure_module",
                "distogram_calc",
                "prediction_heads",
            ],
        }

        ## TIMING
        self.timings: Dict[str, List[float]] = {}
        self._cuda_events: Dict[str, List[tuple]] = {}  # (start_event, end_event) pairs

        ## MEMORY
        self.memory_usage: Dict[str, List[Dict[str, float]]] = {}

        ## COMPUTED METRICS
        self._section_metrics: Dict[str, Dict[str, Any]] = {}
        self._group_metrics: Dict[str, Dict[str, Any]] = {}

        ## PROFILER
        self._torch_profiler: Optional[torch.profiler.profile] = None
        self._profiler_active: bool = False

        if self.enabled:
            self._setup_torch_profiler()

    def __del__(self):
        """Clean up torch profiler on deletion."""
        if hasattr(self, '_torch_profiler') and self._torch_profiler is not None:
            if hasattr(self, '_profiler_active') and self._profiler_active:
                try:
                    self._torch_profiler.__exit__(None, None, None)
                except:
                    pass

    def _setup_torch_profiler(self):
        """Initialize torch profiler with minimal overhead configuration."""
        if not self.enabled:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._torch_profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            with_modules=False
        )

    def reset(self):
        """Reset profiler state for new forward pass."""
        self.timings.clear()
        self.memory_usage.clear()
        self._cuda_events.clear()
        self._section_metrics.clear()
        self._group_metrics.clear()

        if self.enabled and self._torch_profiler is not None:
            # Stop existing profiler if running
            if self._profiler_active:
                try:
                    self._torch_profiler.__exit__(None, None, None)
                except:
                    pass
                self._profiler_active = False

            # Re-initialize for new forward pass
            self._setup_torch_profiler()
            self._torch_profiler.__enter__()
            self._profiler_active = True

    @contextmanager
    def profile(self, name: str):
        if not self.enabled:
            yield
            return

        ## Startign memory
        if self.use_cuda:
            mem_start = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            mem_start = 0.0
            start_time = time.perf_counter()

        ## Profiling
        with torch.profiler.record_function(name):
            try:
                yield
            finally:
                end_event.record()
                mem_end = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

                ## Store events
                if name not in self._cuda_events:
                    self._cuda_events[name] = []
                self._cuda_events[name].append((start_event, end_event))

                if name not in self.memory_usage:
                    self.memory_usage[name] = []
                self.memory_usage[name].append({
                    "start_mb": mem_start,
                    "end_mb": mem_end,
                    "delta_mb": mem_end - mem_start
                })

    def _results(self):
        """Process all profiling data and compute metrics."""
        if not self.enabled or self._torch_profiler is None:
            return

        ## synchronize
        if self.use_cuda and self._cuda_events:
            torch.cuda.synchronize()

            for name, event_pairs in self._cuda_events.items():
                if name not in self.timings:
                    self.timings[name] = []

                for start_event, end_event in event_pairs:
                    elapsed_ms = start_event.elapsed_time(end_event)
                    self.timings[name].append(elapsed_ms)

            self._cuda_events.clear()

        ## stop profiler
        if self._profiler_active:
            try:
                self._torch_profiler.__exit__(None, None, None)
            except:
                pass
            self._profiler_active = False

        ## return events
        try:
            events = self._torch_profiler.events()
        except:
            events = []

        ## compute metrics
        for section_name in self.timings.keys():
            section_events = [e for e in events if e.name == section_name]

            cpu_time_us = sum(e.cpu_time_total for e in section_events if hasattr(e, 'cpu_time_total'))
            cuda_time_us = 0

            for e in section_events:
                if hasattr(e, 'device_time_total') and e.device_time_total is not None:
                    cuda_time_us += e.device_time_total
                elif hasattr(e, 'cuda_time_total') and e.cuda_time_total is not None:
                    cuda_time_us += e.cuda_time_total

            mem_allocated_mb = 0.0
            mem_delta_mb = 0.0
            if section_name in self.memory_usage and self.memory_usage[section_name]:
                mem_data = self.memory_usage[section_name]
                mem_allocated_mb = np.mean([m["end_mb"] for m in mem_data])
                mem_delta_mb = np.sum([m["delta_mb"] for m in mem_data])

            self._section_metrics[section_name] = {
                "cpu_time_ms": cpu_time_us / 1000.0,
                "cuda_time_ms": cuda_time_us / 1000.0,
                "total_time_ms": sum(self.timings[section_name]),
                "memory_mb": mem_allocated_mb,
                "memory_delta_mb": mem_delta_mb,
            }

        ## group metrics
        for group_name, event_names in self.relevant_events.items():
            group_cpu = 0.0
            group_cuda = 0.0
            group_total = 0.0
            group_memory = 0.0
            group_memory_delta = 0.0

            for event_name in event_names:
                if event_name in self._section_metrics:
                    metrics = self._section_metrics[event_name]
                    group_cpu += metrics["cpu_time_ms"]
                    group_cuda += metrics["cuda_time_ms"]
                    group_total += metrics["total_time_ms"]
                    group_memory = max(group_memory, metrics["memory_mb"])
                    group_memory_delta += metrics["memory_delta_mb"]

            self._group_metrics[group_name] = {
                "cpu_time_ms": group_cpu,
                "cuda_time_ms": group_cuda,
                "total_time_ms": group_total,
                "memory_mb": group_memory,
                "memory_delta_mb": group_memory_delta,
                "sections": event_names,
            }

    def summary(self) -> Dict[str, Any]:
        """Return detailed profiling metrics."""
        if not self.enabled:
            return {"sections": {}, "groups": {}}

        if not self._section_metrics:
            self._results()

        return {
            "sections": dict(self._section_metrics),
            "groups": dict(self._group_metrics),
        }

    def percents(self) -> List[tuple]:
        """Return percentage breakdown by group."""
        if "total_forward" not in self.timings:
            return []

        forwardpass_total = float(np.sum(self.timings["total_forward"]))
        if forwardpass_total == 0:
            return []

        result = []
        for group_name, events in self.relevant_events.items():
            group_total = sum(
                float(np.sum(self.timings[event]))
                for event in events
                if event in self.timings
            )
            percent = 100.0 * group_total / forwardpass_total
            result.append((group_name, percent))

        return result

    def __str__(self) -> str:
        """Return formatted profiling report."""
        if not self.timings or "total_forward" not in self.timings:
            return "Profiler: No data collected"

        # Ensure metrics are computed
        if not self._section_metrics:
            self._results()

        max_key_len = max(len(k) for k in self.timings)
        forwardpass_total = float(np.sum(self.timings["total_forward"]))

        result = []
        for group_name, events in self.relevant_events.items():
            result.append(f"\n{group_name}")
            result.append("=" * 60)

            group_total = 0.0
            group_memory = 0.0

            for event in events:
                if event in self.timings:
                    total = float(np.sum(self.timings[event]))
                    group_total += total

                    # Get memory info if available
                    memory_str = ""
                    if event in self._section_metrics:
                        mem_mb = self._section_metrics[event]["memory_mb"]
                        if mem_mb > 0:
                            memory_str = f"  [{mem_mb:>8.1f} MB]"
                            group_memory = max(group_memory, mem_mb)

                    result.append(f"{event:<{max_key_len}}  {total:>10.4f} ms{memory_str}")

            percent = 100.0 * group_total / forwardpass_total if forwardpass_total > 0 else 0.0
            result.append("-" * 60)

            memory_str = f"  [{group_memory:>8.1f} MB]" if group_memory > 0 else ""
            result.append(f"{'Total':<{max_key_len}}  {group_total:>10.4f} ms  ({percent:>5.2f}%){memory_str}")

        return "\n".join(result)
