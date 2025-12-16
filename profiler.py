import time
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import torch


class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()

        self.relevant_events = {
            "INPUT_EMBEDDING" : [
                "esm2_encoding",
                "esm_preprocessing",
            ],
            
            "BIAS_AND_MLP" : [
                "bias_pair_to_sequence",
                "seq_layernorm",
                "seq_mlp",
                "sequence_to_pair",
                "pair_mlp",
            ],
            
            "TRIANGULAR_MULTIPLICATION" : [
                "tri_mul_outgoing",
                "tri_mul_incoming",
            ],
            
            "TRIANGULAR_ATTENTION" : [
                "seq_attention",
                "seq_attention_dropout",
                "tri_att_starting",
                "tri_att_ending",
            ], 

            "STRUCTURE_MODULE" : [
                "trunk2sm_projections",
                "structure_module",
                "distogram_calc",
                "prediction_heads",
            ],
        }
        self.timings: Dict[str, List[float]] = {}

    def __str__(self):

        max_key_len = max(len(k) for k in self.timings)

        forwardpass_total = float(np.sum(self.timings["total_forward"]))
        
        res = ""
        for group_name, events in self.relevant_events.items():
            res += f"{group_name}\n"
            res += "=" * 15 + "\n"
        
            group_total = 0.0
            for event in events:
                total = float(np.sum(self.timings[event]))
                group_total += total
                res += f"{event:<{max_key_len}}  {total:>10.4f}\n"
        
            percent = 100.0 * group_total / forwardpass_total
            res += "-" * 15 + "\n"
            res += f"{'total':<{max_key_len}}  {group_total:>10.4f}   ({percent:.2f} %)\n\n"
        
        return res
    
    def percents(self):
        forwardpass_total = float(np.sum(self.timings["total_forward"]))
        res = []
        for group_name, events in self.relevant_events.items():

            group_total = 0.0
            for event in events:
                total = float(np.sum(self.timings[event]))
                group_total += total

            percent = 100.0 * group_total / forwardpass_total

            res.append((group_name, percent))
        return res

    def reset(self):
        self.timings.clear()

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
            #print(name, elapsed_ms)

