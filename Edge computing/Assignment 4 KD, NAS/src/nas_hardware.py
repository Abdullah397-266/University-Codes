"""
Q2 - hardware-aware evaluation: actual on-device (CPU) measurements and deployment constraints.

For every candidate network the search measures, on the CPU of the machine running the search:
  * latency_ms   - median wall-clock time of a batch-1 forward pass (after warm-up), fixed thread count
  * size_mb      - size of the serialized FP32 state_dict (the bytes that would be shipped to the device)
  * peak_mem_mb  - peak tensor memory during one inference (weights + activations), from the PyTorch
                   profiler's allocation trace; None where the profiler does not support memory tracking
  * params, macs - for the report

`Constraints.check()` rejects a candidate that violates any limit BEFORE it is trained, so no GPU time
is wasted on networks that could never be deployed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn

from common import profile_model


@dataclass
class Constraints:
    max_latency_ms: float = 3.0      # batch-1 CPU latency budget
    max_size_mb: float = 1.5         # model storage budget
    max_peak_mem_mb: float = 2.5     # peak inference memory budget (checked when measurable)
    threads: int = 1                 # CPU threads used for latency (1 = single-core edge device)

    def check(self, prof: dict) -> tuple[bool, list[str]]:
        reasons = []
        if prof["latency_ms"] > self.max_latency_ms:
            reasons.append(f"latency {prof['latency_ms']:.2f} > {self.max_latency_ms} ms")
        if prof["size_mb"] > self.max_size_mb:
            reasons.append(f"size {prof['size_mb']:.2f} > {self.max_size_mb} MB")
        if prof.get("peak_mem_mb") is not None and prof["peak_mem_mb"] > self.max_peak_mem_mb:
            reasons.append(f"peak memory {prof['peak_mem_mb']:.1f} > {self.max_peak_mem_mb} MB")
        return not reasons, reasons

    def to_dict(self) -> dict:
        return asdict(self)


def measure_candidate(net: nn.Module, resolution: int, constraints: Constraints, runs: int = 50) -> dict:
    """Measure one untrained candidate on the CPU (weights do not affect latency / size / memory)."""
    return profile_model(net, (1, 3, resolution, resolution), runs=runs, threads=constraints.threads)
