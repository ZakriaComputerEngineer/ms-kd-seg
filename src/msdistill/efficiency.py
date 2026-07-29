"""Efficiency benchmarking.

Parameter count is a poor proxy for deployment cost and, in our earlier run, a
misleading one: a 7.6x parameter reduction bought only a 1.6x latency
improvement, because the small U-Net does all of its work at full 256x256
resolution while the transformer immediately downsamples. Anyone reading a
parameter-count-only comparison would have drawn the wrong conclusion.

So this module measures what actually matters for a clinical workstation:
multiply-accumulate cost, wall-clock latency at batch size 1 on both GPU and CPU,
sustained throughput, and peak memory.

Measurement protocol
--------------------
* untimed warmup iterations to trigger autotuning and allocator growth
* `torch.cuda.synchronize()` on both sides of the timed region (CUDA launches are
  asynchronous, so timing without it measures queueing, not computation)
* median and interquartile range over many trials rather than a mean, because
  latency distributions on a shared GPU have a long right tail
* CPU timing under a fixed, small thread count, since a clinical machine is not a
  many-core server
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EfficiencyReport:
    name: str
    params_total: int
    params_trainable: int
    checkpoint_mb: float = float("nan")
    gmacs: float = float("nan")
    gflops: float = float("nan")
    flops_method: str = "unavailable"
    gpu_latency_ms: Dict[int, float] = field(default_factory=dict)
    gpu_latency_iqr_ms: Dict[int, float] = field(default_factory=dict)
    gpu_throughput_ips: Dict[int, float] = field(default_factory=dict)
    cpu_latency_ms: Dict[int, float] = field(default_factory=dict)
    peak_gpu_memory_mb: float = float("nan")
    volume_latency_ms: float = float("nan")

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Compute cost
# --------------------------------------------------------------------------- #

def _flops_via_torch_counter(model: nn.Module, example: torch.Tensor) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        model.eval()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(example)
        total = counter.get_total_flops()
        return float(total) if total else None
    except Exception:
        return None


def _flops_via_thop(model: nn.Module, example: torch.Tensor) -> Optional[float]:
    try:
        from thop import profile
    except ImportError:
        return None
    try:
        macs, _ = profile(model, inputs=(example,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


def _flops_via_hooks(model: nn.Module, example: torch.Tensor) -> Optional[float]:
    """Conv/linear-only fallback.

    Reported separately from the exact counters because it omits attention
    matmuls, and so *understates* a transformer's cost. Never mix the two methods
    within one table.
    """
    macs = [0.0]
    handles = []

    def conv_hook(module, inputs, output):
        out_elems = output.numel() / output.shape[0]
        k = int(np.prod(module.kernel_size))
        macs[0] += out_elems * module.in_channels * k / max(module.groups, 1) * output.shape[0]

    def deconv_hook(module, inputs, output):
        in_elems = inputs[0].numel() / inputs[0].shape[0]
        k = int(np.prod(module.kernel_size))
        macs[0] += in_elems * module.out_channels * k / max(module.groups, 1) * inputs[0].shape[0]

    def linear_hook(module, inputs, output):
        macs[0] += float(output.numel()) * module.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.ConvTranspose2d):
            handles.append(m.register_forward_hook(deconv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    try:
        model.eval()
        with torch.no_grad():
            model(example)
    except Exception:
        return None
    finally:
        for h in handles:
            h.remove()
    return macs[0] * 2.0 if macs[0] > 0 else None


def measure_flops(model: nn.Module, example: torch.Tensor) -> Tuple[float, str]:
    for fn, label in ((_flops_via_torch_counter, "torch.utils.flop_counter"),
                      (_flops_via_thop, "thop"),
                      (_flops_via_hooks, "conv/linear hooks (attention omitted)")):
        value = fn(model, example)
        if value:
            return value, label
    return float("nan"), "unavailable"


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #

def _timed_runs(run: Callable[[], None], warmup: int, trials: int,
                synchronize: Optional[Callable[[], None]] = None) -> List[float]:
    for _ in range(warmup):
        run()
    if synchronize:
        synchronize()

    samples: List[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        run()
        if synchronize:
            synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def benchmark_latency(model: nn.Module, batch_size: int, in_channels: int,
                      spatial: Tuple[int, int], device: torch.device,
                      warmup: int, trials: int) -> Tuple[float, float]:
    """Returns (median_ms, interquartile_range_ms) for one forward pass."""
    model = model.to(device).eval()
    example = torch.randn(batch_size, in_channels, *spatial, device=device)

    def run():
        with torch.no_grad():
            model(example)

    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else None
    samples = _timed_runs(run, warmup, trials, sync)
    samples.sort()
    median = statistics.median(samples)
    iqr = samples[int(0.75 * len(samples)) - 1] - samples[int(0.25 * len(samples))]
    return float(median), float(max(iqr, 0.0))


def peak_gpu_memory(model: nn.Module, batch_size: int, in_channels: int,
                    spatial: Tuple[int, int], device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    example = torch.randn(batch_size, in_channels, *spatial, device=device)
    with torch.no_grad():
        model(example)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def checkpoint_size_mb(path: Optional[str]) -> float:
    if path and os.path.exists(path):
        return float(os.path.getsize(path) / (1024 ** 2))
    return float("nan")


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #

def profile_model(model: nn.Module, name: str, cfg, device: torch.device,
                  checkpoint: Optional[str] = None,
                  slices_per_volume: int = 20,
                  cpu_threads: int = 4) -> EfficiencyReport:
    """Full efficiency profile for one architecture."""
    in_channels = len(cfg.modalities)
    spatial = tuple(cfg.target_size)

    report = EfficiencyReport(
        name=name,
        params_total=sum(p.numel() for p in model.parameters()),
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_mb=checkpoint_size_mb(checkpoint),
    )

    example = torch.randn(1, in_channels, *spatial, device=device)
    flops, method = measure_flops(model.to(device), example)
    report.gflops = flops / 1e9 if flops == flops else float("nan")
    report.gmacs = report.gflops / 2.0 if report.gflops == report.gflops else float("nan")
    report.flops_method = method

    for bs in cfg.benchmark_batch_sizes:
        median, iqr = benchmark_latency(model, bs, in_channels, spatial, device,
                                        cfg.benchmark_warmup, cfg.benchmark_trials)
        report.gpu_latency_ms[bs] = median
        report.gpu_latency_iqr_ms[bs] = iqr
        report.gpu_throughput_ips[bs] = (bs / median * 1000.0) if median > 0 else float("nan")

    # Time for one complete patient study, the unit a radiologist waits on.
    if 1 in report.gpu_latency_ms:
        per_slice = report.gpu_latency_ms.get(min(cfg.benchmark_batch_sizes),
                                              report.gpu_latency_ms[1])
        bs = min(cfg.benchmark_batch_sizes)
        report.volume_latency_ms = per_slice * (slices_per_volume / bs)

    report.peak_gpu_memory_mb = peak_gpu_memory(model, 1, in_channels, spatial, device)

    if cfg.benchmark_cpu:
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(max(1, min(cpu_threads, os.cpu_count() or 1)))
            # Same warmup and trial counts as the GPU path. The generated table
            # caption states one protocol for every column, so running a reduced
            # budget here would make the paper describe a measurement it did not
            # take -- and CPU latency is the headline deployment number. Fifty
            # passes of a 0.49M network costs a few seconds.
            cpu_model = model.to("cpu")
            median, _ = benchmark_latency(cpu_model, 1, in_channels, spatial,
                                          torch.device("cpu"),
                                          cfg.benchmark_warmup, cfg.benchmark_trials)
            report.cpu_latency_ms[1] = median
        finally:
            torch.set_num_threads(previous)
            model.to(device)

    return report


def environment_summary(device: torch.device) -> Dict[str, str]:
    """Recorded alongside the timings; latency numbers are meaningless without it."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_type": device.type,
        "cpu_count": str(os.cpu_count()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda or "unknown"
        props = torch.cuda.get_device_properties(device)
        info["gpu_memory_gb"] = f"{props.total_memory / (1024 ** 3):.1f}"
    return info
