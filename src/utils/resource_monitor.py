"""
资源监控。

`ResourceMonitor` 后台线程定时采样进程 RSS（含子进程）+ `torch.cuda.max_memory_reserved`，
stop 后返回一个 `ResourceUsage`（wall-clock + RAM/VRAM 峰值）。
`ensure_stage_metrics / update_stage_metrics` 维护一个按阶段累积的资源字典，挂在
args 上跨阶段聚合（attribute_selection / table_pairing / hierarchical_merging）。
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class ResourceUsage:
    elapsed_time: float = 0.0
    peak_memory_bytes: int = 0
    peak_memory_mb: float = 0.0
    peak_gpu_memory_bytes: int = 0
    peak_gpu_memory_mb: float = 0.0
    sampling_interval: float = 0.1
    available: bool = True
    gpu_available: bool = False


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


class ResourceMonitor:
    """
    Sample process RAM usage in the background and, when CUDA is available,
    also capture peak GPU memory reserved by the current process.
    """

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self._start_time: Optional[float] = None
        self._peak_memory_bytes = 0
        self._peak_gpu_memory_bytes = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process() if psutil is not None else None

    @property
    def available(self) -> bool:
        return self._process is not None

    @property
    def gpu_available(self) -> bool:
        return bool(torch is not None and torch.cuda.is_available())

    def _get_total_rss(self) -> int:
        if self._process is None:
            return 0
        total = 0
        try:
            total += self._process.memory_info().rss
        except Exception:
            pass
        try:
            for child in self._process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    continue
        except Exception:
            pass
        return total

    def _reset_gpu_peak_stats(self):
        if not self.gpu_available:
            return
        try:
            for device_idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_idx)
        except Exception:
            pass

    def _get_total_gpu_peak(self) -> int:
        if not self.gpu_available:
            return 0
        total = 0
        try:
            for device_idx in range(torch.cuda.device_count()):
                total += int(torch.cuda.max_memory_reserved(device_idx))
        except Exception:
            return 0
        return total

    def _sample_loop(self):
        while self._running:
            current_ram = self._get_total_rss()
            if current_ram > self._peak_memory_bytes:
                self._peak_memory_bytes = current_ram

            current_gpu = self._get_total_gpu_peak()
            if current_gpu > self._peak_gpu_memory_bytes:
                self._peak_gpu_memory_bytes = current_gpu

            time.sleep(self.sampling_interval)

    def start(self):
        if self._running:
            raise RuntimeError("ResourceMonitor is already running.")
        self._start_time = time.perf_counter()
        self._peak_memory_bytes = self._get_total_rss()
        self._reset_gpu_peak_stats()
        self._peak_gpu_memory_bytes = self._get_total_gpu_peak()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceUsage:
        if not self._running:
            raise RuntimeError("ResourceMonitor is not running.")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.sampling_interval * 3 + 0.2)

        self._peak_memory_bytes = max(self._peak_memory_bytes, self._get_total_rss())
        self._peak_gpu_memory_bytes = max(self._peak_gpu_memory_bytes, self._get_total_gpu_peak())

        elapsed = 0.0 if self._start_time is None else time.perf_counter() - self._start_time
        return ResourceUsage(
            elapsed_time=elapsed,
            peak_memory_bytes=int(self._peak_memory_bytes),
            peak_memory_mb=float(self._peak_memory_bytes / (1024 ** 2)),
            peak_gpu_memory_bytes=int(self._peak_gpu_memory_bytes),
            peak_gpu_memory_mb=float(self._peak_gpu_memory_bytes / (1024 ** 2)),
            sampling_interval=self.sampling_interval,
            available=self.available,
            gpu_available=self.gpu_available,
        )


def ensure_stage_metrics(container) -> Dict[str, dict]:
    metrics = getattr(container, "_stage_resource_metrics", None)
    if metrics is None:
        metrics = {
            "attribute_selection": _new_stage_metrics("Attribute Selection"),
            "table_pairing": _new_stage_metrics("Table Pairing"),
            "hierarchical_merging": _new_stage_metrics("Hierarchical Merging"),
        }
        setattr(container, "_stage_resource_metrics", metrics)
    return metrics


def _new_stage_metrics(label: str) -> dict:
    return {
        "label": label,
        "total_time": 0.0,
        "peak_memory_bytes": 0,
        "peak_memory_mb": 0.0,
        "peak_gpu_memory_bytes": 0,
        "peak_gpu_memory_mb": 0.0,
        "available": True,
        "gpu_available": False,
        "invocations": 0,
    }


def update_stage_metrics(container, stage_name: str, usage: ResourceUsage):
    metrics = ensure_stage_metrics(container)
    stage = metrics.setdefault(stage_name, _new_stage_metrics(stage_name))
    stage["total_time"] += usage.elapsed_time
    stage["peak_memory_bytes"] = max(stage["peak_memory_bytes"], int(usage.peak_memory_bytes))
    stage["peak_memory_mb"] = stage["peak_memory_bytes"] / (1024 ** 2)
    stage["peak_gpu_memory_bytes"] = max(stage["peak_gpu_memory_bytes"], int(usage.peak_gpu_memory_bytes))
    stage["peak_gpu_memory_mb"] = stage["peak_gpu_memory_bytes"] / (1024 ** 2)
    stage["available"] = bool(stage["available"] and usage.available)
    stage["gpu_available"] = bool(stage["gpu_available"] or usage.gpu_available)
    stage["invocations"] += 1
    return stage
