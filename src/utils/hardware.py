from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def detect_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_model_parameters(model: Any | None) -> int | None:
    if model is None or not hasattr(model, "parameters"):
        return None
    return sum(parameter.numel() for parameter in model.parameters())


def _get_vram_snapshot_mb() -> list[float]:
    if torch is not None and torch.cuda.is_available():
        values = []
        for index in range(torch.cuda.device_count()):
            values.append(torch.cuda.max_memory_allocated(index) / (1024 * 1024))
        return values
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            snapshots = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                snapshots.append(memory.used / (1024 * 1024))
            return snapshots
        except Exception:
            return []
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    return [float(line.strip()) for line in output.splitlines() if line.strip()]


@dataclass
class ResourceMonitor:
    device: str = "cpu"
    process: psutil.Process = field(default_factory=lambda: psutil.Process(os.getpid()))
    start_time: float = field(default=0.0, init=False)
    peak_ram_mb: float = field(default=0.0, init=False)
    peak_vram_mb: list[float] = field(default_factory=list, init=False)

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.peak_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        if self.device == "cuda" and torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)

    def sample(self) -> None:
        current_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = max(self.peak_ram_mb, current_ram_mb)
        current_vram = _get_vram_snapshot_mb()
        if not self.peak_vram_mb:
            self.peak_vram_mb = current_vram
        else:
            width = max(len(self.peak_vram_mb), len(current_vram))
            merged: list[float] = []
            for index in range(width):
                left = self.peak_vram_mb[index] if index < len(self.peak_vram_mb) else 0.0
                right = current_vram[index] if index < len(current_vram) else 0.0
                merged.append(max(left, right))
            self.peak_vram_mb = merged

    def stop(self) -> dict[str, Any]:
        self.sample()
        runtime_seconds = time.perf_counter() - self.start_time
        return {
            "runtime_seconds": runtime_seconds,
            "peak_ram_mb": round(self.peak_ram_mb, 3),
            "peak_vram_mb": [round(value, 3) for value in self.peak_vram_mb],
        }
