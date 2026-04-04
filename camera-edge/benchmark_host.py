#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import psutil
except Exception:
    psutil = None


TEGRA_GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%@(\d+)")
TEGRA_EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%@(\d+)")
TEGRA_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")


def parse_tegrastats_line(line: str) -> dict[str, float | int | None]:
    gpu_match = TEGRA_GPU_RE.search(line)
    emc_match = TEGRA_EMC_RE.search(line)
    ram_match = TEGRA_RAM_RE.search(line)
    cpu_block = re.search(r"CPU\s+\[([^\]]+)\]", line)
    cpu_samples: list[int] = []
    if cpu_block:
        for token in cpu_block.group(1).split(","):
            token = token.strip()
            match = re.match(r"(\d+)%@", token)
            if match:
                cpu_samples.append(int(match.group(1)))
    cpu_util = None
    if cpu_samples:
        cpu_util = sum(cpu_samples) / float(len(cpu_samples))

    return {
        "cpu_percent": cpu_util,
        "gpu_util_percent": float(gpu_match.group(1)) if gpu_match else None,
        "gpu_clock_mhz": float(gpu_match.group(2)) if gpu_match else None,
        "emc_util_percent": float(emc_match.group(1)) if emc_match else None,
        "emc_clock_mhz": float(emc_match.group(2)) if emc_match else None,
        "mem_used_mb": float(ram_match.group(1)) if ram_match else None,
        "mem_total_mb": float(ram_match.group(2)) if ram_match else None,
    }


def parse_nvidia_smi_csv(text: str) -> dict[str, float | None]:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    return {
        "gpu_util_percent": _parse_optional_float(parts[0]),
        "gpu_mem_used_mb": _parse_optional_float(parts[1]),
        "gpu_mem_total_mb": _parse_optional_float(parts[2]),
    }


def parse_ping_output(text: str) -> float | None:
    match = re.search(r"time[=<]\s*([0-9.]+)\s*ms", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"Average = (\d+)ms", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def measure_ping_ms(host: str, timeout_ms: int = 1000) -> float | None:
    if not host:
        return None
    if shutil.which("ping") is None:
        return None
    if os.name == "nt":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout_ms)), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, int(timeout_ms / 1000))), host]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(2.0, timeout_ms / 1000.0 + 1.0))
    except Exception:
        return None
    return parse_ping_output((proc.stdout or "") + "\n" + (proc.stderr or ""))


def query_nvidia_smi() -> dict[str, float | None]:
    if shutil.which("nvidia-smi") is None:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    if proc.returncode != 0:
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None}
    return parse_nvidia_smi_csv(proc.stdout or "")


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"n/a", "[n/a]", "na", "-", "--"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


@dataclass
class _NetSnapshot:
    ts_mono: float
    bytes_sent: float
    bytes_recv: float


class HostMetricsSampler:
    def __init__(
        self,
        *,
        role: str,
        nic_name: str,
        ping_host: str,
        write_row: Callable[[dict[str, Any]], None],
        start_mono_s: float,
        interval_s: float = 1.0,
    ) -> None:
        self.role = str(role)
        self.nic_name = str(nic_name)
        self.ping_host = str(ping_host)
        self.write_row = write_row
        self.start_mono_s = float(start_mono_s)
        self.interval_s = max(0.2, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ping_history: list[float] = []
        self._net_prev: _NetSnapshot | None = None
        self._tegra_proc: subprocess.Popen[str] | None = None
        self._latest_tegra: dict[str, float | int | None] = {}
        self._tegra_thread: threading.Thread | None = None

    def start(self) -> None:
        self._start_tegrastats()
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name=f"host-metrics-{self.role}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._tegra_proc is not None:
            try:
                self._tegra_proc.terminate()
                self._tegra_proc.wait(timeout=2.0)
            except Exception:
                pass
            self._tegra_proc = None
        if self._tegra_thread is not None:
            self._tegra_thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            row = self._sample_row()
            self.write_row(row)
            self._stop.wait(self.interval_s)

    def _sample_row(self) -> dict[str, Any]:
        now_ns = time.time_ns()
        now_mono = time.monotonic()
        row: dict[str, Any] = {
            "timestamp_ns": now_ns,
            "elapsed_s": max(0.0, now_mono - self.start_mono_s),
            "role": self.role,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "nic_name": self.nic_name,
            "cpu_percent": None,
            "mem_used_mb": None,
            "mem_total_mb": None,
            "mem_percent": None,
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "net_tx_mbps": None,
            "net_rx_mbps": None,
            "net_bytes_sent": None,
            "net_bytes_recv": None,
            "ping_ms": None,
            "ping_jitter_ms": None,
            "emc_util_percent": None,
        }

        if psutil is not None:
            try:
                row["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                row["mem_used_mb"] = float(vm.used) / (1024.0 * 1024.0)
                row["mem_total_mb"] = float(vm.total) / (1024.0 * 1024.0)
                row["mem_percent"] = float(vm.percent)
            except Exception:
                pass
            try:
                pernic = psutil.net_io_counters(pernic=True)
                if self.nic_name in pernic:
                    nic = pernic[self.nic_name]
                    row["net_bytes_sent"] = float(nic.bytes_sent)
                    row["net_bytes_recv"] = float(nic.bytes_recv)
                    current = _NetSnapshot(now_mono, float(nic.bytes_sent), float(nic.bytes_recv))
                    if self._net_prev is not None:
                        dt = max(1e-6, current.ts_mono - self._net_prev.ts_mono)
                        row["net_tx_mbps"] = ((current.bytes_sent - self._net_prev.bytes_sent) * 8.0) / (dt * 1e6)
                        row["net_rx_mbps"] = ((current.bytes_recv - self._net_prev.bytes_recv) * 8.0) / (dt * 1e6)
                    self._net_prev = current
            except Exception:
                pass

        ping_ms = measure_ping_ms(self.ping_host)
        if ping_ms is not None:
            row["ping_ms"] = ping_ms
            self._ping_history.append(float(ping_ms))
            if len(self._ping_history) >= 2:
                row["ping_jitter_ms"] = abs(self._ping_history[-1] - self._ping_history[-2])

        nvidia = query_nvidia_smi()
        row.update({k: v for k, v in nvidia.items() if v is not None})
        row.update({k: v for k, v in self._latest_tegra.items() if v is not None})
        return row

    def _start_tegrastats(self) -> None:
        if shutil.which("tegrastats") is None:
            return
        try:
            self._tegra_proc = subprocess.Popen(
                ["tegrastats", "--interval", str(int(self.interval_s * 1000.0))],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception:
            self._tegra_proc = None
            return
        self._tegra_thread = threading.Thread(target=self._tegrastats_loop, name="tegrastats-reader", daemon=True)
        self._tegra_thread.start()

    def _tegrastats_loop(self) -> None:
        proc = self._tegra_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if self._stop.is_set():
                    break
                parsed = parse_tegrastats_line(line.strip())
                if parsed:
                    self._latest_tegra = parsed
        except Exception:
            return
