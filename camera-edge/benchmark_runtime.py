#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmark_profiles import BenchmarkProfile


@dataclass(frozen=True)
class RunContext:
    role: str
    run_id: str
    profile: BenchmarkProfile
    run_dir: Path
    start_wall_ns: int
    start_iso_utc: str
    hostname: str
    duration_sec: float | None

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "run_id": self.run_id,
            "start_wall_ns": self.start_wall_ns,
            "start_iso_utc": self.start_iso_utc,
            "hostname": self.hostname,
            "duration_sec": self.duration_sec,
            "profile": asdict(self.profile),
        }


class CsvMetricLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow({key: _jsonable(row.get(key)) for key in self._writer.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def build_run_context(
    *,
    role: str,
    profile: BenchmarkProfile,
    run_id: str | None,
    duration_sec: float | None,
) -> RunContext:
    effective_run_id = str(run_id or default_run_id(profile.name))
    root = Path(profile.artifact_root).resolve()
    run_dir = root / effective_run_id / role
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        role=str(role),
        run_id=effective_run_id,
        profile=profile,
        run_dir=run_dir,
        start_wall_ns=time.time_ns(),
        start_iso_utc=now_utc_iso(),
        hostname=socket.gethostname(),
        duration_sec=None if duration_sec is None else float(duration_sec),
    )


def elapsed_s(start_mono_s: float) -> float:
    return max(0.0, time.monotonic() - float(start_mono_s))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def write_manifest(context: RunContext, extra: Mapping[str, Any] | None = None) -> None:
    payload = context.manifest_dict()
    if extra:
        payload.update(dict(extra))
    write_json(context.run_dir / "manifest.json", payload)


def metric_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    series = [float(v) for v in values if v is not None]
    if not series:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(series)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(ordered, 95.0),
        "max": ordered[-1],
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = max(0.0, min(100.0, float(pct))) / 100.0
    idx = p * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def safe_ratio(num: float | int, den: float | int) -> float | None:
    if float(den) == 0.0:
        return None
    return float(num) / float(den)


def fps(count: int, duration_s: float | None) -> float | None:
    if duration_s is None or float(duration_s) <= 0.0:
        return None
    return float(count) / float(duration_s)


def classify_bottleneck(summary: Mapping[str, Any]) -> str:
    sender = dict(summary.get("sender", {}) or {})
    receiver = dict(summary.get("receiver", {}) or {})
    host = dict(summary.get("host", {}) or {})

    sender_depth_fps = _pick_number(sender, "depth_send_fps")
    receiver_depth_fps = _pick_number(receiver, "depth_receive_fps")
    stale_rate = _pick_number(receiver, "stale_depth_fraction")
    gap_rate = _pick_number(receiver, "sequence_gap_fraction")
    comp_ms = _pick_number(sender, "compression_ms_mean")
    decomp_ms = _pick_number(receiver, "decompression_ms_mean")
    frame_budget_ms = _pick_number(summary, "depth_frame_budget_ms")
    cpu_pct = _pick_number(host, "cpu_percent_mean")
    gpu_pct = _pick_number(host, "gpu_util_percent_mean")
    tx_mbps = _pick_number(host, "net_tx_mbps_mean")
    rx_mbps = _pick_number(host, "net_rx_mbps_mean")
    bandwidth = float(summary.get("profile", {}).get("nic_bandwidth_mbps", 100.0))

    if stale_rate is not None and stale_rate >= 0.05:
        return "network-bound"
    if gap_rate is not None and gap_rate >= 0.05:
        return "network-bound"
    if sender_depth_fps is not None and receiver_depth_fps is not None and receiver_depth_fps < (0.9 * sender_depth_fps):
        return "network-bound"
    if frame_budget_ms is not None:
        if comp_ms is not None and comp_ms >= 0.6 * frame_budget_ms:
            return "jetson-compute-bound"
        if decomp_ms is not None and decomp_ms >= 0.6 * frame_budget_ms:
            return "receiver-bound"
    if cpu_pct is not None and cpu_pct >= 85.0:
        return "jetson-compute-bound" if sender else "receiver-bound"
    if gpu_pct is not None and gpu_pct >= 85.0:
        return "jetson-compute-bound" if sender else "receiver-bound"
    if tx_mbps is not None and tx_mbps >= 0.85 * bandwidth:
        return "network-bound"
    if rx_mbps is not None and rx_mbps >= 0.85 * bandwidth:
        return "network-bound"
    if decomp_ms is not None and comp_ms is not None and decomp_ms > comp_ms:
        return "receiver-bound"
    return "balanced"


def summarize_host_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    cpu = metric_stats(_numbers(rows, "cpu_percent"))
    gpu = metric_stats(_numbers(rows, "gpu_util_percent"))
    ping = metric_stats(_numbers(rows, "ping_ms"))
    tx = metric_stats(_numbers(rows, "net_tx_mbps"))
    rx = metric_stats(_numbers(rows, "net_rx_mbps"))
    pwr = metric_stats(_numbers(rows, "power_mw"))
    tmp = metric_stats(_numbers(rows, "temp_c"))
    return {
        "sample_count": len(rows),
        "cpu_percent_mean": cpu["mean"],
        "cpu_percent_p95": cpu["p95"],
        "gpu_util_percent_mean": gpu["mean"],
        "gpu_util_percent_p95": gpu["p95"],
        "ping_ms_mean": ping["mean"],
        "ping_ms_p95": ping["p95"],
        "net_tx_mbps_mean": tx["mean"],
        "net_rx_mbps_mean": rx["mean"],
        "power_mw_mean": pwr["mean"],
        "temp_c_mean": tmp["mean"],
    }


def summarize_sender_rows(rows: list[Mapping[str, Any]], duration_s: float | None) -> dict[str, Any]:
    rgb_camera_rows = [row for row in rows if row.get("event") == "rgb_camera"]
    rgb_write_rows = [row for row in rows if row.get("event") == "rgb_rtsp_write"]
    depth_rows = [row for row in rows if row.get("event") == "depth_send"]
    depth_raw = [float(row["raw_bytes"]) for row in depth_rows if row.get("raw_bytes") not in ("", None)]
    depth_comp = [float(row["compressed_bytes"]) for row in depth_rows if row.get("compressed_bytes") not in ("", None)]
    ratios = [
        float(comp) / float(raw)
        for raw, comp in zip(depth_raw, depth_comp)
        if float(raw) > 0.0
    ]
    drained_rgb = sum(int(float(row.get("drained_count", 0) or 0)) for row in rgb_write_rows)
    drained_depth = sum(int(float(row.get("drained_count", 0) or 0)) for row in depth_rows)
    return {
        "rgb_camera_fps": fps(len(rgb_camera_rows), duration_s),
        "rgb_rtsp_write_fps": fps(len(rgb_write_rows), duration_s),
        "depth_send_fps": fps(len(depth_rows), duration_s),
        "compression_ms_mean": metric_stats(_numbers(depth_rows, "compression_ms"))["mean"],
        "send_ms_mean": metric_stats(_numbers(depth_rows, "send_ms"))["mean"],
        "compression_ratio_mean": metric_stats(ratios)["mean"],
        "depth_bytes_mean": metric_stats(depth_comp)["mean"],
        "packet_count_mean": metric_stats(_numbers(depth_rows, "packet_count"))["mean"],
        "rgb_drained_frames": drained_rgb,
        "depth_drained_frames": drained_depth,
        "depth_frames": len(depth_rows),
    }


def summarize_receiver_rows(rows: list[Mapping[str, Any]], duration_s: float | None) -> dict[str, Any]:
    rgb_rows = [row for row in rows if row.get("event") == "rgb_read"]
    depth_rows = [row for row in rows if row.get("event") == "depth_receive"]
    stale_rows = [row for row in rows if row.get("event") == "depth_stale_drop"]
    stale_drops = sum(int(float(row.get("stale_drops", 0) or 0)) for row in stale_rows) + sum(
        int(float(row.get("stale_drops", 0) or 0)) for row in depth_rows
    )
    seq_gaps = sum(int(float(row.get("sequence_gap", 0) or 0)) for row in depth_rows)
    depth_count = len(depth_rows)
    return {
        "rgb_read_fps": fps(len(rgb_rows), duration_s),
        "depth_receive_fps": fps(depth_count, duration_s),
        "decompression_ms_mean": metric_stats(_numbers(depth_rows, "decompression_ms"))["mean"],
        "latency_ms_mean": metric_stats(_numbers(depth_rows, "latency_ms"))["mean"],
        "latency_ms_p95": metric_stats(_numbers(depth_rows, "latency_ms"))["p95"],
        "rgb_latency_ms_mean": metric_stats(_numbers(depth_rows, "rgb_latency_ms"))["mean"],
        "rgb_latency_ms_p95": metric_stats(_numbers(depth_rows, "rgb_latency_ms"))["p95"],
        "fused_ready_latency_ms_mean": metric_stats(_numbers(depth_rows, "fused_ready_latency_ms"))["mean"],
        "fused_ready_latency_ms_p95": metric_stats(_numbers(depth_rows, "fused_ready_latency_ms"))["p95"],
        "fused_source_skew_ms_mean": metric_stats(_numbers(depth_rows, "fused_source_skew_ms"))["mean"],
        "fused_source_skew_ms_p95": metric_stats(_numbers(depth_rows, "fused_source_skew_ms"))["p95"],
        "rgb_pair_delta_ms_mean": metric_stats(_numbers(depth_rows, "rgb_pair_delta_ms"))["mean"],
        "stale_depth_drops": stale_drops,
        "sequence_gaps": seq_gaps,
        "missing_packets_total": sum(int(float(row.get("missing_packets", 0) or 0)) for row in stale_rows),
        "stale_depth_fraction": safe_ratio(stale_drops, depth_count),
        "sequence_gap_fraction": safe_ratio(seq_gaps, depth_count),
        "depth_frames": depth_count,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        result.append(float(value))
    return result


def _pick_number(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    return float(value)
