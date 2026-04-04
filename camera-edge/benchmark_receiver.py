#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

try:
    import cv2
except Exception:
    cv2 = None

from benchmark_host import HostMetricsSampler
from benchmark_profiles import BenchmarkProfile, get_profile, profile_names
from benchmark_runtime import (
    CsvMetricLogger,
    build_run_context,
    classify_bottleneck,
    elapsed_s,
    summarize_host_rows,
    summarize_receiver_rows,
    write_json,
    write_manifest,
)
from benchmark_sync import ClockSyncServer
from rgbd_protocol import (
    HDR_CAL,
    HDR_DPT,
    MAGIC_CAL,
    MAGIC_DPT,
    compression_name,
    decompress_blob,
    make_decompressor,
)


RECEIVER_FIELDS = [
    "timestamp_ns",
    "elapsed_s",
    "event",
    "seq",
    "compression",
    "packet_count",
    "compressed_bytes",
    "depth_bytes",
    "assembly_ms",
    "decompression_ms",
    "latency_ms",
    "rgb_pair_delta_ms",
    "sequence_gap",
    "stale_drops",
    "missing_packets",
    "calibration_version",
    "rgb_width",
    "rgb_height",
    "depth_width",
    "depth_height",
]

HOST_FIELDS = [
    "timestamp_ns",
    "elapsed_s",
    "role",
    "hostname",
    "platform",
    "nic_name",
    "cpu_percent",
    "mem_used_mb",
    "mem_total_mb",
    "mem_percent",
    "gpu_util_percent",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "net_tx_mbps",
    "net_rx_mbps",
    "net_bytes_sent",
    "net_bytes_recv",
    "ping_ms",
    "ping_jitter_ms",
    "emc_util_percent",
]


@dataclass(frozen=True)
class _RgbFrame:
    recv_mono_s: float
    recv_wall_ns: int
    width: int
    height: int


class TransportBenchmarkReceiver:
    def __init__(
        self,
        *,
        profile: BenchmarkProfile,
        record_metric,
        start_mono_s: float,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for RTSP capture on the benchmark receiver.")
        self.profile = profile
        self.record_metric = record_metric
        self.start_mono_s = float(start_mono_s)
        self.stop_event = threading.Event()
        self.rgb_history: deque[_RgbFrame] = deque(maxlen=max(2, int(profile.max_rgb_buffer)))
        self.rgb_lock = threading.Lock()
        self.pending: dict[int, dict[str, object]] = {}
        self.threads: list[threading.Thread] = []
        self.sock: socket.socket | None = None
        self.cap = None
        self.calibration_version: int | None = None
        self.prev_complete_seq: int | None = None
        self.fatal_error: str | None = None
        self.rgb_frames = 0
        self.depth_frames = 0
        self.stale_drops = 0
        self.sequence_gaps = 0
        self.last_latency_ms = 0.0

    def start(self) -> None:
        self.threads = [
            threading.Thread(target=self._rgb_loop, name="benchmark-rgb", daemon=True),
            threading.Thread(target=self._depth_loop, name="benchmark-depth", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        for thread in list(self.threads):
            thread.join(timeout=5.0)
        self.threads = []

    def status(self) -> dict[str, float | int]:
        return {
            "rgb_frames": int(self.rgb_frames),
            "depth_frames": int(self.depth_frames),
            "stale_drops": int(self.stale_drops),
            "sequence_gaps": int(self.sequence_gaps),
            "last_latency_ms": float(self.last_latency_ms),
        }

    def _rgb_loop(self) -> None:
        try:
            ffmpeg_opts = f"rtsp_transport;{self.profile.rtsp_transport}|fflags;nobuffer|flags;low_delay"
            if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts

            try:
                cap = cv2.VideoCapture(self.profile.rtsp_url, cv2.CAP_FFMPEG)
            except Exception:
                cap = None
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = cv2.VideoCapture(self.profile.rtsp_url)
            self.cap = cap
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            except Exception:
                pass
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open RTSP stream: {self.profile.rtsp_url}")

            while not self.stop_event.is_set():
                ok, frame_bgr = cap.read()
                recv_mono_s = time.monotonic()
                recv_wall_ns = time.time_ns()
                if not ok or frame_bgr is None:
                    time.sleep(0.01)
                    continue
                height, width = frame_bgr.shape[:2]
                rgb_msg = _RgbFrame(
                    recv_mono_s=float(recv_mono_s),
                    recv_wall_ns=int(recv_wall_ns),
                    width=int(width),
                    height=int(height),
                )
                with self.rgb_lock:
                    self.rgb_history.append(rgb_msg)
                self.rgb_frames += 1
                self.record_metric(
                    {
                        "timestamp_ns": recv_wall_ns,
                        "elapsed_s": elapsed_s(self.start_mono_s),
                        "event": "rgb_read",
                        "rgb_width": width,
                        "rgb_height": height,
                    }
                )
        except Exception as exc:
            self.fatal_error = f"RGB loop failed: {exc}"
            self.stop_event.set()

    def _depth_loop(self) -> None:
        try:
            zstd_d, lz4mod = make_decompressor()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(self.profile.socket_rcvbuf))
            sock.bind((self.profile.bind_ip, int(self.profile.depth_port)))
            sock.settimeout(0.5)
            self.sock = sock

            while not self.stop_event.is_set():
                self._cleanup_stale()
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise

                if len(data) < 4:
                    continue
                magic = data[:4]
                if magic == MAGIC_CAL:
                    if len(data) >= HDR_CAL.size:
                        _magic, version, _ts_ns, width, height = HDR_CAL.unpack_from(data, 0)
                        self.calibration_version = int(version)
                        self.record_metric(
                            {
                                "timestamp_ns": time.time_ns(),
                                "elapsed_s": elapsed_s(self.start_mono_s),
                                "event": "calibration",
                                "calibration_version": int(version),
                                "depth_width": int(width),
                                "depth_height": int(height),
                            }
                        )
                    continue

                if magic != MAGIC_DPT or len(data) < HDR_DPT.size:
                    continue

                _magic, seq, idx, count, ts_ns, width, height, comp, n = HDR_DPT.unpack_from(data, 0)
                payload = data[HDR_DPT.size : HDR_DPT.size + n]
                if len(payload) != n or int(idx) >= int(count):
                    continue

                recv_mono_s = time.monotonic()
                state = self.pending.get(int(seq))
                if state is None:
                    state = {
                        "t0": recv_mono_s,
                        "w": int(width),
                        "h": int(height),
                        "comp": int(comp),
                        "count": int(count),
                        "got": 0,
                        "parts": [None] * int(count),
                        "ts_ns": int(ts_ns),
                    }
                    self.pending[int(seq)] = state

                parts = state["parts"]
                assert isinstance(parts, list)
                if parts[int(idx)] is None:
                    parts[int(idx)] = payload
                    state["got"] = int(state["got"]) + 1

                if int(state["got"]) != int(state["count"]):
                    continue

                blob = b"".join(parts)
                self.pending.pop(int(seq), None)
                expected = int(state["w"]) * int(state["h"]) * 2
                t0 = time.perf_counter()
                try:
                    raw = decompress_blob(blob, int(state["comp"]), expected=expected, zstd_d=zstd_d, lz4mod=lz4mod)
                except Exception:
                    continue
                decomp_ms = (time.perf_counter() - t0) * 1000.0
                if len(raw) < expected:
                    continue

                pair_delta_ms, rgb_width, rgb_height = self._pair_rgb(float(recv_mono_s))
                gap = 0
                if self.prev_complete_seq is not None:
                    gap = max(0, int(seq) - int(self.prev_complete_seq) - 1)
                    self.sequence_gaps += gap
                self.prev_complete_seq = int(seq)
                self.depth_frames += 1
                self.last_latency_ms = (time.time_ns() - int(state["ts_ns"])) / 1e6
                self.record_metric(
                    {
                        "timestamp_ns": time.time_ns(),
                        "elapsed_s": elapsed_s(self.start_mono_s),
                        "event": "depth_receive",
                        "seq": int(seq),
                        "compression": compression_name(int(state["comp"])),
                        "packet_count": int(state["count"]),
                        "compressed_bytes": len(blob),
                        "depth_bytes": expected,
                        "assembly_ms": (float(recv_mono_s) - float(state["t0"])) * 1000.0,
                        "decompression_ms": decomp_ms,
                        "latency_ms": self.last_latency_ms,
                        "rgb_pair_delta_ms": pair_delta_ms,
                        "sequence_gap": gap,
                        "stale_drops": 0,
                        "missing_packets": 0,
                        "calibration_version": self.calibration_version,
                        "rgb_width": rgb_width,
                        "rgb_height": rgb_height,
                        "depth_width": int(state["w"]),
                        "depth_height": int(state["h"]),
                    }
                )
        except Exception as exc:
            self.fatal_error = f"Depth loop failed: {exc}"
            self.stop_event.set()

    def _cleanup_stale(self) -> None:
        now_mono = time.monotonic()
        stale = []
        for seq, state in list(self.pending.items()):
            age_ms = (now_mono - float(state["t0"])) * 1000.0
            if age_ms > float(self.profile.stale_depth_ms):
                stale.append((int(seq), state))
                self.pending.pop(int(seq), None)
        for seq, state in stale:
            missing = max(0, int(state["count"]) - int(state["got"]))
            self.stale_drops += 1
            self.record_metric(
                {
                    "timestamp_ns": time.time_ns(),
                    "elapsed_s": elapsed_s(self.start_mono_s),
                    "event": "depth_stale_drop",
                    "seq": seq,
                    "compression": compression_name(int(state["comp"])),
                    "packet_count": int(state["count"]),
                    "compressed_bytes": None,
                    "depth_bytes": int(state["w"]) * int(state["h"]) * 2,
                    "assembly_ms": None,
                    "decompression_ms": None,
                    "latency_ms": None,
                    "rgb_pair_delta_ms": None,
                    "sequence_gap": 0,
                    "stale_drops": 1,
                    "missing_packets": missing,
                    "calibration_version": self.calibration_version,
                    "depth_width": int(state["w"]),
                    "depth_height": int(state["h"]),
                }
            )

    def _pair_rgb(self, recv_mono_s: float) -> tuple[float | None, int | None, int | None]:
        with self.rgb_lock:
            if not self.rgb_history:
                return None, None, None
            best = min(self.rgb_history, key=lambda msg: abs(float(msg.recv_mono_s) - float(recv_mono_s)))
        delta_ms = abs(float(recv_mono_s) - float(best.recv_mono_s)) * 1000.0
        return delta_ms, int(best.width), int(best.height)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Transport-focused RGB-D benchmark receiver.")
    ap.add_argument("--profile", choices=profile_names(), default=None, help="Benchmark profile name.")
    ap.add_argument("--run-id", default=None, help="Optional run identifier shared with the sender.")
    ap.add_argument("--duration-sec", type=float, default=None, help="Override the profile duration.")
    ap.add_argument("--print-config", action="store_true", help="Print the effective config and exit.")
    return ap.parse_args()


def print_config(profile: BenchmarkProfile, duration_sec: float | None) -> None:
    print("Effective benchmark receiver config:")
    for key, value in sorted(profile.to_dict().items()):
        print(f"  {key}: {value}")
    print(f"  effective_receiver_nic_name: {profile.nic_name_for_role('receiver')}")
    print(f"  effective_receiver_ping_host: {profile.ping_host_for_role('receiver')}")
    print(f"  effective_duration_sec: {duration_sec}")


def main() -> int:
    args = parse_args()
    profile = get_profile(args.profile)
    duration_sec = float(args.duration_sec) if args.duration_sec is not None else float(profile.duration_sec)
    if args.print_config:
        print_config(profile, duration_sec)
        return 0

    context = build_run_context(role="receiver", profile=profile, run_id=args.run_id, duration_sec=duration_sec)
    receiver_logger = CsvMetricLogger(context.run_dir / "receiver_metrics.csv", RECEIVER_FIELDS)
    host_logger = CsvMetricLogger(context.run_dir / "host_metrics.csv", HOST_FIELDS)
    receiver_rows: list[dict[str, object]] = []
    host_rows: list[dict[str, object]] = []
    start_mono_s = time.monotonic()
    deadline = None if duration_sec <= 0 else start_mono_s + duration_sec

    def record_metric(row: dict[str, object]) -> None:
        receiver_rows.append(row)
        receiver_logger.write(row)

    def record_host(row: dict[str, object]) -> None:
        host_rows.append(row)
        host_logger.write(row)

    host_sampler = HostMetricsSampler(
        role="receiver",
        nic_name=profile.nic_name_for_role("receiver"),
        ping_host=profile.ping_host_for_role("receiver"),
        write_row=record_host,
        start_mono_s=start_mono_s,
        interval_s=1.0,
    )
    host_sampler.start()

    clock_server = ClockSyncServer(bind_ip=profile.bind_ip, port=profile.control_port)
    clock_server.start()

    write_manifest(
        context,
        extra={
            "compression_backend": "cpu",
            "gpu_interpretation": "GPU charts measure host load only; no GPU Zstd/LZ4 path is implemented here.",
            "clock_sync_server": {"bind_ip": profile.bind_ip, "port": int(profile.control_port)},
        },
    )

    receiver = TransportBenchmarkReceiver(profile=profile, record_metric=record_metric, start_mono_s=start_mono_s)
    print(f"[receiver] profile={profile.name} run_id={context.run_id}")
    print(f"[receiver] RTSP <- {profile.rtsp_url} ({profile.rtsp_transport})")
    print(f"[receiver] UDP depth <- {profile.bind_ip}:{profile.depth_port}")
    print(f"[receiver] Clock sync server <- {profile.bind_ip}:{profile.control_port}")

    try:
        receiver.start()
        last_print = time.monotonic()
        while True:
            if receiver.fatal_error:
                raise RuntimeError(receiver.fatal_error)
            if deadline is not None and time.monotonic() >= deadline:
                print("[receiver] duration reached; stopping benchmark run.")
                break
            if time.monotonic() - last_print >= 1.0:
                status = receiver.status()
                live_elapsed = max(1e-6, elapsed_s(start_mono_s))
                print(
                    f"[receiver] rgb_read_fps={status['rgb_frames'] / live_elapsed:.2f} "
                    f"depth_receive_fps={status['depth_frames'] / live_elapsed:.2f} "
                    f"latency_ms={status['last_latency_ms']:.1f} "
                    f"stale={status['stale_drops']} gaps={status['sequence_gaps']}"
                )
                last_print = time.monotonic()
            time.sleep(0.1)
    finally:
        receiver.stop()
        clock_server.stop()
        host_sampler.stop()
        receiver_logger.close()
        host_logger.close()

    actual_duration_s = elapsed_s(start_mono_s)
    summary = {
        "role": "receiver",
        "run_id": context.run_id,
        "profile_name": profile.name,
        "network_label": profile.network_label,
        "duration_s": actual_duration_s,
        "depth_frame_budget_ms": 1000.0 / max(1.0, float(profile.depth_fps)),
        "receiver": summarize_receiver_rows(receiver_rows, actual_duration_s),
        "host": summarize_host_rows(host_rows),
        "compression_backend": "cpu",
        "gpu_interpretation": "GPU charts measure host load only; no GPU Zstd/LZ4 path is implemented here.",
    }
    summary["bottleneck"] = classify_bottleneck(summary)
    write_json(context.run_dir / "summary.json", summary)
    print(f"[receiver] artifacts -> {context.run_dir}")
    print(f"[receiver] bottleneck hint -> {summary['bottleneck']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
