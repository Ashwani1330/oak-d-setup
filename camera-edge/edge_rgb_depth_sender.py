#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import threading
import time

import numpy as np

try:
    import depthai as dai
except Exception:
    dai = None

from benchmark_host import HostMetricsSampler
from benchmark_profiles import BenchmarkProfile, get_profile, profile_names
from benchmark_runtime import (
    CsvMetricLogger,
    build_run_context,
    classify_bottleneck,
    elapsed_s,
    summarize_host_rows,
    summarize_sender_rows,
    write_json,
    write_manifest,
)
from benchmark_sync import estimate_clock_offset
from rgbd_protocol import CAL_FLOATS, HDR_CAL, HDR_DPT, MAGIC_CAL, MAGIC_DPT, UNITS_CM, DepthCompressor


quit_event = threading.Event()

SENDER_FIELDS = [
    "timestamp_ns",
    "elapsed_s",
    "event",
    "index",
    "seq",
    "compression",
    "raw_bytes",
    "compressed_bytes",
    "compression_ms",
    "send_ms",
    "packet_count",
    "packet_bytes",
    "ffmpeg_write_bytes",
    "ffmpeg_write_ms",
    "drained_count",
    "source_ts_ns",
    "clock_offset_ns",
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


def install_signals() -> None:
    signal.signal(signal.SIGINT, lambda *_: quit_event.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())


def make_ffmpeg_cmd(profile: BenchmarkProfile) -> list[str]:
    rtp_ticks = 90000 // max(1, int(profile.fps))
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "warning",
        "-f",
        "h264",
        "-i",
        "pipe:0",
        "-c:v",
        "copy",
        "-bsf:v",
        f"setts=ts=N*{rtp_ticks}",
        "-flush_packets",
        "1",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        profile.rtsp_transport,
        "-pkt_size",
        str(int(profile.depth_packet_bytes)),
        profile.rtsp_url,
    ]


def build_cal_packet(w: int, h: int) -> bytes:
    if dai is None:
        raise RuntimeError("depthai is required to build the calibration packet on the sender.")
    with dai.Device() as device:
        calib = device.readCalibration()
        k_rgb = np.array(
            calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h),
            dtype=np.float32,
        )
        k_depth = np.array(
            calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h),
            dtype=np.float32,
        )
        t_depth_to_rgb = np.array(
            calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A),
            dtype=np.float32,
        )
        t_3x4 = t_depth_to_rgb[:3, :4]
        payload_floats = list(k_rgb.reshape(-1)) + list(k_depth.reshape(-1)) + list(t_3x4.reshape(-1))
        assert len(payload_floats) == 30
    ts_ns = time.time_ns()
    return HDR_CAL.pack(MAGIC_CAL, 1, ts_ns, w, h) + CAL_FLOATS.pack(*payload_floats, UNITS_CM)


def configure_stereo_filters(
    stereo: dai.node.StereoDepth,
    *,
    min_mm: int,
    max_mm: int,
    force_decimation_1: bool,
) -> None:
    try:
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    except Exception:
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        except Exception:
            pass

    try:
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    except Exception:
        pass

    cfg = stereo.initialConfig
    if force_decimation_1:
        try:
            cfg.postProcessing.decimationFilter.decimationFactor = 1
        except Exception as exc:
            print("[sender] WARN: cannot set decimationFactor=1:", exc)

    try:
        cfg.postProcessing.thresholdFilter.minRange = int(min_mm)
        cfg.postProcessing.thresholdFilter.maxRange = int(max_mm)
    except Exception:
        pass
    try:
        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 50
    except Exception:
        pass
    try:
        cfg.postProcessing.temporalFilter.enable = True
        if hasattr(cfg.postProcessing.temporalFilter, "alpha"):
            cfg.postProcessing.temporalFilter.alpha = 0.4
    except Exception:
        pass
    try:
        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1
    except Exception:
        pass


def drain_latest(queue_obj):
    message = queue_obj.tryGet()
    if message is None:
        return None, 0
    drained = 0
    while True:
        newer = queue_obj.tryGet()
        if newer is None:
            break
        message = newer
        drained += 1
    return message, drained


def print_config(profile: BenchmarkProfile, duration_sec: float | None) -> None:
    print("Effective benchmark sender config:")
    for key, value in sorted(profile.to_dict().items()):
        print(f"  {key}: {value}")
    print(f"  effective_sender_nic_name: {profile.nic_name_for_role('sender')}")
    print(f"  effective_sender_ping_host: {profile.ping_host_for_role('sender')}")
    print(f"  effective_duration_sec: {duration_sec}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Profile-driven OAK-D sender with benchmark artifact capture.")
    ap.add_argument("--profile", choices=profile_names(), default=None, help="Benchmark profile name.")
    ap.add_argument("--run-id", default=None, help="Optional run identifier shared with the receiver.")
    ap.add_argument("--duration-sec", type=float, default=None, help="Override the profile duration.")
    ap.add_argument("--print-config", action="store_true", help="Print the effective config and exit.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_profile(args.profile)
    duration_sec = float(args.duration_sec) if args.duration_sec is not None else float(profile.duration_sec)
    if args.print_config:
        print_config(profile, duration_sec)
        return 0
    if dai is None:
        raise RuntimeError("depthai is not installed in this environment. Install the camera-edge requirements on the Jetson sender.")

    install_signals()
    context = build_run_context(role="sender", profile=profile, run_id=args.run_id, duration_sec=duration_sec)
    sender_logger = CsvMetricLogger(context.run_dir / "sender_metrics.csv", SENDER_FIELDS)
    host_logger = CsvMetricLogger(context.run_dir / "host_metrics.csv", HOST_FIELDS)
    sender_rows: list[dict[str, object]] = []
    host_rows: list[dict[str, object]] = []
    start_mono_s = 0.0

    def record_sender(row: dict[str, object]) -> None:
        sender_rows.append(row)
        sender_logger.write(row)

    def record_host(row: dict[str, object]) -> None:
        host_rows.append(row)
        host_logger.write(row)

    host_sampler = None

    clock_sync = estimate_clock_offset(host=profile.depth_host, port=profile.control_port, timeout_s=0.4, samples=5)
    clock_offset_ns = 0 if clock_sync is None else int(clock_sync.offset_ns)
    write_manifest(
        context,
        extra={
            "clock_sync": None
            if clock_sync is None
            else {
                "offset_ns": int(clock_sync.offset_ns),
                "rtt_ms": float(clock_sync.rtt_ms),
                "sample_count": int(clock_sync.sample_count),
            },
            "compression_backend": "cpu",
            "gpu_interpretation": "GPU charts measure host load only; Zstd/LZ4 depth compression here is CPU-based.",
        },
    )

    dest = (profile.depth_host, int(profile.depth_port))
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(profile.socket_sndbuf))

    cal_pkt = build_cal_packet(profile.width, profile.height)
    last_cal_send = 0.0

    ffmpeg_cmd = make_ffmpeg_cmd(profile)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)
    compressor = DepthCompressor(profile.depth_comp)
    packet_bytes = max(HDR_DPT.size + 32, int(profile.depth_packet_bytes))
    max_payload = max(64, packet_bytes - HDR_DPT.size)

    print(f"[sender] profile={profile.name} run_id={context.run_id}")
    print(f"[sender] RTSP -> {profile.rtsp_url} ({profile.rtsp_transport})")
    print(f"[sender] Depth UDP -> {dest} comp={profile.depth_comp} packet_bytes={packet_bytes}")
    if clock_sync is not None:
        print(
            f"[sender] clock sync offset_ns={clock_sync.offset_ns} rtt_ms={clock_sync.rtt_ms:.3f} "
            f"samples={clock_sync.sample_count}"
        )
    else:
        print("[sender] clock sync unavailable; using sender wall clock timestamps.")

    rgb_camera_index = 0
    rgb_write_index = 0
    depth_seq = 0
    progress_last = 0.0
    duration_deadline = None

    try:
        with dai.Pipeline() as pipeline:
            cam_a = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam_nv12 = cam_a.requestOutput(
                (profile.width, profile.height),
                dai.ImgFrame.Type.NV12,
                dai.ImgResizeMode.CROP,
                profile.fps,
            )

            q_cam_rgb = None
            try:
                q_cam_rgb = cam_nv12.createOutputQueue(maxSize=2, blocking=False)
            except Exception:
                q_cam_rgb = None

            enc = pipeline.create(dai.node.VideoEncoder).build(
                cam_nv12,
                frameRate=profile.fps,
                profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            )
            try:
                enc.setNumBFrames(0)
            except Exception:
                pass
            q_vid = enc.out.createOutputQueue(maxSize=2, blocking=False)

            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            out_l = left.requestOutput((profile.width, profile.height), fps=profile.depth_fps)
            out_r = right.requestOutput((profile.width, profile.height), fps=profile.depth_fps)

            stereo = pipeline.create(dai.node.StereoDepth)
            try:
                stereo.setOutputSize(profile.width, profile.height)
            except Exception as exc:
                print("[sender] WARN: stereo.setOutputSize unavailable:", exc)

            stereo.setLeftRightCheck(True)
            if profile.extended_disparity:
                try:
                    stereo.setExtendedDisparity(True)
                except Exception as exc:
                    print("[sender] WARN: setExtendedDisparity unavailable:", exc)
            try:
                stereo.setSubpixel(bool(profile.subpixel))
            except Exception:
                pass

            configure_stereo_filters(
                stereo,
                min_mm=profile.min_mm,
                max_mm=profile.max_mm,
                force_decimation_1=bool(profile.force_decimation_1),
            )
            out_l.link(stereo.left)
            out_r.link(stereo.right)
            q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

            pipeline.start()
            start_mono_s = time.monotonic()
            duration_deadline = None if duration_sec <= 0 else start_mono_s + duration_sec
            progress_last = start_mono_s
            host_sampler = HostMetricsSampler(
                role="sender",
                nic_name=profile.nic_name_for_role("sender"),
                ping_host=profile.ping_host_for_role("sender"),
                write_row=record_host,
                start_mono_s=start_mono_s,
                interval_s=1.0,
            )
            host_sampler.start()
            next_depth_t = time.monotonic()
            depth_period = 1.0 / max(1.0, float(profile.depth_fps))

            while pipeline.isRunning() and not quit_event.is_set():
                now_mono = time.monotonic()
                if duration_deadline is not None and now_mono >= duration_deadline:
                    print("[sender] duration reached; stopping benchmark run.")
                    break

                if now_mono - last_cal_send >= float(profile.calib_period_sec):
                    try:
                        udp.sendto(cal_pkt, dest)
                        last_cal_send = now_mono
                    except Exception as exc:
                        print("[sender] WARN: failed to resend CAL0:", exc)

                if q_cam_rgb is not None:
                    cam_msg, cam_drained = drain_latest(q_cam_rgb)
                    if cam_msg is not None:
                        record_sender(
                            {
                                "timestamp_ns": time.time_ns(),
                                "elapsed_s": elapsed_s(start_mono_s),
                                "event": "rgb_camera",
                                "index": rgb_camera_index,
                                "drained_count": cam_drained,
                            }
                        )
                        rgb_camera_index += 1

                pkt, rgb_drained = drain_latest(q_vid)
                if pkt is not None:
                    if proc.poll() is not None:
                        print("[sender] ffmpeg exited unexpectedly.")
                        break
                    payload = pkt.getData().tobytes()
                    t0 = time.perf_counter()
                    try:
                        assert proc.stdin is not None
                        proc.stdin.write(payload)
                    except BrokenPipeError:
                        print("[sender] ffmpeg pipe closed.")
                        break
                    write_ms = (time.perf_counter() - t0) * 1000.0
                    record_sender(
                        {
                            "timestamp_ns": time.time_ns(),
                            "elapsed_s": elapsed_s(start_mono_s),
                            "event": "rgb_rtsp_write",
                            "index": rgb_write_index,
                            "ffmpeg_write_bytes": len(payload),
                            "ffmpeg_write_ms": write_ms,
                            "drained_count": rgb_drained,
                        }
                    )
                    rgb_write_index += 1

                if now_mono >= next_depth_t:
                    dmsg, depth_drained = drain_latest(q_depth)
                    if dmsg is not None:
                        depth = dmsg.getFrame().astype(np.uint16, copy=False)
                        raw = depth.tobytes(order="C")
                        t_comp0 = time.perf_counter()
                        payload, comp_id = compressor.compress(raw)
                        compression_ms = (time.perf_counter() - t_comp0) * 1000.0

                        source_ts_ns = time.time_ns() + int(clock_offset_ns)
                        count = max(1, (len(payload) + max_payload - 1) // max_payload)
                        t_send0 = time.perf_counter()
                        for idx in range(count):
                            chunk = payload[idx * max_payload : (idx + 1) * max_payload]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT,
                                depth_seq,
                                idx,
                                count,
                                source_ts_ns,
                                depth.shape[1],
                                depth.shape[0],
                                comp_id,
                                len(chunk),
                            )
                            udp.sendto(hdr + chunk, dest)
                        send_ms = (time.perf_counter() - t_send0) * 1000.0
                        record_sender(
                            {
                                "timestamp_ns": time.time_ns(),
                                "elapsed_s": elapsed_s(start_mono_s),
                                "event": "depth_send",
                                "seq": depth_seq,
                                "compression": compressor.mode,
                                "raw_bytes": len(raw),
                                "compressed_bytes": len(payload),
                                "compression_ms": compression_ms,
                                "send_ms": send_ms,
                                "packet_count": count,
                                "packet_bytes": packet_bytes,
                                "drained_count": depth_drained,
                                "source_ts_ns": source_ts_ns,
                                "clock_offset_ns": clock_offset_ns,
                            }
                        )
                        depth_seq = (depth_seq + 1) & 0xFFFFFFFF
                    next_depth_t += depth_period

                if time.monotonic() - progress_last >= 1.0:
                    depth_frames = sum(1 for row in sender_rows if row.get("event") == "depth_send")
                    rgb_frames = sum(1 for row in sender_rows if row.get("event") == "rgb_rtsp_write")
                    live_elapsed = max(1e-6, elapsed_s(start_mono_s))
                    print(
                        f"[sender] rgb_write_fps={rgb_frames / live_elapsed:.2f} "
                        f"depth_send_fps={depth_frames / live_elapsed:.2f}"
                    )
                    progress_last = time.monotonic()
                time.sleep(0.0005)
    finally:
        if host_sampler is not None:
            host_sampler.stop()
        sender_logger.close()
        host_logger.close()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            udp.close()
        except Exception:
            pass

    actual_duration_s = 0.0 if start_mono_s <= 0.0 else elapsed_s(start_mono_s)
    summary = {
        "role": "sender",
        "run_id": context.run_id,
        "profile_name": profile.name,
        "network_label": profile.network_label,
        "duration_s": actual_duration_s,
        "depth_frame_budget_ms": 1000.0 / max(1.0, float(profile.depth_fps)),
        "sender": summarize_sender_rows(sender_rows, actual_duration_s),
        "host": summarize_host_rows(host_rows),
        "compression_backend": "cpu",
        "gpu_interpretation": "GPU charts measure host load only; depth compression remains CPU-based.",
    }
    summary["bottleneck"] = classify_bottleneck(summary)
    write_json(context.run_dir / "summary.json", summary)
    if not summary["sender"].get("rgb_rtsp_write_fps"):
        print("[sender] WARN: no encoded RGB packets were written. Ensure MediaMTX is running and give the sender enough post-startup runtime.")
    if not summary["sender"].get("depth_send_fps"):
        print("[sender] WARN: no depth frames were sent. Check that the OAK-D pipeline is producing frames.")
    print(f"[sender] artifacts -> {context.run_dir}")
    print(f"[sender] bottleneck hint -> {summary['bottleneck']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
