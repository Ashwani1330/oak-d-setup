#!/usr/bin/env python3
from __future__ import annotations

import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import depthai as dai
except Exception:
    dai = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERA_EDGE_ROOT = Path(__file__).resolve().parent
if str(CAMERA_EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMERA_EDGE_ROOT))

from antropi_config import JETSON_SENDER_CONFIG
from runtime_profiles import RuntimeProfile, get_profile
from rgbd_protocol import CAL_FLOATS, HDR_CAL, HDR_DPT, MAGIC_CAL, MAGIC_DPT, UNITS_CM, DepthCompressor


quit_event = threading.Event()


def install_signals() -> None:
    quit_event.clear()
    signal.signal(signal.SIGINT, lambda *_: quit_event.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())


def elapsed_s(start_mono_s: float) -> float:
    return max(0.0, time.monotonic() - float(start_mono_s))


def make_ffmpeg_cmd(profile: RuntimeProfile) -> list[str]:
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
    if np is None or dai is None:
        raise RuntimeError("depthai and numpy are required to build the calibration packet on the sender.")
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


def print_config(profile: RuntimeProfile, duration_sec: float | None) -> None:
    print("Effective remote sender config:")
    for key, value in sorted(profile.to_dict().items()):
        print(f"  {key}: {value}")
    print(f"  effective_sender_nic_name: {profile.nic_name_for_role('sender')}")
    print("  benchmark_artifacts: disabled")
    print("  clock_sync_probe: disabled")
    print("  rgb_metadata_udp: disabled")
    print(f"  effective_duration_sec: {duration_sec if duration_sec is not None else 'unbounded'}")


def main() -> int:
    sender_cfg = JETSON_SENDER_CONFIG
    profile = get_profile(sender_cfg.profile_name)
    duration_sec = None if sender_cfg.run_duration_sec is None else float(sender_cfg.run_duration_sec)
    if sender_cfg.print_config_only:
        print_config(profile, duration_sec)
        return 0
    if np is None or dai is None:
        raise RuntimeError("depthai and numpy are required on the Jetson sender. Install the repo requirements first.")

    install_signals()

    udp = None
    proc = None
    start_mono_s = 0.0
    rgb_write_index = 0
    depth_frames = 0
    packet_bytes = max(HDR_DPT.size + 32, int(profile.depth_packet_bytes))
    max_payload = max(64, packet_bytes - HDR_DPT.size)
    dest = (profile.depth_host, int(profile.depth_port))

    try:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(profile.socket_sndbuf))

        cal_pkt = build_cal_packet(profile.width, profile.height)
        last_cal_send = 0.0

        ffmpeg_cmd = make_ffmpeg_cmd(profile)
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)
        compressor = DepthCompressor(profile.depth_comp)

        print(f"[sender] profile={profile.name}")
        print(f"[sender] RTSP -> {profile.rtsp_url} ({profile.rtsp_transport})")
        print(f"[sender] Depth UDP -> {dest} comp={profile.depth_comp} packet_bytes={packet_bytes}")
        print("[sender] RGB metadata disabled for normal workflow.")

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
            duration_deadline = None if duration_sec is None or duration_sec <= 0 else start_mono_s + duration_sec
            progress_last = start_mono_s
            next_depth_t = time.monotonic()
            depth_period = 1.0 / max(1.0, float(profile.depth_fps))

            while pipeline.isRunning() and not quit_event.is_set():
                now_mono = time.monotonic()
                if duration_deadline is not None and now_mono >= duration_deadline:
                    print("[sender] duration reached; stopping sender.")
                    break

                if now_mono - last_cal_send >= float(profile.calib_period_sec):
                    try:
                        udp.sendto(cal_pkt, dest)
                        last_cal_send = now_mono
                    except Exception as exc:
                        print(f"[sender] WARN: failed to resend CAL0: {exc}")

                if q_cam_rgb is not None:
                    drain_latest(q_cam_rgb)

                pkt, _rgb_drained = drain_latest(q_vid)
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
                    _write_ms = (time.perf_counter() - t0) * 1000.0
                    rgb_write_index += 1

                if now_mono >= next_depth_t:
                    dmsg, _depth_drained = drain_latest(q_depth)
                    if dmsg is not None:
                        depth = dmsg.getFrame().astype(np.uint16, copy=False)
                        raw = depth.tobytes(order="C")
                        payload, comp_id = compressor.compress(raw)
                        source_ts_ns = time.time_ns()
                        count = max(1, (len(payload) + max_payload - 1) // max_payload)
                        for idx in range(count):
                            chunk = payload[idx * max_payload : (idx + 1) * max_payload]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT,
                                depth_frames & 0xFFFFFFFF,
                                idx,
                                count,
                                source_ts_ns,
                                depth.shape[1],
                                depth.shape[0],
                                comp_id,
                                len(chunk),
                            )
                            udp.sendto(hdr + chunk, dest)
                        depth_frames += 1
                    next_depth_t += depth_period

                if time.monotonic() - progress_last >= 1.0:
                    live_elapsed = max(1e-6, elapsed_s(start_mono_s))
                    print(
                        f"[sender] rgb_write_fps={rgb_write_index / live_elapsed:.2f} "
                        f"depth_send_fps={depth_frames / live_elapsed:.2f}"
                    )
                    progress_last = time.monotonic()
                time.sleep(0.0005)
    finally:
        if proc is not None:
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
                proc.wait(timeout=3.0)
            except Exception:
                pass
        if udp is not None:
            try:
                udp.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
