#!/usr/bin/env python3
import argparse, subprocess, signal, threading, time, socket, struct, zlib
import numpy as np
import depthai as dai

# ---------------- UDP depth packet format ----------------
MAGIC_DPT = b"DPT0"
# magic(4) seq(u32) idx(u16) cnt(u16) ts_ns(u64) w(u16) h(u16) comp(u8) n(u16)
HDR_DPT = struct.Struct("<4sIHHQHHBH")

MAGIC_CAL = b"CAL0"
# magic(4) ver(u8) ts_ns(u64) w(u16) h(u16)  (then 30 floats)  units(u8)
HDR_CAL = struct.Struct("<4sBQHH")
CAL_FLOATS = struct.Struct("<30fB")

UNITS_CM = 1  # getCameraExtrinsics translation is in centimeters (DepthAI calibration API docs).  # noqa

MTU = 1200
MAX_PAYLOAD = MTU - HDR_DPT.size

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

try:
    import zstandard as zstd
    HAS_ZSTD = True
    ZSTD_C = zstd.ZstdCompressor(level=1)  # build once (important on Jetson)
except Exception:
    HAS_ZSTD = False
    ZSTD_C = None

try:
    import lz4.frame
    HAS_LZ4 = True
except Exception:
    HAS_LZ4 = False


def compress_depth(raw: bytes, mode: str):
    if mode == "none":
        return raw, COMP_NONE
    if mode == "lz4" and HAS_LZ4:
        return lz4.frame.compress(raw, compression_level=0), COMP_LZ4
    if mode == "zstd" and HAS_ZSTD:
        return ZSTD_C.compress(raw), COMP_ZSTD
    if mode == "zlib":
        return zlib.compress(raw, level=1), COMP_ZLIB
    return raw, COMP_NONE


def pick_enum(enum_obj, *names):
    for n in names:
        v = getattr(enum_obj, n, None)
        if v is not None:
            return v
    return None


def send_calibration_udp(udp: socket.socket, host: str, port: int,
                         calib: dai.CalibrationHandler, w: int, h: int,
                         repeats: int = 3):
    """
    Sends K_rgb, K_right, and T_right_to_rgb (3x4) as CAL0.
    """
    K_rgb = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)
    K_right = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h), dtype=np.float32)

    T = np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A),
                 dtype=np.float32)  # 4x4, translation in cm
    T_3x4 = T[:3, :4]

    payload_floats = list(K_rgb.reshape(-1)) + list(K_right.reshape(-1)) + list(T_3x4.reshape(-1))
    assert len(payload_floats) == 30

    ts_ns = time.time_ns()
    pkt = HDR_CAL.pack(MAGIC_CAL, 1, ts_ns, w, h) + CAL_FLOATS.pack(*payload_floats, UNITS_CM)

    for _ in range(repeats):
        udp.sendto(pkt, (host, port))
        time.sleep(0.02)


def configure_stereo_depth_filters(stereo: dai.node.StereoDepth,
                                  min_mm: int, max_mm: int,
                                  temporal: bool, spatial: bool, speckle: bool):
    """
    On-device depth cleanup (StereoDepth post-processing).
    Uses the documented Depth post-processing knobs: median/speckle/temporal/spatial/threshold/decimation.
    """
    # Pick a preset that exists across versions.
    preset = pick_enum(dai.node.StereoDepth.PresetMode,
                       "FAST_ACCURACY", "DEFAULT", "FAST_DENSITY", "HIGH_DENSITY", "HIGH_DETAIL")
    if preset is not None:
        stereo.setDefaultProfilePreset(preset)

    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)  # keep latency + compute low

    # Hardware median is the cheapest denoise
    try:
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    except Exception:
        pass

    # Confidence threshold can reduce flying pixels (tune 180–240)
    try:
        stereo.initialConfig.setConfidenceThreshold(200)
    except Exception:
        pass

    # Post-processing block (device-side)
    cfg = stereo.initialConfig

    try:
        cfg.postProcessing.thresholdFilter.minRange = int(min_mm)
        cfg.postProcessing.thresholdFilter.maxRange = int(max_mm)

        cfg.postProcessing.decimationFilter.decimationFactor = 1

        cfg.postProcessing.speckleFilter.enable = bool(speckle)
        if speckle:
            cfg.postProcessing.speckleFilter.speckleRange = 50

        cfg.postProcessing.temporalFilter.enable = bool(temporal)
        if temporal:
            cfg.postProcessing.temporalFilter.alpha = 0.7  # 1.0=no filter; lower=more smoothing
            pm = pick_enum(cfg.postProcessing.temporalFilter.PersistencyMode,
                           "VALID_2_IN_LAST_4", "VALID_1_IN_LAST_5", "VALID_2_IN_LAST_3")
            if pm is not None:
                cfg.postProcessing.temporalFilter.persistencyMode = pm

        cfg.postProcessing.spatialFilter.enable = bool(spatial)
        if spatial:
            cfg.postProcessing.spatialFilter.holeFillingRadius = 2
            cfg.postProcessing.spatialFilter.numIterations = 1
    except Exception:
        # If a specific build lacks some knobs, we still run with what is available.
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth-fps", type=int, default=15)

    ap.add_argument("--rtsp-url", default="rtsp://192.168.1.182:8554/oak")
    ap.add_argument("--depth-host", default="192.168.1.182")
    ap.add_argument("--depth-port", type=int, default=9000)

    ap.add_argument("--depth-comp", choices=["none", "lz4", "zstd", "zlib"], default="lz4")

    ap.add_argument("--min-mm", type=int, default=300)
    ap.add_argument("--max-mm", type=int, default=4000)

    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-speckle", action="store_true")

    ap.add_argument("--bitrate-kbps", type=int, default=1200)
    args = ap.parse_args()

    W, H = args.w, args.h
    FPS = args.fps
    DEPTH_FPS = args.depth_fps

    quit_event = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

    # --- ffmpeg publishes RTSP (expects RTSP server like mediamtx running on rtsp-url host) ---
    rtp_ticks = 90000 // max(1, FPS)
    ffmpeg_cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "warning",
        "-f", "h264", "-i", "pipe:0",
        "-c:v", "copy",
        "-bsf:v", f"setts=ts=N*{rtp_ticks}:duration={rtp_ticks}:time_base=1/90000",
        "-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0",
        "-f", "rtsp", "-rtsp_transport", "udp", "-pkt_size", "1200",
        args.rtsp_url,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    except Exception:
        pass

    depth_seq = 0
    next_depth_t = 0.0
    last_cal_resend = 0.0

    with dai.Pipeline() as pipeline:
        dev = pipeline.getDefaultDevice()
        calib = dev.readCalibration()

        # Send CAL0 immediately (and we’ll re-send periodically)
        send_calibration_udp(udp, args.depth_host, args.depth_port, calib, W, H, repeats=5)
        last_cal_resend = time.monotonic()

        # ---------------- RGB (CAM_A) -> H264 -> ffmpeg stdin ----------------
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # v3 requestOutput signature (size/type/resize_mode/fps)
        cam_nv12 = cam.requestOutput(size=(W, H),
                                     type=dai.ImgFrame.Type.NV12,
                                     resize_mode=dai.ImgResizeMode.CROP,
                                     fps=FPS)

        # Also produce an RGB reference stream for alignment (same size)
        rgb_for_align = cam.requestOutput(size=(W, H), fps=FPS)

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_nv12, frameRate=FPS, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
        )

        try: enc.setNumBFrames(0)
        except Exception: pass
        try: enc.setKeyframeFrequency(int(FPS))
        except Exception: pass
        try: enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.CBR)
        except Exception: pass
        try: enc.setBitrateKbps(int(args.bitrate_kbps))
        except Exception: pass

        q_vid = enc.out.createOutputQueue(maxSize=1, blocking=False)

        # ---------------- Stereo (CAM_B + CAM_C) -> depth (aligned) ----------------
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_out  = left.requestOutput(size=(W, H), fps=DEPTH_FPS)
        right_out = right.requestOutput(size=(W, H), fps=DEPTH_FPS)

        stereo = pipeline.create(dai.node.StereoDepth)
        configure_stereo_depth_filters(
            stereo,
            min_mm=args.min_mm,
            max_mm=args.max_mm,
            temporal=not args.no_temporal,
            spatial=not args.no_spatial,
            speckle=not args.no_speckle,
        )

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # Depth alignment:
        # - RVC4: use ImageAlign node
        # - RVC2 (OAK-D): use StereoDepth.inputAlignTo
        depth_aligned_out = stereo.depth
        try:
            if dev.getPlatform() == dai.Platform.RVC4:
                align = pipeline.create(dai.node.ImageAlign)
                stereo.depth.link(align.input)
                rgb_for_align.link(align.inputAlignTo)
                depth_aligned_out = align.outputAligned
            else:
                rgb_for_align.link(stereo.inputAlignTo)
                depth_aligned_out = stereo.depth
        except Exception:
            # Fallback: no alignment (still works; coloring will be approximate)
            depth_aligned_out = stereo.depth

        q_depth = depth_aligned_out.createOutputQueue(maxSize=1, blocking=False)

        try:
            pipeline.setXLinkChunkSize(0)
        except Exception:
            pass

        pipeline.start()

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                # ---- VIDEO freshest-wins ----
                vp = q_vid.tryGet()
                if vp is not None:
                    while True:
                        newer = q_vid.tryGet()
                        if newer is None:
                            break
                        vp = newer

                    if proc.poll() is not None:
                        print("ffmpeg exited.")
                        break
                    try:
                        proc.stdin.write(vp.getData().tobytes())
                    except BrokenPipeError:
                        print("ffmpeg pipe closed.")
                        break

                # ---- DEPTH freshest-wins (rate limited) ----
                now = time.monotonic()
                if now >= next_depth_t:
                    dp = q_depth.tryGet()
                    if dp is not None:
                        while True:
                            newer = q_depth.tryGet()
                            if newer is None:
                                break
                            dp = newer

                        depth = dp.getFrame()  # uint16 mm
                        raw = depth.tobytes(order="C")
                        payload, comp_id = compress_depth(raw, args.depth_comp)

                        ts_ns = time.time_ns()
                        total = len(payload)
                        cnt = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD

                        for idx in range(cnt):
                            chunk = payload[idx*MAX_PAYLOAD:(idx+1)*MAX_PAYLOAD]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT, depth_seq, idx, cnt, ts_ns,
                                depth.shape[1], depth.shape[0], comp_id, len(chunk)
                            )
                            udp.sendto(hdr + chunk, (args.depth_host, args.depth_port))

                        depth_seq = (depth_seq + 1) & 0xFFFFFFFF

                    next_depth_t = now + (1.0 / max(1.0, DEPTH_FPS))

                # resend CAL0 periodically so the viewer can start anytime
                if (now - last_cal_resend) > 2.0:
                    send_calibration_udp(udp, args.depth_host, args.depth_port, calib, W, H, repeats=1)
                    last_cal_resend = now

                time.sleep(0.0005)

        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            pipeline.stop()
            pipeline.wait()


if __name__ == "__main__":
    main()
