#!/usr/bin/env python3
import argparse, subprocess, signal, threading, time, socket, struct, zlib
import depthai as dai
import numpy as np

# ============================================================
# QUICK DEBUG CONFIG (edit these, then run with no CLI args)
# ============================================================
DEFAULT = dict(
    W=640,
    H=400,
    FPS=30,
    DEPTH_FPS=15,

    RTSP_URL="rtsp://192.168.1.182:8554/oak",
    RTSP_TRANSPORT="udp",

    UDP_HOST="192.168.1.182",
    UDP_PORT=9000,

    DEPTH_COMP="zstd",  # none|zstd|lz4|zlib
    MIN_MM=120,
    MAX_MM=2500,

    CALIB_PERIOD_SEC=5.0,

    EXTENDED_DISPARITY=True,
    SUBPIXEL=False,
    FORCE_DECIMATION_1=True,   # prevent surprise half-res
)

# ---- UDP depth packet format ----
MAGIC_DPT = b"DPT0"
HDR_DPT = struct.Struct("<4sIHHQHHBH")   # magic, seq, idx, cnt, ts_ns, w, h, comp, payload_len
MTU = 1200
MAX_PAYLOAD = MTU - HDR_DPT.size

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

# ---- CAL0 packet ----
MAGIC_CAL = b"CAL0"
HDR_CAL = struct.Struct("<4sBQHH")       # magic, ver(u8), ts_ns(u64), w(u16), h(u16)
CAL_FLOATS = struct.Struct("<30fB")      # 30 floats + units byte
UNITS_CM = 1  # DepthAI extrinsics translation is in centimeters

# ---- VID0 packet (RGB metadata over UDP for server-side sync) ----
MAGIC_VID = b"VID0"
HDR_VID = struct.Struct("<4sBIQHH")      # magic, ver(u8), frame_id(u32), ts_ns(u64), w(u16), h(u16)

quit_event = threading.Event()

def add_bool_arg(ap: argparse.ArgumentParser, name: str, default: bool, help_txt: str):
    dest = name.replace("-", "_")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help_txt + f" (default={default})")
    g.add_argument(f"--no-{name}", dest=dest, action="store_false", help="Disable: " + help_txt)
    ap.set_defaults(**{dest: default})

def install_signals():
    signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

def timepoint_to_ns(t) -> int:
    """
    DepthAI returns timestamps as chrono time_points in C++.
    In Python, these often come as datetime.timedelta-like objects.
    """
    if t is None:
        return time.time_ns()
    if hasattr(t, "total_seconds"):
        return int(t.total_seconds() * 1e9)
    # Sometimes it's already an int-like
    try:
        return int(t)
    except Exception:
        return time.time_ns()

def get_device_ts_ns(msg) -> int:
    # Prefer device clock (monotonic, shared across streams on same OAK)
    for fn in ("getTimestampDevice", "getTimestamp"):
        if hasattr(msg, fn):
            f = getattr(msg, fn)
            try:
                return timepoint_to_ns(f())
            except TypeError:
                # some builds may require exposure offset; try MIDDLE if available
                try:
                    return timepoint_to_ns(f(dai.CameraExposureOffset.MIDDLE))
                except Exception:
                    pass
            except Exception:
                pass
    return time.time_ns()

def make_ffmpeg_cmd(rtsp_url: str, fps: int, transport: str):
    # RTSP timestamps are synthetic because we're piping elementary H264.
    # True sync is achieved via VID0+device timestamps.
    rtp_ticks = 90000 // fps
    return [
        "ffmpeg", "-nostdin", "-loglevel", "warning",
        "-f", "h264", "-i", "pipe:0",
        "-c:v", "copy",
        "-bsf:v", f"setts=ts=N*{rtp_ticks}:duration={rtp_ticks}:time_base=1/90000",
        "-flush_packets", "1",
        "-muxdelay", "0", "-muxpreload", "0",
        "-f", "rtsp",
        "-rtsp_transport", transport,
        "-pkt_size", str(MTU),
        rtsp_url,
    ]

class DepthCompressor:
    def __init__(self, mode: str):
        self.mode = mode
        self._zstd_c = None
        self._lz4 = None
        if mode == "zstd":
            try:
                import zstandard as zstd
                self._zstd_c = zstd.ZstdCompressor(level=1)
            except Exception:
                self.mode = "none"
        elif mode == "lz4":
            try:
                import lz4.frame
                self._lz4 = lz4.frame
            except Exception:
                self.mode = "none"
        elif mode == "zlib":
            pass
        else:
            self.mode = "none"

    def compress(self, raw: bytes) -> tuple[bytes, int]:
        if self.mode == "none":
            return raw, COMP_NONE
        if self.mode == "zstd" and self._zstd_c is not None:
            return self._zstd_c.compress(raw), COMP_ZSTD
        if self.mode == "lz4" and self._lz4 is not None:
            return self._lz4.compress(raw, compression_level=0), COMP_LZ4
        if self.mode == "zlib":
            return zlib.compress(raw, level=1), COMP_ZLIB
        return raw, COMP_NONE

def build_cal_packet(w: int, h: int) -> bytes:
    with dai.Device() as device:
        calib = device.readCalibration()
        K_rgb = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)
        K_right = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h), dtype=np.float32)
        T = np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A), dtype=np.float32)
        T_3x4 = T[:3, :4]
        payload_floats = list(K_rgb.reshape(-1)) + list(K_right.reshape(-1)) + list(T_3x4.reshape(-1))
        assert len(payload_floats) == 30

    ts_ns = time.time_ns()
    return HDR_CAL.pack(MAGIC_CAL, 1, ts_ns, w, h) + CAL_FLOATS.pack(*payload_floats, UNITS_CM)

def configure_stereo_filters(stereo: dai.node.StereoDepth, min_mm: int, max_mm: int, force_decimation_1: bool):
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
        except Exception as e:
            print("[WARN] cannot set decimationFactor=1:", e)

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

def main():
    ap = argparse.ArgumentParser(description="OAK-D RGB->RTSP + Depth->UDP sender (device-timestamped sync).")

    ap.add_argument("--w", type=int, default=DEFAULT["W"])
    ap.add_argument("--h", type=int, default=DEFAULT["H"])
    ap.add_argument("--fps", type=int, default=DEFAULT["FPS"])
    ap.add_argument("--depth-fps", type=int, default=DEFAULT["DEPTH_FPS"])

    ap.add_argument("--rtsp-url", default=DEFAULT["RTSP_URL"])
    ap.add_argument("--rtsp-transport", choices=["udp", "tcp"], default=DEFAULT["RTSP_TRANSPORT"])

    ap.add_argument("--udp-host", default=DEFAULT["UDP_HOST"])
    ap.add_argument("--udp-port", type=int, default=DEFAULT["UDP_PORT"])

    ap.add_argument("--depth-comp", choices=["none", "zstd", "lz4", "zlib"], default=DEFAULT["DEPTH_COMP"])
    ap.add_argument("--min-mm", type=int, default=DEFAULT["MIN_MM"])
    ap.add_argument("--max-mm", type=int, default=DEFAULT["MAX_MM"])
    ap.add_argument("--calib-period-sec", type=float, default=DEFAULT["CALIB_PERIOD_SEC"])

    add_bool_arg(ap, "extended-disparity", DEFAULT["EXTENDED_DISPARITY"], "Improve close range depth")
    add_bool_arg(ap, "subpixel", DEFAULT["SUBPIXEL"], "Improve depth precision (may add shimmer)")
    add_bool_arg(ap, "force-decimation-1", DEFAULT["FORCE_DECIMATION_1"], "Force decimationFactor=1 (avoid 320x200)")

    args = ap.parse_args()

    install_signals()

    DEST = (args.udp_host, args.udp_port)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    cal_pkt = build_cal_packet(args.w, args.h)
    last_cal_send = 0.0

    ffmpeg_cmd = make_ffmpeg_cmd(args.rtsp_url, args.fps, args.rtsp_transport)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    compressor = DepthCompressor(args.depth_comp)
    depth_seq = 0

    print("[client] RTSP:", args.rtsp_url, "transport:", args.rtsp_transport)
    print("[client] UDP:", DEST, "comp:", args.depth_comp, "size:", args.w, "x", args.h)

    with dai.Pipeline() as pipeline:
        # RGB (CAM_A) -> H264 -> ffmpeg -> RTSP
        camA = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_nv12 = camA.requestOutput((args.w, args.h), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, args.fps)

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_nv12, frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
        )
        try:
            enc.setNumBFrames(0)  # reduce latency / avoid reordering
        except Exception:
            pass
        q_vid = enc.out.createOutputQueue(maxSize=2, blocking=False)

        # Stereo depth (CAM_B/C)
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        outL = left.requestOutput((args.w, args.h), fps=args.depth_fps)
        outR = right.requestOutput((args.w, args.h), fps=args.depth_fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        try:
            stereo.setOutputSize(args.w, args.h)
        except Exception as e:
            print("[WARN] stereo.setOutputSize not available:", e)

        stereo.setLeftRightCheck(True)

        if args.extended_disparity:
            try:
                stereo.setExtendedDisparity(True)
            except Exception as e:
                print("[WARN] setExtendedDisparity not available:", e)

        try:
            stereo.setSubpixel(bool(args.subpixel))
        except Exception:
            pass

        configure_stereo_filters(stereo, args.min_mm, args.max_mm, bool(args.force_decimation_1))

        outL.link(stereo.left)
        outR.link(stereo.right)

        q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()

        next_depth_t = time.monotonic()
        depth_period = 1.0 / max(1.0, float(args.depth_fps))

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                now = time.monotonic()

                # CAL0 resend
                if now - last_cal_send >= args.calib_period_sec:
                    try:
                        udp.sendto(cal_pkt, DEST)
                        last_cal_send = now
                    except Exception:
                        pass

                # VIDEO -> send VID0(meta) + push H264 to ffmpeg
                pkt = q_vid.tryGet()
                if pkt is not None:
                    # keep newest only
                    while True:
                        newer = q_vid.tryGet()
                        if newer is None:
                            break
                        pkt = newer

                    if proc.poll() is not None:
                        print("[client] ffmpeg exited.")
                        break

                    # --- NEW: VID0 metadata with DEVICE timestamp ---
                    ts_ns = get_device_ts_ns(pkt)
                    try:
                        frame_id = int(pkt.getSequenceNum()) & 0xFFFFFFFF
                    except Exception:
                        frame_id = 0
                    vid_pkt = HDR_VID.pack(MAGIC_VID, 1, frame_id, ts_ns, args.w, args.h)
                    try:
                        udp.sendto(vid_pkt, DEST)
                    except Exception:
                        pass

                    # H264 payload
                    try:
                        proc.stdin.write(pkt.getData().tobytes())
                    except BrokenPipeError:
                        print("[client] ffmpeg pipe closed.")
                        break

                # DEPTH -> UDP (DEVICE timestamp)
                if now >= next_depth_t:
                    dmsg = q_depth.tryGet()
                    if dmsg is not None:
                        while True:
                            newer = q_depth.tryGet()
                            if newer is None:
                                break
                            dmsg = newer

                        depth = dmsg.getFrame().astype(np.uint16, copy=False)

                        if (depth_seq % 30) == 0:
                            print("[client] DEPTH OUTPUT:", depth.shape[1], "x", depth.shape[0])

                        raw = depth.tobytes(order="C")
                        payload, comp_id = compressor.compress(raw)

                        ts_ns = get_device_ts_ns(dmsg)  # --- NEW: device timestamp ---
                        total = len(payload)
                        count = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD

                        for idx in range(count):
                            chunk = payload[idx * MAX_PAYLOAD: (idx + 1) * MAX_PAYLOAD]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT, depth_seq, idx, count, ts_ns,
                                depth.shape[1], depth.shape[0], comp_id, len(chunk)
                            )
                            udp.sendto(hdr + chunk, DEST)

                        depth_seq = (depth_seq + 1) & 0xFFFFFFFF

                    next_depth_t += depth_period

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
