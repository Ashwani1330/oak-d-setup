#!/usr/bin/env python3
import argparse, subprocess, signal, threading, time, socket, struct, zlib
import depthai as dai
import numpy as np

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
UNITS_CM = 1  # depthai extrinsics translation is in centimeters

quit_event = threading.Event()

def install_signals():
    signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
    signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

def make_ffmpeg_cmd(rtsp_url: str, fps: int, transport: str):
    rtp_ticks = 90000 // fps
    return [
        "ffmpeg", "-nostdin", "-loglevel", "warning",
        "-f", "h264", "-i", "pipe:0",
        "-c:v", "copy",
        # force monotonic timestamps for RTP/RTSP
        "-bsf:v", f"setts=ts=N*{rtp_ticks}:duration={rtp_ticks}:time_base=1/90000",
        "-flush_packets", "1",
        "-muxdelay", "0", "-muxpreload", "0",
        "-f", "rtsp",
        "-rtsp_transport", transport,  # udp or tcp
        "-pkt_size", str(MTU),
        rtsp_url,
    ]

class DepthCompressor:
    def __init__(self, mode: str):
        self.mode = mode
        self.comp_id = COMP_NONE
        self._zstd_c = None
        self._lz4 = None

        if mode == "zstd":
            try:
                import zstandard as zstd
                self._zstd_c = zstd.ZstdCompressor(level=1)
                self.comp_id = COMP_ZSTD
            except Exception:
                self.mode = "none"
                self.comp_id = COMP_NONE

        elif mode == "lz4":
            try:
                import lz4.frame
                self._lz4 = lz4.frame
                self.comp_id = COMP_LZ4
            except Exception:
                self.mode = "none"
                self.comp_id = COMP_NONE

        elif mode == "zlib":
            self.comp_id = COMP_ZLIB

        else:
            self.mode = "none"
            self.comp_id = COMP_NONE

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
    # Read calibration once (make sure you’re reading from the correct device if multiple OAKs exist!)
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

def configure_stereo_filters(stereo: dai.node.StereoDepth, min_mm: int, max_mm: int):
    """
    Uses DepthAI StereoDepth postProcessing fields as documented in Luxonis examples:
    speckle/temporal/spatial/threshold/decimation etc.
    """
    try:
        # A good “robotics-ish” baseline preset (if available)
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        except Exception:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        cfg = stereo.initialConfig.get()
        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 50

        cfg.postProcessing.temporalFilter.enable = True
        # alpha: 1 = no filter, lower = more smoothing
        try:
            cfg.postProcessing.temporalFilter.alpha = 0.4
        except Exception:
            pass

        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1

        # This is the “remove walls/background by range” hammer:
        cfg.postProcessing.thresholdFilter.minRange = int(min_mm)
        cfg.postProcessing.thresholdFilter.maxRange = int(max_mm)

        # Keep decimationFactor=1 unless you intentionally want heavier downsampling
        cfg.postProcessing.decimationFilter.decimationFactor = 1

        stereo.initialConfig.set(cfg)
    except Exception as e:
        print(f"[WARN] Stereo post-processing config not applied: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth-fps", type=int, default=15)
    ap.add_argument("--rtsp-url", required=True)
    ap.add_argument("--rtsp-transport", choices=["udp","tcp"], default="udp")
    ap.add_argument("--depth-host", required=True)
    ap.add_argument("--depth-port", type=int, required=True)
    ap.add_argument("--depth-comp", choices=["none","zstd","lz4","zlib"], default="zstd")
    ap.add_argument("--min-mm", type=int, default=300)
    ap.add_argument("--max-mm", type=int, default=4000)
    ap.add_argument("--calib-period-sec", type=float, default=5.0, help="resend CAL0 every N seconds")
    args = ap.parse_args()

    install_signals()

    # UDP socket
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    udp.connect((args.depth_host, args.depth_port))

    cal_pkt = build_cal_packet(args.w, args.h)
    last_cal_send = 0.0

    # ffmpeg publisher
    ffmpeg_cmd = make_ffmpeg_cmd(args.rtsp_url, args.fps, args.rtsp_transport)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    compressor = DepthCompressor(args.depth_comp)
    depth_seq = 0

    with dai.Pipeline() as pipeline:
        # ---- RGB camera (CAM_A) ----
        camA = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_nv12 = camA.requestOutput((args.w, args.h), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, args.fps)

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_nv12, frameRate=args.fps, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
        )
        q_vid = enc.out.createOutputQueue(maxSize=2, blocking=False)

        # ---- Stereo depth (CAM_B/C) ----
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        outL = left.requestOutput((args.w, args.h), fps=args.depth_fps)
        outR = right.requestOutput((args.w, args.h), fps=args.depth_fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)  # disable if you prefer less shimmer / faster response
        configure_stereo_filters(stereo, args.min_mm, args.max_mm)

        outL.link(stereo.left)
        outR.link(stereo.right)

        depth_stream = stereo.depth
        q_depth = depth_stream.createOutputQueue(maxSize=2, blocking=False)

        pipeline.start()

        next_depth_t = time.monotonic()
        depth_period = 1.0 / max(1.0, float(args.depth_fps))

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                # Periodic calibration resend
                now = time.monotonic()
                if now - last_cal_send >= args.calib_period_sec:
                    try:
                        udp.send(cal_pkt)
                        last_cal_send = now
                    except Exception:
                        pass

                # --- VIDEO (H264 -> ffmpeg pipe) ---
                pkt = q_vid.tryGet()
                if pkt is not None:
                    # keep newest only
                    while True:
                        newer = q_vid.tryGet()
                        if newer is None:
                            break
                        pkt = newer

                    if proc.poll() is not None:
                        print("ffmpeg exited.")
                        break
                    try:
                        proc.stdin.write(pkt.getData().tobytes())
                    except BrokenPipeError:
                        print("ffmpeg pipe closed.")
                        break

                # --- DEPTH (UDP) ---
                if now >= next_depth_t:
                    dmsg = q_depth.tryGet()
                    if dmsg is not None:
                        while True:
                            newer = q_depth.tryGet()
                            if newer is None:
                                break
                            dmsg = newer

                        depth = dmsg.getFrame().astype(np.uint16, copy=False)
                        raw = depth.tobytes(order="C")
                        payload, comp_id = compressor.compress(raw)

                        ts_ns = time.time_ns()
                        total = len(payload)
                        count = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD

                        for idx in range(count):
                            chunk = payload[idx * MAX_PAYLOAD : (idx + 1) * MAX_PAYLOAD]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT, depth_seq, idx, count, ts_ns,
                                depth.shape[1], depth.shape[0], comp_id, len(chunk)
                            )
                            udp.send(hdr + chunk)

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
