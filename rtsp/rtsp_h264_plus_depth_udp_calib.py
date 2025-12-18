#!/usr/bin/env python3
import subprocess, signal, threading, time, socket, struct, zlib
import depthai as dai
import numpy as np

W, H = 640, 400
FPS = 30
DEPTH_FPS = 30

RTSP_URL = "rtsp://192.168.1.182:8554/oak"
DEPTH_HOST = "192.168.1.182"
DEPTH_PORT = 9000

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

# ---- UDP depth packet format ----
MAGIC_DPT = b"DPT0"
HDR_DPT = struct.Struct("<4sIHHQHHBH")
MTU = 1200
MAX_PAYLOAD = MTU - HDR_DPT.size

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False

try:
    import lz4.frame
    HAS_LZ4 = True
except Exception:
    HAS_LZ4 = False

def compress_depth(raw: bytes, mode: str):
    if mode == "none":
        return raw, COMP_NONE
    if mode == "zstd" and HAS_ZSTD:
        return zstd.ZstdCompressor(level=1).compress(raw), COMP_ZSTD
    if mode == "lz4" and HAS_LZ4:
        return lz4.frame.compress(raw, compression_level=0), COMP_LZ4
    if mode == "zlib":
        return zlib.compress(raw, level=1), COMP_ZLIB
    return raw, COMP_NONE

# ---- CAL0 packet ----
MAGIC_CAL = b"CAL0"
HDR_CAL = struct.Struct("<4sBQHH")
CAL_FLOATS = struct.Struct("<30fB")
UNITS_CM = 1  # depthai extrinsics translation is in centimeters :contentReference[oaicite:2]{index=2}

def send_calibration_udp(udp: socket.socket, host: str, port: int, w: int, h: int, repeats: int = 5):
    """
    Sends K_rgb, K_right, and T_right_to_rgb once at startup.
    Uses depthai calibration APIs: readCalibration/getCameraIntrinsics/getCameraExtrinsics. :contentReference[oaicite:3]{index=3}
    """
    with dai.Device() as device:
        calib = device.readCalibration()

        K_rgb = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)  # RGB
        K_right = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w, h), dtype=np.float32)  # right mono

        # 4x4, translation in cm :contentReference[oaicite:4]{index=4}
        T = np.array(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A), dtype=np.float32)
        T_3x4 = T[:3, :4]

        payload_floats = list(K_rgb.reshape(-1)) + list(K_right.reshape(-1)) + list(T_3x4.reshape(-1))
        assert len(payload_floats) == 30

        ts_ns = time.time_ns()
        pkt = HDR_CAL.pack(MAGIC_CAL, 1, ts_ns, w, h) + CAL_FLOATS.pack(*payload_floats, UNITS_CM)

        for _ in range(repeats):
            udp.sendto(pkt, (host, port))
            time.sleep(0.05)

RTP_TICKS = 90000 // FPS

ffmpeg_cmd = [
    "ffmpeg", "-nostdin", "-loglevel", "warning",
    "-f", "h264", "-i", "pipe:0",
    "-c:v", "copy",
    "-bsf:v", f"setts=ts=N*{RTP_TICKS}:duration={RTP_TICKS}:time_base=1/90000",
    "-flush_packets", "1",
    "-muxdelay", "0", "-muxpreload", "0",
    "-f", "rtsp",
    "-rtsp_transport", "udp",
    "-pkt_size", "1200",
    RTSP_URL,
]

def main(depth_comp="zstd"):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_calibration_udp(udp, DEPTH_HOST, DEPTH_PORT, W, H, repeats=5)

    depth_seq = 0
    next_depth_t = 0.0

    with dai.Pipeline() as pipeline:
        # ---- RGB camera (CAM_A) ----
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        cam_nv12 = cam.requestOutput((W, H), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, FPS)

        # Use an RGB output as the “align-to” reference; keep it the same size as your stream.
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
        try: enc.setBitrateKbps(1200)
        except Exception: pass

        q_vid = enc.out.createOutputQueue(maxSize=1, blocking=False)

        # ---- Stereo depth ----
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth)

        try:
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
        except Exception:
            pass

        leftOut  = left.requestOutput(size=(W, H), fps=DEPTH_FPS)
        rightOut = right.requestOutput(size=(W, H), fps=DEPTH_FPS)
        leftOut.link(stereo.left)
        rightOut.link(stereo.right)

        # Align depth to RGB (and keep depth output size sane if you ever align to higher-res RGB) :contentReference[oaicite:5]{index=5}
        try:
            rgb_for_align.link(stereo.inputAlignTo)
            depth_stream = stereo.depth
        except Exception:
            depth_stream = stereo.depth

        q_depth = depth_stream.createOutputQueue(maxSize=1, blocking=False)

        try: pipeline.setXLinkChunkSize(0)
        except Exception: pass

        pipeline.start()

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                # --- VIDEO (H264 -> ffmpeg pipe) ---
                pkt = q_vid.tryGet()
                if pkt is not None:
                    while True:
                        newer = q_vid.tryGet()
                        if newer is None: break
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
                now = time.monotonic()
                if now >= next_depth_t:
                    dmsg = q_depth.tryGet()
                    if dmsg is not None:
                        while True:
                            newer = q_depth.tryGet()
                            if newer is None: break
                            dmsg = newer

                        depth = dmsg.getFrame().astype(np.uint16, copy=False)
                        raw = depth.tobytes(order="C")
                        payload, comp_id = compress_depth(raw, depth_comp)

                        ts_ns = time.time_ns()
                        total = len(payload)
                        count = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD

                        for idx in range(count):
                            chunk = payload[idx*MAX_PAYLOAD:(idx+1)*MAX_PAYLOAD]
                            hdr = HDR_DPT.pack(
                                MAGIC_DPT, depth_seq, idx, count, ts_ns,
                                depth.shape[1], depth.shape[0], comp_id, len(chunk)
                            )
                            udp.sendto(hdr + chunk, (DEPTH_HOST, DEPTH_PORT))

                        depth_seq = (depth_seq + 1) & 0xFFFFFFFF

                    next_depth_t = now + (1.0 / max(1.0, DEPTH_FPS))

                time.sleep(0.0005)

        finally:
            try: proc.stdin.close()
            except Exception: pass
            proc.terminate()
            pipeline.stop()
            pipeline.wait()

if __name__ == "__main__":
    # depth_comp: none | zstd | lz4 | zlib
    main(depth_comp="zstd")
