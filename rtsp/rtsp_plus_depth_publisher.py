#!/usr/bin/env python3
import os, subprocess, signal, threading, time, socket, struct, zlib
import depthai as dai
import numpy as np
from datetime import timedelta

W, H = 640, 400
FPS = 30
DEPTH_FPS = 15

RTSP_URL = "rtsp://192.168.1.182:8554/oak"

DEPTH_HOST = "192.168.1.182"
DEPTH_PORT = 9000

# ---- UDP depth packet format ----
MAGIC = b"DPT0"
# magic(4) seq(u32) idx(u16) count(u16) ts_ns(u64) w(u16) h(u16) comp(u8) n(u16)
HDR = struct.Struct("<4sIHHQHHBH")
MTU = 1200
MAX_PAYLOAD = MTU - HDR.size

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

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

RTP_TICKS = 90000 // FPS  # 3000 for 30fps

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
    depth_seq = 0
    next_depth_t = 0.0

    with dai.Pipeline() as pipeline:
        platform = pipeline.getDefaultDevice().getPlatform()

        # ---- RGB camera (CAM_A) ----
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # Stream used for encoder (NV12 required by VideoEncoder) :contentReference[oaicite:2]{index=2}
        cam_nv12 = cam.requestOutput(
            (W, H),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            FPS
        )

        # Stream used as "align-to" reference (can be same size/fps)
        rgb_for_align = cam.requestOutput(size=(W, H), fps=FPS, enableUndistortion=True)

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_nv12,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )

        try: enc.setNumBFrames(0)
        except Exception: pass
        try: enc.setKeyframeFrequency(FPS)   # 1s GOP
        except Exception: pass
        try: enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.CBR)
        except Exception: pass
        try: enc.setBitrateKbps(1200)
        except Exception: pass

        q_vid = enc.out.createOutputQueue(maxSize=1, blocking=False)

        # ---- Stereo + depth aligned to RGB ---- (Luxonis depth-align pattern) :contentReference[oaicite:3]{index=3}
        left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth)

        # optional tuning
        try:
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
        except Exception:
            pass

        leftOut  = left.requestOutput(size=(W, H), fps=DEPTH_FPS)
        rightOut = right.requestOutput(size=(W, H), fps=DEPTH_FPS)

        leftOut.link(stereo.left)
        rightOut.link(stereo.right)

        if platform == dai.Platform.RVC4:
            align = pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            rgb_for_align.link(align.inputAlignTo)
            depth_stream = align.outputAligned
        else:
            # On RVC2: stereo can align using inputAlignTo (per Luxonis example)
            rgb_for_align.link(stereo.inputAlignTo)
            depth_stream = stereo.depth

        q_depth = depth_stream.createOutputQueue(maxSize=1, blocking=False)

        # Optional: reduce latency from chunking (0 disables chunking) :contentReference[oaicite:4]{index=4}
        try: pipeline.setXLinkChunkSize(0)
        except Exception: pass

        pipeline.start()

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                # ---- video ----
                pkt = q_vid.tryGet()
                if pkt is not None:
                    # freshest-packet-wins
                    while True:
                        newer = q_vid.tryGet()
                        if newer is None:
                            break
                        pkt = newer

                    if proc.poll() is not None:
                        print("ffmpeg exited (publish failed).")
                        break

                    try:
                        proc.stdin.write(pkt.getData().tobytes())
                    except BrokenPipeError:
                        print("ffmpeg pipe closed.")
                        break

                # ---- depth ---- (rate limited)
                now = time.monotonic()
                if now >= next_depth_t:
                    dmsg = q_depth.tryGet()
                    if dmsg is not None:
                        # freshest-depth-wins
                        while True:
                            newer = q_depth.tryGet()
                            if newer is None:
                                break
                            dmsg = newer

                        depth = dmsg.getFrame().astype(np.uint16, copy=False)
                        raw = depth.tobytes(order="C")
                        payload, comp_id = compress_depth(raw, depth_comp)

                        ts_ns = time.time_ns()
                        total = len(payload)
                        count = (total + MAX_PAYLOAD - 1) // MAX_PAYLOAD

                        for idx in range(count):
                            chunk = payload[idx*MAX_PAYLOAD:(idx+1)*MAX_PAYLOAD]
                            hdr = HDR.pack(MAGIC, depth_seq, idx, count, ts_ns, depth.shape[1], depth.shape[0], comp_id, len(chunk))
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
