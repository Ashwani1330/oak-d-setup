#!/usr/bin/env python3
import argparse, asyncio, struct, time, zlib
import depthai as dai
import numpy as np
import websockets

_HDR = struct.Struct("<cBIQHHI")

VIDEO_MJPEG = 1
VIDEO_H265  = 2
_COMP_NONE = 0
_COMP_ZSTD = 1
_COMP_LZ4  = 2
_COMP_ZLIB = 3

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
        return raw, _COMP_NONE
    if mode == "zstd" and HAS_ZSTD:
        return zstd.ZstdCompressor(level=1).compress(raw), _COMP_ZSTD
    if mode == "lz4" and HAS_LZ4:
        return lz4.frame.compress(raw, compression_level=0), _COMP_LZ4
    if mode == "zlib":
        return zlib.compress(raw, level=1), _COMP_ZLIB
    return raw, _COMP_NONE

def build_pipeline(w, h, fps, depth_fps, codec, mjpeg_q, h265_kbps, keyint, bframes):
    pipeline = dai.Pipeline()
    platform = pipeline.getDefaultDevice().getPlatform()

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    rgb_for_align = cam.requestOutput(size=(w,h), fps=fps, enableUndistortion=True)

    rgb_nv12 = cam.requestOutput(size=(w,h), fps=fps, type=dai.ImgFrame.Type.NV12)
    profile = dai.VideoEncoderProperties.Profile.MJPEG if codec == VIDEO_MJPEG else dai.VideoEncoderProperties.Profile.H265_MAIN
    enc = pipeline.create(dai.node.VideoEncoder).build(rgb_nv12, frameRate=float(fps), profile=profile)

    if codec == VIDEO_MJPEG:
        try: enc.setQuality(int(mjpeg_q))
        except Exception: pass
    else:
        try:
            enc.setBitrateKbps(int(h265_kbps))
            enc.setKeyframeFrequency(int(keyint))
            enc.setNumBFrames(int(bframes))  # 0 = lowest latency
        except Exception: pass

    qv = enc.out.createOutputQueue(maxSize=1, blocking=False)

    left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    try:
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
    except Exception:
        pass

    l = left.requestOutput(size=(w,h), fps=depth_fps)
    r = right.requestOutput(size=(w,h), fps=depth_fps)
    l.link(stereo.left); r.link(stereo.right)

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_for_align.link(align.inputAlignTo)
        depth_stream = align.outputAligned
    else:
        rgb_for_align.link(stereo.inputAlignTo)
        depth_stream = stereo.depth

    qd = depth_stream.createOutputQueue(maxSize=1, blocking=False)

    try: pipeline.setXLinkChunkSize(0)
    except Exception: pass

    return pipeline, qv, qd

async def run(args):
    codec = VIDEO_MJPEG if args.codec == "mjpeg" else VIDEO_H265
    pipeline, qv, qd = build_pipeline(args.width, args.height, args.fps, args.depth_fps,
                                      codec, args.mjpeg_quality, args.h265_bitrate_kbps,
                                      args.h265_keyint, args.h265_bframes)

    base = args.server.rstrip("/")
    async with websockets.connect(base + "/ws/video", max_size=None) as wsv, \
               websockets.connect(base + "/ws/depth", max_size=None) as wsd:
        with pipeline:
            pipeline.start()
            sv = 0
            sd = 0
            last_d = 0.0

            while pipeline.isRunning():
                enc = qv.tryGet()
                if enc is not None:
                    payload = bytes(enc.getData())
                    ts = time.time_ns()
                    hdr = _HDR.pack(b"V", codec, sv, ts, args.width, args.height, len(payload))
                    await wsv.send(hdr + payload)
                    sv = (sv + 1) & 0xFFFFFFFF

                now = time.monotonic()
                if args.depth_fps > 0 and (now - last_d) >= (1.0 / args.depth_fps):
                    d = qd.tryGet()
                    if d is not None:
                        dm = d.getFrame().astype(np.uint16, copy=False)
                        raw = dm.tobytes(order="C")
                        comp_payload, comp_id = compress_depth(raw, args.depth_comp)

                        ts = time.time_ns()
                        hdr = _HDR.pack(b"D", comp_id, sd, ts, dm.shape[1], dm.shape[0], len(comp_payload))
                        await wsd.send(hdr + comp_payload)
                        sd = (sd + 1) & 0xFFFFFFFF
                        last_d = now

                await asyncio.sleep(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="ws://CENTRAL_IP:8000")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--depth-fps", type=float, default=15.0)
    ap.add_argument("--codec", choices=["mjpeg","h265"], default="mjpeg")
    ap.add_argument("--mjpeg-quality", type=int, default=80)
    ap.add_argument("--h265-bitrate-kbps", type=int, default=2500)
    ap.add_argument("--h265-keyint", type=int, default=30)
    ap.add_argument("--h265-bframes", type=int, default=0)
    ap.add_argument("--depth-comp", choices=["none","zstd","lz4","zlib"], default="zstd")
    args = ap.parse_args()
    asyncio.run(run(args))
