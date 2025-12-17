#!/usr/bin/env python3
import argparse
import socket
import struct
import time

import depthai as dai
import numpy as np

MAGIC = b"DEP0"
MAGIC_CAL = b"CAL0"
VERSION = 1

# magic(4) ver(u8) flags(u8) reserved(u16)
# frame_id(u32) w(u16) h(u16) chunks(u16) chunk_idx(u16)
# timestamp_ns(u64)
HDR_FMT = "!4sBBH I H H H H Q"
HDR_SIZE = struct.calcsize(HDR_FMT)

# CAL packet:
# magic(4) ver(u8) reserved(u8) w(u16) h(u16) fx(f32) fy(f32) cx(f32) cy(f32)
CAL_FMT = "!4sBB H H f f f f"
CAL_SIZE = struct.calcsize(CAL_FMT)


def get_intrinsics_from_pipeline_device(pipeline: dai.Pipeline, socket_name: dai.CameraBoardSocket, w: int, h: int):
    # v3: Pipeline has a live device handle that can be queried during creation/use. :contentReference[oaicite:3]{index=3}
    calib = pipeline.getDevice().readCalibration()
    K = np.array(calib.getCameraIntrinsics(socket_name, w, h), dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", required=True)
    ap.add_argument("--server-port", type=int, default=5005)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=1400)
    ap.add_argument("--send-cal-every", type=float, default=2.0)

    # OAK-D typical mapping: CAM_B left mono, CAM_C right mono, CAM_A RGB
    ap.add_argument("--left-socket", default="CAM_B")
    ap.add_argument("--right-socket", default="CAM_C")
    ap.add_argument("--align-socket", default="CAM_A")
    args = ap.parse_args()

    server_addr = (args.server_ip, args.server_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)

    left_socket = getattr(dai.CameraBoardSocket, args.left_socket)
    right_socket = getattr(dai.CameraBoardSocket, args.right_socket)
    align_socket = getattr(dai.CameraBoardSocket, args.align_socket)

    pipeline = dai.Pipeline()

    monoLeft = pipeline.create(dai.node.Camera).build(left_socket)   # v3 Camera node build(socket) :contentReference[oaicite:4]{index=4}
    monoRight = pipeline.create(dai.node.Camera).build(right_socket)

    stereo = pipeline.create(dai.node.StereoDepth)

    # Your build: FAST_ACCURACY exists; HIGH_ACCURACY doesn't.
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)

    stereo.setSubpixel(True)              # long-range precision; see StereoDepth options :contentReference[oaicite:5]{index=5}
    stereo.setExtendedDisparity(False)    # short-range mode off
    stereo.setLeftRightCheck(True)        # better occlusion handling (costs some compute) :contentReference[oaicite:6]{index=6}
    stereo.setDepthAlign(align_socket)

    # Luxonis v3 stereo example uses requestFullResolutionOutput() and links to stereo. :contentReference[oaicite:7]{index=7}
    leftOut = monoLeft.requestFullResolutionOutput()
    rightOut = monoRight.requestFullResolutionOutput()
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    # v3: queues come directly from outputs (no XLinkOut). :contentReference[oaicite:8]{index=8}
    depthQueue = stereo.depth.createOutputQueue(maxSize=4, blocking=True)

    frame_id = 0
    last_cal_sent = 0.0

    with pipeline:
        pipeline.start()  # v3 replacement for dai.Device(pipeline). :contentReference[oaicite:9]{index=9}

        fx, fy, cx, cy = get_intrinsics_from_pipeline_device(pipeline, align_socket, args.w, args.h)

        print(f"[sender] depth -> udp://{server_addr[0]}:{server_addr[1]}")
        print(f"[sender] sockets L={args.left_socket} R={args.right_socket} align={args.align_socket}")
        print(f"[sender] intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

        while pipeline.isRunning():
            now = time.monotonic()
            if now - last_cal_sent >= args.send_cal_every:
                cal_pkt = struct.pack(CAL_FMT, MAGIC_CAL, VERSION, 0, args.w, args.h, fx, fy, cx, cy)
                sock.sendto(cal_pkt, server_addr)
                last_cal_sent = now

            msg = depthQueue.get()
            assert isinstance(msg, dai.ImgFrame)
            depth_u16 = msg.getFrame().astype(np.uint16)

            # If full-res is not (w,h), we still send actual shape.
            h0, w0 = depth_u16.shape
            payload = depth_u16.tobytes()
            ts_ns = time.time_ns()

            total_chunks = (len(payload) + args.chunk - 1) // args.chunk
            for idx in range(total_chunks):
                start = idx * args.chunk
                chunk_bytes = payload[start:start + args.chunk]
                hdr = struct.pack(
                    HDR_FMT,
                    MAGIC, VERSION, 0, 0,
                    frame_id,
                    w0, h0,
                    total_chunks, idx,
                    ts_ns,
                )
                sock.sendto(hdr + chunk_bytes, server_addr)

            frame_id = (frame_id + 1) & 0xFFFFFFFF


if __name__ == "__main__":
    main()
