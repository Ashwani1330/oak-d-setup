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

# CAL: magic(4) ver(u8) reserved(u8) w(u16) h(u16) fx fy cx cy (f32)
CAL_FMT = "!4sBB H H f f f f"
CAL_SIZE = struct.calcsize(CAL_FMT)


def send_cal(sock, addr, w, h, fx, fy, cx, cy):
    pkt = struct.pack(CAL_FMT, MAGIC_CAL, VERSION, 0, w, h, fx, fy, cx, cy)
    sock.sendto(pkt, addr)


with dai.Pipeline() as p:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ip", required=True)
    ap.add_argument("--server-port", type=int, default=5005)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=1400)
    ap.add_argument("--send-cal-every", type=float, default=2.0)
    ap.add_argument("--left-socket", default="CAM_B")
    ap.add_argument("--right-socket", default="CAM_C")
    ap.add_argument("--align-socket", default="CAM_A")  # RGB on OAK-D
    args = ap.parse_args()

    server_addr = (args.server_ip, args.server_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4_000_000)

    left_socket = getattr(dai.CameraBoardSocket, args.left_socket)
    right_socket = getattr(dai.CameraBoardSocket, args.right_socket)
    align_socket = getattr(dai.CameraBoardSocket, args.align_socket)

    # Nodes (v3 style like your doc snippet)
    left = p.create(dai.node.Camera)
    right = p.create(dai.node.Camera)
    stereo = p.create(dai.node.StereoDepth)

    left.build(left_socket)
    right.build(right_socket)

    # Preset enum differs by build; DEFAULT is safe and matches doc snippet.
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)               # long-range precision :contentReference[oaicite:3]{index=3}
    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)         # occlusion handling :contentReference[oaicite:4]{index=4}
    stereo.setDepthAlign(align_socket)

    # Link cameras to stereo (v3 requestOutput() API) :contentReference[oaicite:5]{index=5}
    left.requestOutput((args.w, args.h)).link(stereo.left)
    right.requestOutput((args.w, args.h)).link(stereo.right)

    # v3: output queue directly from the output (no XLinkOut) :contentReference[oaicite:6]{index=6}
    depthQ = stereo.depth.createOutputQueue(maxSize=4, blocking=True)

    # Device handle is obtained from pipeline (docs use getDefaultDevice()) :contentReference[oaicite:7]{index=7}
    dev = p.getDefaultDevice()
    calib = dev.readCalibration()
    K = np.array(calib.getCameraIntrinsics(align_socket, args.w, args.h), dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    p.start()  # v3 start :contentReference[oaicite:8]{index=8}

    print(f"[sender] depth {args.w}x{args.h}@{args.fps} -> udp://{server_addr[0]}:{server_addr[1]}")
    print(f"[sender] sockets L={args.left_socket} R={args.right_socket} align={args.align_socket}")
    print(f"[sender] intrinsics fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    frame_id = 0
    last_cal = 0.0

    while p.isRunning():
        now = time.monotonic()
        if now - last_cal >= args.send_cal_every:
            send_cal(sock, server_addr, args.w, args.h, fx, fy, cx, cy)
            last_cal = now

        msg = depthQ.get()
        depth_u16 = msg.getFrame().astype(np.uint16)
        h0, w0 = depth_u16.shape  # trust actual
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
