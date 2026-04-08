#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time


MAGIC = b"NLT0"
HEADER = struct.Struct("!4sIQd")


def _format_mbps(byte_count: int, elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    return (8.0 * float(byte_count) / float(elapsed_s)) / 1_000_000.0


def _build_packet(seq: int, payload_size: int, filler: bytes) -> bytes:
    if payload_size < HEADER.size:
        raise ValueError(f"payload_size must be >= {HEADER.size}")
    body_size = int(payload_size) - HEADER.size
    if len(filler) < body_size:
        filler = (filler * ((body_size // max(len(filler), 1)) + 1))[:body_size]
    return HEADER.pack(MAGIC, int(seq), time.time_ns(), time.monotonic()) + filler[:body_size]


def run_sender(args: argparse.Namespace) -> int:
    dest = (str(args.host), int(args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(args.sndbuf))
    filler = os.urandom(max(1, int(args.payload_bytes) - HEADER.size))

    interval_s = 0.0 if float(args.pps) <= 0.0 else 1.0 / float(args.pps)
    deadline = time.monotonic() + float(args.duration)
    next_send_t = time.monotonic()
    seq = 0
    sent_bytes = 0
    last_report_t = time.monotonic()
    report_bytes = 0
    report_packets = 0

    print(f"[send] dest={dest[0]}:{dest[1]} payload_bytes={args.payload_bytes} pps={args.pps} duration_s={args.duration}")
    try:
        while time.monotonic() < deadline:
            now_t = time.monotonic()
            if interval_s > 0.0 and now_t < next_send_t:
                time.sleep(min(0.001, next_send_t - now_t))
                continue

            packet = _build_packet(seq=seq, payload_size=int(args.payload_bytes), filler=filler)
            sock.sendto(packet, dest)
            seq += 1
            sent_bytes += len(packet)
            report_bytes += len(packet)
            report_packets += 1

            if interval_s > 0.0:
                next_send_t += interval_s

            if now_t - last_report_t >= 1.0:
                elapsed = max(now_t - last_report_t, 1e-6)
                print(
                    f"[send] pps={report_packets / elapsed:.1f} "
                    f"mbps={_format_mbps(report_bytes, elapsed):.2f} "
                    f"seq={seq - 1}"
                )
                last_report_t = now_t
                report_bytes = 0
                report_packets = 0
    finally:
        sock.close()

    total_elapsed = float(args.duration)
    print(
        f"[send] done packets={seq} bytes={sent_bytes} "
        f"avg_mbps={_format_mbps(sent_bytes, total_elapsed):.2f}"
    )
    return 0


def run_receiver(args: argparse.Namespace) -> int:
    bind_addr = (str(args.bind), int(args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(args.rcvbuf))
    sock.bind(bind_addr)
    sock.settimeout(0.5)

    deadline = time.monotonic() + float(args.duration)
    recv_packets = 0
    recv_bytes = 0
    bad_packets = 0
    out_of_order = 0
    missing_total = 0
    first_seq: int | None = None
    last_seq: int | None = None
    max_gap = 0
    last_report_t = time.monotonic()
    report_packets = 0
    report_bytes = 0

    print(f"[recv] bind={bind_addr[0]}:{bind_addr[1]} duration_s={args.duration}")
    try:
        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(max(65535, int(args.payload_bytes) + 64))
            except socket.timeout:
                now_t = time.monotonic()
                if now_t - last_report_t >= 1.0:
                    elapsed = max(now_t - last_report_t, 1e-6)
                    print(
                        f"[recv] pps={report_packets / elapsed:.1f} "
                        f"mbps={_format_mbps(report_bytes, elapsed):.2f} "
                        f"missing_total={missing_total} out_of_order={out_of_order}"
                    )
                    last_report_t = now_t
                    report_packets = 0
                    report_bytes = 0
                continue

            if len(data) < HEADER.size:
                bad_packets += 1
                continue

            magic, seq, sent_wall_ns, sent_mono_s = HEADER.unpack_from(data, 0)
            if magic != MAGIC:
                bad_packets += 1
                continue

            now_t = time.monotonic()
            recv_packets += 1
            recv_bytes += len(data)
            report_packets += 1
            report_bytes += len(data)

            seq = int(seq)
            if first_seq is None:
                first_seq = seq
            if last_seq is not None:
                if seq < last_seq:
                    out_of_order += 1
                elif seq > last_seq + 1:
                    gap = seq - last_seq - 1
                    missing_total += gap
                    max_gap = max(max_gap, gap)
            last_seq = seq if last_seq is None else max(last_seq, seq)

            if now_t - last_report_t >= 1.0:
                elapsed = max(now_t - last_report_t, 1e-6)
                print(
                    f"[recv] from={addr[0]}:{addr[1]} "
                    f"pps={report_packets / elapsed:.1f} "
                    f"mbps={_format_mbps(report_bytes, elapsed):.2f} "
                    f"missing_total={missing_total} out_of_order={out_of_order}"
                )
                last_report_t = now_t
                report_packets = 0
                report_bytes = 0
    finally:
        sock.close()

    expected = 0
    if first_seq is not None and last_seq is not None:
        expected = (last_seq - first_seq) + 1
    loss_pct = 0.0 if expected <= 0 else (100.0 * float(missing_total) / float(expected))
    print(
        f"[recv] done packets={recv_packets} bytes={recv_bytes} "
        f"expected={expected} missing_total={missing_total} "
        f"loss_pct={loss_pct:.3f} out_of_order={out_of_order} "
        f"bad_packets={bad_packets} max_gap={max_gap}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Small raw UDP link test for Jetson-to-PC network debugging.")
    ap.add_argument("--mode", choices=["send", "recv"], required=True)
    ap.add_argument("--host", default="127.0.0.1", help="Receiver host for sender mode.")
    ap.add_argument("--bind", default="0.0.0.0", help="Bind IP for receiver mode.")
    ap.add_argument("--port", type=int, default=19000, help="UDP port.")
    ap.add_argument("--duration", type=float, default=20.0, help="Test duration in seconds.")
    ap.add_argument("--payload-bytes", type=int, default=1200, help="UDP payload size including header.")
    ap.add_argument("--pps", type=float, default=500.0, help="Packets per second in sender mode. <=0 means unthrottled.")
    ap.add_argument("--sndbuf", type=int, default=4 * 1024 * 1024, help="Sender UDP socket send buffer.")
    ap.add_argument("--rcvbuf", type=int, default=4 * 1024 * 1024, help="Receiver UDP socket receive buffer.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "send":
        return run_sender(args)
    if args.mode == "recv":
        return run_receiver(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
