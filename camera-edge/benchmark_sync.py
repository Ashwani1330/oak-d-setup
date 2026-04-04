#!/usr/bin/env python3
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass


SYNC_REQ = struct.Struct("<4sQ")
SYNC_RES = struct.Struct("<4sQQQ")
MAGIC_REQ = b"TSQ0"
MAGIC_RES = b"TSP0"


@dataclass(frozen=True)
class ClockSyncEstimate:
    offset_ns: int
    rtt_ms: float
    sample_count: int


class ClockSyncServer:
    def __init__(self, *, bind_ip: str, port: int) -> None:
        self.bind_ip = str(bind_ip)
        self.port = int(port)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="clock-sync-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.bind_ip, self.port))
        sock.settimeout(0.5)
        self._sock = sock
        try:
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(256)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                if len(data) < SYNC_REQ.size:
                    continue
                magic, client_send_ns = SYNC_REQ.unpack_from(data, 0)
                if magic != MAGIC_REQ:
                    continue
                server_recv_ns = time.time_ns()
                server_send_ns = time.time_ns()
                response = SYNC_RES.pack(MAGIC_RES, int(client_send_ns), server_recv_ns, server_send_ns)
                sock.sendto(response, addr)
        finally:
            try:
                sock.close()
            except Exception:
                pass
            self._sock = None


def estimate_clock_offset(
    *,
    host: str,
    port: int,
    timeout_s: float = 0.6,
    samples: int = 5,
) -> ClockSyncEstimate | None:
    best: tuple[int, float] | None = None
    success = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(max(0.1, float(timeout_s)))
    try:
        for _ in range(max(1, int(samples))):
            client_send_ns = time.time_ns()
            sock.sendto(SYNC_REQ.pack(MAGIC_REQ, client_send_ns), (host, int(port)))
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            client_recv_ns = time.time_ns()
            if len(data) < SYNC_RES.size:
                continue
            magic, echoed_send_ns, server_recv_ns, server_send_ns = SYNC_RES.unpack_from(data, 0)
            if magic != MAGIC_RES or int(echoed_send_ns) != int(client_send_ns):
                continue
            rtt_ns = max(0, client_recv_ns - client_send_ns - (server_send_ns - server_recv_ns))
            offset_ns = int(((server_recv_ns + server_send_ns) - (client_send_ns + client_recv_ns)) / 2.0)
            rtt_ms = rtt_ns / 1e6
            success += 1
            if best is None or rtt_ms < best[1]:
                best = (offset_ns, rtt_ms)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    if best is None:
        return None
    return ClockSyncEstimate(offset_ns=int(best[0]), rtt_ms=float(best[1]), sample_count=success)
