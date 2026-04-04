#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parent / "benchmark_runs"

# Edit this when you want `python camera-edge/<script>.py` to pick a different default.
ACTIVE_PROFILE_NAME = "wired_lz4"

# Real host/network details from the current setup.
SERVER_PC_WIFI_IP = "192.168.1.183"
SERVER_PC_WIRED_IP = "172.31.1.148"
ORIN_WIFI_IP = "192.168.1.211"
ORIN_WIRED_IP = "172.31.1.211"

SERVER_PC_WIFI_NIC = "Wi-Fi"
SERVER_PC_WIRED_NIC = "Ethernet"
ORIN_WIFI_NIC = "wlP1p1s0"
ORIN_WIRED_NIC = "enP8p1s0"


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    network_label: str
    nic_name: str
    peer_host: str
    ping_host: str
    run_notes: str
    rtsp_url: str
    sender_nic_name: str | None = None
    receiver_nic_name: str | None = None
    sender_ping_host: str | None = None
    receiver_ping_host: str | None = None
    rtsp_transport: str = "udp"
    bind_ip: str = "0.0.0.0"
    depth_host: str = "127.0.0.1"
    depth_port: int = 9000
    control_port: int = 9001
    rgb_meta_port: int = 9002
    nic_bandwidth_mbps: float = 100.0

    width: int = 640
    height: int = 400
    fps: int = 30
    depth_fps: int = 15
    depth_comp: str = "lz4"
    depth_packet_bytes: int = 1200

    min_mm: int = 120
    max_mm: int = 2500
    calib_period_sec: float = 5.0

    extended_disparity: bool = True
    subpixel: bool = False
    force_decimation_1: bool = True

    stale_depth_ms: int = 120
    socket_rcvbuf: int = 8_388_608
    socket_sndbuf: int = 4 * 1024 * 1024
    max_rgb_buffer: int = 4
    max_rgb_depth_delta_ms: float = 30.0

    duration_sec: float = 30.0
    artifact_root: str = str(ARTIFACT_ROOT)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def nic_name_for_role(self, role: str) -> str:
        if str(role) == "sender":
            return str(self.sender_nic_name or self.nic_name)
        if str(role) == "receiver":
            return str(self.receiver_nic_name or self.nic_name)
        return str(self.nic_name)

    def ping_host_for_role(self, role: str) -> str:
        if str(role) == "sender":
            return str(self.sender_ping_host or self.ping_host)
        if str(role) == "receiver":
            return str(self.receiver_ping_host or self.ping_host)
        return str(self.ping_host)


def _base_profile() -> BenchmarkProfile:
    return BenchmarkProfile(
        name="base",
        network_label="wireless",
        nic_name=ORIN_WIFI_NIC,
        peer_host=SERVER_PC_WIFI_IP,
        ping_host=SERVER_PC_WIFI_IP,
        sender_nic_name=ORIN_WIFI_NIC,
        receiver_nic_name=SERVER_PC_WIFI_NIC,
        sender_ping_host=SERVER_PC_WIFI_IP,
        receiver_ping_host=ORIN_WIFI_IP,
        run_notes="Wi-Fi path configured from the current Orin and server PC addresses.",
        rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
        bind_ip="0.0.0.0",
        depth_host=SERVER_PC_WIFI_IP,
        depth_port=9000,
        control_port=9001,
        rgb_meta_port=9002,
        nic_bandwidth_mbps=100.0,
        width=640,
        height=400,
        fps=30,
        depth_fps=15,
        depth_comp="lz4",
        depth_packet_bytes=1200,
        min_mm=120,
        max_mm=2500,
        calib_period_sec=5.0,
        extended_disparity=True,
        subpixel=False,
        force_decimation_1=True,
        stale_depth_ms=120,
        socket_rcvbuf=8_388_608,
        socket_sndbuf=4 * 1024 * 1024,
        max_rgb_buffer=4,
        max_rgb_depth_delta_ms=30.0,
        duration_sec=30.0,
        artifact_root=str(ARTIFACT_ROOT),
    )


def _with_name(profile: BenchmarkProfile, name: str, **kwargs: Any) -> BenchmarkProfile:
    return replace(profile, name=name, **kwargs)


def build_profiles() -> dict[str, BenchmarkProfile]:
    base = _base_profile()
    return {
        "wired_none": _with_name(
            base,
            "wired_none",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            depth_comp="none",
            nic_bandwidth_mbps=1000.0,
            run_notes="Main comparison: wired Ethernet without depth compression.",
        ),
        "wired_lz4": _with_name(
            base,
            "wired_lz4",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            depth_comp="lz4",
            nic_bandwidth_mbps=1000.0,
            run_notes="Main comparison: wired Ethernet with LZ4 depth compression.",
        ),
        "wired_zstd": _with_name(
            base,
            "wired_zstd",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            depth_comp="zstd",
            nic_bandwidth_mbps=1000.0,
            run_notes="Main comparison: wired Ethernet with Zstd depth compression.",
        ),
        "wired_none_15x15": _with_name(
            base,
            "wired_none_15x15",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            fps=15,
            depth_fps=15,
            depth_comp="none",
            nic_bandwidth_mbps=1000.0,
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: wired Ethernet, RGB 15 FPS, depth 15 FPS, no depth compression.",
        ),
        "wired_lz4_15x15": _with_name(
            base,
            "wired_lz4_15x15",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            fps=15,
            depth_fps=15,
            depth_comp="lz4",
            nic_bandwidth_mbps=1000.0,
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: wired Ethernet, RGB 15 FPS, depth 15 FPS, LZ4 depth compression.",
        ),
        "wired_zstd_15x15": _with_name(
            base,
            "wired_zstd_15x15",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            fps=15,
            depth_fps=15,
            depth_comp="zstd",
            nic_bandwidth_mbps=1000.0,
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: wired Ethernet, RGB 15 FPS, depth 15 FPS, Zstd depth compression.",
        ),
        "wireless_none": _with_name(
            base,
            "wireless_none",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            depth_comp="none",
            run_notes="Main comparison: Wi-Fi without depth compression.",
        ),
        "wireless_lz4": _with_name(
            base,
            "wireless_lz4",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            depth_comp="lz4",
            run_notes="Main comparison: Wi-Fi with LZ4 depth compression.",
        ),
        "wireless_zstd": _with_name(
            base,
            "wireless_zstd",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            depth_comp="zstd",
            run_notes="Main comparison: Wi-Fi with Zstd depth compression.",
        ),
        "wireless_none_15x15": _with_name(
            base,
            "wireless_none_15x15",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            fps=15,
            depth_fps=15,
            depth_comp="none",
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: Wi-Fi, RGB 15 FPS, depth 15 FPS, no depth compression.",
        ),
        "wireless_lz4_15x15": _with_name(
            base,
            "wireless_lz4_15x15",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            fps=15,
            depth_fps=15,
            depth_comp="lz4",
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: Wi-Fi, RGB 15 FPS, depth 15 FPS, LZ4 depth compression.",
        ),
        "wireless_zstd_15x15": _with_name(
            base,
            "wireless_zstd_15x15",
            network_label="wireless",
            nic_name=ORIN_WIFI_NIC,
            peer_host=SERVER_PC_WIFI_IP,
            ping_host=SERVER_PC_WIFI_IP,
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host=SERVER_PC_WIFI_IP,
            receiver_ping_host=ORIN_WIFI_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIFI_IP}:8554/oak",
            depth_host=SERVER_PC_WIFI_IP,
            fps=15,
            depth_fps=15,
            depth_comp="zstd",
            max_rgb_depth_delta_ms=30.0,
            run_notes="Cadence-matched sync study: Wi-Fi, RGB 15 FPS, depth 15 FPS, Zstd depth compression.",
        ),
        "local_baseline": _with_name(
            base,
            "local_baseline",
            network_label="local",
            nic_name=ORIN_WIFI_NIC,
            ping_host="127.0.0.1",
            sender_nic_name=ORIN_WIFI_NIC,
            receiver_nic_name=SERVER_PC_WIFI_NIC,
            sender_ping_host="127.0.0.1",
            receiver_ping_host="127.0.0.1",
            depth_host="127.0.0.1",
            peer_host="127.0.0.1",
            run_notes="Sender-only baseline for camera/compression cost with loopback target.",
        ),
        "packet_overhead": _with_name(
            base,
            "packet_overhead",
            network_label="wired",
            nic_name=ORIN_WIRED_NIC,
            peer_host=SERVER_PC_WIRED_IP,
            ping_host=SERVER_PC_WIRED_IP,
            sender_nic_name=ORIN_WIRED_NIC,
            receiver_nic_name=SERVER_PC_WIRED_NIC,
            sender_ping_host=SERVER_PC_WIRED_IP,
            receiver_ping_host=ORIN_WIRED_IP,
            rtsp_url=f"rtsp://{SERVER_PC_WIRED_IP}:8554/oak",
            depth_host=SERVER_PC_WIRED_IP,
            depth_comp="lz4",
            depth_packet_bytes=900,
            run_notes="Wired packet-overhead microbench. Edit depth_packet_bytes for sweeps.",
        ),
    }


PROFILES = build_profiles()


def get_profile(name: str | None = None) -> BenchmarkProfile:
    selected = ACTIVE_PROFILE_NAME if name is None else str(name)
    try:
        return PROFILES[selected]
    except KeyError as exc:
        names = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown benchmark profile '{selected}'. Available: {names}") from exc


def profile_names() -> list[str]:
    return sorted(PROFILES)
