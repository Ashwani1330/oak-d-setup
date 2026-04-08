#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from antropi_config import CAMERA_EDGE_RUNTIME_CONFIG


@dataclass(frozen=True)
class RuntimeProfile:
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
    max_rgb_buffer: int = 8
    max_rgb_depth_delta_ms: float = 250.0

    duration_sec: float = 30.0

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


def _base_profile() -> RuntimeProfile:
    cfg = CAMERA_EDGE_RUNTIME_CONFIG
    return RuntimeProfile(
        name="base",
        network_label="wireless",
        nic_name=cfg.orin_wifi_nic,
        peer_host=cfg.server_pc_wifi_ip,
        ping_host=cfg.server_pc_wifi_ip,
        sender_nic_name=cfg.orin_wifi_nic,
        receiver_nic_name=cfg.server_pc_wifi_nic,
        sender_ping_host=cfg.server_pc_wifi_ip,
        receiver_ping_host=cfg.orin_wifi_ip,
        run_notes="Wi-Fi path configured from the current Orin and server PC addresses.",
        rtsp_url=f"rtsp://{cfg.server_pc_wifi_ip}:8554/oak",
        bind_ip="0.0.0.0",
        depth_host=cfg.server_pc_wifi_ip,
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
        max_rgb_buffer=8,
        max_rgb_depth_delta_ms=250.0,
        duration_sec=30.0,
    )


def _with_name(profile: RuntimeProfile, name: str, **kwargs: Any) -> RuntimeProfile:
    return replace(profile, name=name, **kwargs)


def build_profiles() -> dict[str, RuntimeProfile]:
    cfg = CAMERA_EDGE_RUNTIME_CONFIG
    base = _base_profile()
    return {
        "wired_lz4": _with_name(
            base,
            "wired_lz4",
            network_label="wired",
            nic_name=cfg.orin_wired_nic,
            peer_host=cfg.server_pc_wired_ip,
            ping_host=cfg.server_pc_wired_ip,
            sender_nic_name=cfg.orin_wired_nic,
            receiver_nic_name=cfg.server_pc_wired_nic,
            sender_ping_host=cfg.server_pc_wired_ip,
            receiver_ping_host=cfg.orin_wired_ip,
            rtsp_url=f"rtsp://{cfg.server_pc_wired_ip}:8554/oak",
            depth_host=cfg.server_pc_wired_ip,
            depth_comp="lz4",
            nic_bandwidth_mbps=1000.0,
            run_notes="Main comparison: wired Ethernet with LZ4 depth compression.",
        ),
        "wired_zstd": _with_name(
            base,
            "wired_zstd",
            network_label="wired",
            nic_name=cfg.orin_wired_nic,
            peer_host=cfg.server_pc_wired_ip,
            ping_host=cfg.server_pc_wired_ip,
            sender_nic_name=cfg.orin_wired_nic,
            receiver_nic_name=cfg.server_pc_wired_nic,
            sender_ping_host=cfg.server_pc_wired_ip,
            receiver_ping_host=cfg.orin_wired_ip,
            rtsp_url=f"rtsp://{cfg.server_pc_wired_ip}:8554/oak",
            depth_host=cfg.server_pc_wired_ip,
            depth_comp="zstd",
            nic_bandwidth_mbps=1000.0,
            run_notes="Main comparison: wired Ethernet with Zstd depth compression.",
        ),
        "wireless_lz4": _with_name(
            base,
            "wireless_lz4",
            network_label="wireless",
            nic_name=cfg.orin_wifi_nic,
            peer_host=cfg.server_pc_wifi_ip,
            ping_host=cfg.server_pc_wifi_ip,
            sender_nic_name=cfg.orin_wifi_nic,
            receiver_nic_name=cfg.server_pc_wifi_nic,
            sender_ping_host=cfg.server_pc_wifi_ip,
            receiver_ping_host=cfg.orin_wifi_ip,
            rtsp_url=f"rtsp://{cfg.server_pc_wifi_ip}:8554/oak",
            depth_host=cfg.server_pc_wifi_ip,
            depth_comp="lz4",
            run_notes="Main comparison: Wi-Fi with LZ4 depth compression.",
        ),
        "wireless_zstd": _with_name(
            base,
            "wireless_zstd",
            network_label="wireless",
            nic_name=cfg.orin_wifi_nic,
            peer_host=cfg.server_pc_wifi_ip,
            ping_host=cfg.server_pc_wifi_ip,
            sender_nic_name=cfg.orin_wifi_nic,
            receiver_nic_name=cfg.server_pc_wifi_nic,
            sender_ping_host=cfg.server_pc_wifi_ip,
            receiver_ping_host=cfg.orin_wifi_ip,
            rtsp_url=f"rtsp://{cfg.server_pc_wifi_ip}:8554/oak",
            depth_host=cfg.server_pc_wifi_ip,
            depth_comp="zstd",
            run_notes="Main comparison: Wi-Fi with Zstd depth compression.",
        ),
    }


PROFILES = build_profiles()


def get_profile(name: str | None = None) -> RuntimeProfile:
    selected = CAMERA_EDGE_RUNTIME_CONFIG.active_profile_name if name is None else str(name)
    try:
        return PROFILES[selected]
    except KeyError as exc:
        names = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown runtime profile '{selected}'. Available: {names}") from exc


def profile_names() -> list[str]:
    return sorted(PROFILES)
