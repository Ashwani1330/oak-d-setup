#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmark_profiles import ARTIFACT_ROOT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate rich comparison reports from benchmark run artifacts.")
    ap.add_argument("--runs-root", default=str(ARTIFACT_ROOT), help="Primary runs root, typically the PC receiver artifacts.")
    ap.add_argument(
        "--aux-runs-root",
        action="append",
        default=[],
        help="Additional runs roots to merge by run_id, e.g. a copied Jetson sender artifact root.",
    )
    ap.add_argument("--output-dir", default=None, help="Where to place the generated report. Defaults to <runs-root>/reports.")
    ap.add_argument("--run-id", default=None, help="Optional single run_id filter.")
    return ap.parse_args()


def load_role_bundle(run_root: Path, role: str) -> dict[str, Any] | None:
    summary_path = run_root / role / "summary.json"
    manifest_path = run_root / role / "manifest.json"
    if not summary_path.exists():
        return None
    return {
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
        "root": str(run_root),
    }


def load_run_bundle(runs_root: Path, run_id: str) -> dict[str, Any]:
    run_root = runs_root / run_id
    return {
        "run_id": run_id,
        "sender": load_role_bundle(run_root, "sender"),
        "receiver": load_role_bundle(run_root, "receiver"),
    }


def merge_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"run_id": bundles[0]["run_id"], "sender": None, "receiver": None}
    for role in ("sender", "receiver"):
        for bundle in bundles:
            value = bundle.get(role)
            if value is not None:
                merged[role] = value
                break
    return merged


def bundle_rows(bundle: dict[str, Any]) -> dict[str, Any]:
    sender = bundle.get("sender") or {}
    receiver = bundle.get("receiver") or {}
    sender_summary = sender.get("summary", {})
    receiver_summary = receiver.get("summary", {})
    sender_manifest = sender.get("manifest", {})
    receiver_manifest = receiver.get("manifest", {})
    sender_metrics = sender_summary.get("sender", {})
    receiver_metrics = receiver_summary.get("receiver", {})
    sender_host = sender_summary.get("host", {})
    receiver_host = receiver_summary.get("host", {})

    profile = (
        sender_summary.get("profile_name")
        or receiver_summary.get("profile_name")
        or sender_manifest.get("profile", {}).get("name")
        or receiver_manifest.get("profile", {}).get("name")
        or "unknown"
    )
    network = (
        sender_summary.get("network_label")
        or receiver_summary.get("network_label")
        or sender_manifest.get("profile", {}).get("network_label")
        or receiver_manifest.get("profile", {}).get("network_label")
        or "unknown"
    )
    compression = (
        sender_manifest.get("profile", {}).get("depth_comp")
        or receiver_manifest.get("profile", {}).get("depth_comp")
        or sender_summary.get("profile_name", "").split("_")[-1]
        or "unknown"
    )

    sender_depth_fps = _num(sender_metrics.get("depth_send_fps"))
    receiver_depth_fps = _num(receiver_metrics.get("depth_receive_fps"))
    compression_ms = _num(sender_metrics.get("compression_ms_mean"))
    send_ms = _num(sender_metrics.get("send_ms_mean"))
    decomp_ms = _num(receiver_metrics.get("decompression_ms_mean"))
    packet_count = _num(sender_metrics.get("packet_count_mean"))
    compression_ratio = _num(sender_metrics.get("compression_ratio_mean"))
    depth_bytes = _num(sender_metrics.get("depth_bytes_mean"))
    depth_mbps_est = None
    if sender_depth_fps is not None and depth_bytes is not None:
        depth_mbps_est = sender_depth_fps * depth_bytes * 8.0 / 1e6

    sender_duration = _num(sender_summary.get("duration_s"))
    receiver_duration = _num(receiver_summary.get("duration_s"))
    frame_budget_ms = _num(sender_summary.get("depth_frame_budget_ms")) or _num(receiver_summary.get("depth_frame_budget_ms"))
    latency_mean = _num(receiver_metrics.get("latency_ms_mean"))
    latency_p95 = _num(receiver_metrics.get("latency_ms_p95"))
    rgb_latency_mean = _num(receiver_metrics.get("rgb_latency_ms_mean"))
    fused_latency_mean = _num(receiver_metrics.get("fused_ready_latency_ms_mean"))
    fused_latency_p95 = _num(receiver_metrics.get("fused_ready_latency_ms_p95"))
    fused_skew_mean = _num(receiver_metrics.get("fused_source_skew_ms_mean"))
    rgb_pair_delta = _num(receiver_metrics.get("rgb_pair_delta_ms_mean"))
    practical_latency = fused_latency_mean
    practical_latency_basis = "direct_fused_ready"
    if practical_latency is None:
        if latency_mean is not None and rgb_pair_delta is not None:
            practical_latency = latency_mean + rgb_pair_delta
            practical_latency_basis = "heuristic_depth_plus_pair_delta"
        elif latency_mean is not None:
            practical_latency = latency_mean
            practical_latency_basis = "depth_only_fallback"
        else:
            practical_latency_basis = "unavailable"
    clock_sync = sender_manifest.get("clock_sync")
    clock_sync_rtt_ms = _num((clock_sync or {}).get("rtt_ms"))
    clock_sync_offset_ms = None
    if clock_sync is not None and (clock_sync or {}).get("offset_ns") is not None:
        clock_sync_offset_ms = float(clock_sync["offset_ns"]) / 1e6

    receiver_net_rx = _num(receiver_host.get("net_rx_mbps_mean"))
    sender_cpu = _num(sender_host.get("cpu_percent_mean"))
    receiver_cpu = _num(receiver_host.get("cpu_percent_mean"))
    sender_gpu = _num(sender_host.get("gpu_util_percent_mean"))
    receiver_gpu = _num(receiver_host.get("gpu_util_percent_mean"))
    receiver_ping = _num(receiver_host.get("ping_ms_mean"))
    sender_ping = _num(sender_host.get("ping_ms_mean"))

    stale_drops = _num(receiver_metrics.get("stale_depth_drops")) or 0.0
    missing_packets = _num(receiver_metrics.get("missing_packets_total")) or 0.0
    seq_gaps = _num(receiver_metrics.get("sequence_gaps")) or 0.0
    receiver_sync_warning = bool(rgb_pair_delta is not None and rgb_pair_delta > 250.0)
    latency_valid = bool(latency_mean is not None and clock_sync is not None)
    bottleneck = infer_bottleneck(
        sender_depth_fps=sender_depth_fps,
        receiver_depth_fps=receiver_depth_fps,
        frame_budget_ms=frame_budget_ms,
        compression_ms=compression_ms,
        send_ms=send_ms,
        decomp_ms=decomp_ms,
        stale_drops=stale_drops,
        seq_gaps=seq_gaps,
        missing_packets=missing_packets,
        sender_cpu=sender_cpu,
        receiver_cpu=receiver_cpu,
        rgb_pair_delta=rgb_pair_delta,
        fused_skew_ms=fused_skew_mean,
        compression=compression,
        depth_mbps_est=depth_mbps_est,
    )

    return {
        "run_id": bundle["run_id"],
        "profile_name": profile,
        "network_label": network,
        "compression": compression,
        "sender_present": bundle.get("sender") is not None,
        "receiver_present": bundle.get("receiver") is not None,
        "sender_duration_s": sender_duration,
        "receiver_duration_s": receiver_duration,
        "depth_frame_budget_ms": frame_budget_ms,
        "sender_depth_fps": sender_depth_fps,
        "receiver_depth_fps": receiver_depth_fps,
        "receiver_fps_efficiency": _safe_div(receiver_depth_fps, sender_depth_fps),
        "latency_ms_mean": latency_mean,
        "latency_ms_p95": latency_p95,
        "latency_valid": latency_valid,
        "rgb_latency_ms_mean": rgb_latency_mean,
        "usable_fused_latency_ms_mean": fused_latency_mean,
        "usable_fused_latency_ms_p95": fused_latency_p95,
        "fused_source_skew_ms_mean": fused_skew_mean,
        "rgb_pair_delta_ms_mean": rgb_pair_delta,
        "practical_latency_ms_est": practical_latency,
        "practical_latency_basis": practical_latency_basis,
        "receiver_sync_warning": receiver_sync_warning,
        "compression_ms_mean": compression_ms,
        "send_ms_mean": send_ms,
        "decompression_ms_mean": decomp_ms,
        "packet_count_mean": packet_count,
        "compression_ratio_mean": compression_ratio,
        "depth_bytes_mean": depth_bytes,
        "depth_payload_mbps_est": depth_mbps_est,
        "receiver_net_rx_mbps_mean": receiver_net_rx,
        "sender_cpu_mean": sender_cpu,
        "receiver_cpu_mean": receiver_cpu,
        "sender_gpu_mean": sender_gpu,
        "receiver_gpu_mean": receiver_gpu,
        "sender_ping_ms_mean": sender_ping,
        "receiver_ping_ms_mean": receiver_ping,
        "receiver_stale_depth_drops": stale_drops,
        "receiver_missing_packets_total": missing_packets,
        "receiver_sequence_gaps": seq_gaps,
        "clock_sync_rtt_ms": clock_sync_rtt_ms,
        "clock_sync_offset_ms": clock_sync_offset_ms,
        "sender_bottleneck": sender_summary.get("bottleneck"),
        "receiver_bottleneck": receiver_summary.get("bottleneck"),
        "inferred_bottleneck": bottleneck,
        "scenario_kind": scenario_kind(profile),
    }


def infer_bottleneck(
    *,
    sender_depth_fps: float | None,
    receiver_depth_fps: float | None,
    frame_budget_ms: float | None,
    compression_ms: float | None,
    send_ms: float | None,
    decomp_ms: float | None,
    stale_drops: float,
    seq_gaps: float,
    missing_packets: float,
    sender_cpu: float | None,
    receiver_cpu: float | None,
    rgb_pair_delta: float | None,
    fused_skew_ms: float | None,
    compression: str,
    depth_mbps_est: float | None,
) -> str:
    if stale_drops > 0 or seq_gaps > 0 or missing_packets > 0:
        return "network-loss-bound"
    if rgb_pair_delta is not None and rgb_pair_delta > 250.0:
        return "rgb-depth-sync-bound"
    if fused_skew_ms is not None and fused_skew_ms > 80.0:
        return "rgb-depth-sync-bound"
    if sender_depth_fps is not None and receiver_depth_fps is not None and sender_depth_fps > 0.0:
        efficiency = receiver_depth_fps / sender_depth_fps
        if efficiency < 0.78:
            return "network-or-receiver-bound"
    if frame_budget_ms is not None:
        if send_ms is not None and send_ms > max(5.0, frame_budget_ms * 0.35):
            if compression == "none" or (depth_mbps_est is not None and depth_mbps_est > 40.0):
                return "packetization-or-network-bound"
            return "sender-io-bound"
        if compression_ms is not None and compression_ms > max(4.0, frame_budget_ms * 0.12):
            return "compression-bound"
        if decomp_ms is not None and decomp_ms > max(1.0, frame_budget_ms * 0.08):
            return "receiver-decompression-bound"
    if sender_cpu is not None and sender_cpu > 12.0:
        return "jetson-cpu-bound"
    if receiver_cpu is not None and receiver_cpu > 15.0:
        return "receiver-cpu-bound"
    return "balanced"


def scenario_kind(profile_name: str) -> str:
    if profile_name in {"packet_overhead", "local_baseline"}:
        return "secondary"
    return "main"


def generate_markdown(rows: list[dict[str, Any]]) -> str:
    main_rows = [row for row in rows if row["scenario_kind"] == "main"]
    secondary_rows = [row for row in rows if row["scenario_kind"] != "main"]
    lines = [
        "# Camera-Edge Benchmark Report",
        "",
        f"Runs analyzed: {len(rows)}",
        "",
        "## Key Takeaways",
        "",
    ]
    lines.extend(generate_takeaways(rows))
    lines.extend(
        [
            "",
            "## Main Matrix",
            "",
            "| run_id | net | comp | sender fps | receiver fps | fps efficiency | depth latency ms | rgb latency ms | usable fused latency ms | payload mbps | packet ct | comp ms | send ms | decomp ms | source skew ms | bottleneck |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in main_rows:
        lines.append(
            "| {run_id} | {network_label} | {compression} | {sender_depth_fps} | {receiver_depth_fps} | "
            "{receiver_fps_efficiency} | {latency_ms_mean} | {rgb_latency_ms_mean} | {practical_latency_ms_est} | {depth_payload_mbps_est} | {packet_count_mean} | "
            "{compression_ms_mean} | {send_ms_mean} | {decompression_ms_mean} | {fused_source_skew_ms_mean} | {inferred_bottleneck} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )

    if secondary_rows:
        lines.extend(
            [
                "",
                "## Secondary Runs",
                "",
                "| run_id | profile | receiver fps | latency ms | payload mbps | packet ct | comp ms | send ms | bottleneck |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in secondary_rows:
            lines.append(
                "| {run_id} | {profile_name} | {receiver_depth_fps} | {practical_latency_ms_est} | {depth_payload_mbps_est} | "
                "{packet_count_mean} | {compression_ms_mean} | {send_ms_mean} | {inferred_bottleneck} |".format(
                    **{key: _fmt(value) for key, value in row.items()}
                )
            )

    lines.extend(["", "## Recommendations", ""])
    lines.extend(generate_recommendations(rows))

    lines.extend(["", "## Caveats", ""])
    lines.extend(generate_caveats(rows))
    return "\n".join(lines) + "\n"


def generate_takeaways(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    main_rows = [row for row in rows if row["scenario_kind"] == "main"]
    if not main_rows:
        return ["- No main-matrix runs were found."]

    best_throughput = max(main_rows, key=lambda row: row.get("receiver_depth_fps") or -1.0)
    valid_latency_rows = [row for row in main_rows if row.get("latency_valid") and row.get("latency_ms_mean") is not None]
    valid_practical_rows = [row for row in main_rows if row.get("practical_latency_ms_est") is not None]
    if valid_latency_rows:
        best_latency = min(valid_latency_rows, key=lambda row: row["latency_ms_mean"])
        lines.append(
            f"- Lowest raw depth-transport latency in the main matrix: `{best_latency['profile_name']}` at "
            f"{_fmt(best_latency['latency_ms_mean'])} ms."
        )
    if valid_practical_rows:
        best_practical = min(valid_practical_rows, key=lambda row: row["practical_latency_ms_est"])
        lines.append(
            f"- Lowest usable fused-frame latency in the main matrix: `{best_practical['profile_name']}` at "
            f"{_fmt(best_practical['practical_latency_ms_est'])} ms."
        )
    lines.append(
        f"- Highest receiver throughput in the main matrix: `{best_throughput['profile_name']}` at "
        f"{_fmt(best_throughput['receiver_depth_fps'])} FPS."
    )

    main_by_net_comp = {(row["network_label"], row["compression"]): row for row in main_rows}
    for comp in ("none", "lz4", "zstd"):
        wired = main_by_net_comp.get(("wired", comp))
        wireless = main_by_net_comp.get(("wireless", comp))
        if wired and wireless:
            fps_delta = _delta(wired.get("receiver_depth_fps"), wireless.get("receiver_depth_fps"))
            lines.append(
                f"- `{comp}` wired-vs-wireless receiver FPS delta: {_fmt(fps_delta)} "
                f"(positive means wired is faster)."
            )
    return lines


def generate_recommendations(rows: list[dict[str, Any]]) -> list[str]:
    main_rows = [row for row in rows if row["scenario_kind"] == "main"]
    if not main_rows:
        return ["- No recommendations available because no main-matrix runs were found."]

    best_balanced = rank_rows(main_rows, weights={"fps": 0.45, "practical_latency": 0.30, "bandwidth": 0.15, "sync": 0.10})[0]
    best_latency = rank_rows(main_rows, weights={"fps": 0.20, "practical_latency": 0.55, "bandwidth": 0.10, "sync": 0.15})[0]
    best_bandwidth = rank_rows(main_rows, weights={"fps": 0.20, "practical_latency": 0.10, "bandwidth": 0.55, "sync": 0.15})[0]
    highest_fps = max(main_rows, key=lambda row: row.get("receiver_depth_fps") or -1.0)

    lines = [
        f"- Best balanced default: `{best_balanced['profile_name']}` "
        f"(receiver FPS {_fmt(best_balanced['receiver_depth_fps'])}, usable fused latency {_fmt(best_balanced['practical_latency_ms_est'])} ms, "
        f"payload {_fmt(best_balanced['depth_payload_mbps_est'])} Mbps, bottleneck `{best_balanced['inferred_bottleneck']}`).",
        f"- Best latency-first option: `{best_latency['profile_name']}` at {_fmt(best_latency['practical_latency_ms_est'])} ms usable fused latency.",
        f"- Best bandwidth-efficient option: `{best_bandwidth['profile_name']}` at "
        f"{_fmt(best_bandwidth['depth_payload_mbps_est'])} Mbps estimated depth payload.",
        f"- Highest throughput option: `{highest_fps['profile_name']}` at {_fmt(highest_fps['receiver_depth_fps'])} FPS.",
    ]

    for row in main_rows:
        if row.get("inferred_bottleneck") == "packetization-or-network-bound":
            lines.append(
                f"- `{row['profile_name']}` looks packet/network-heavy: packet count {_fmt(row['packet_count_mean'])}, "
                f"send time {_fmt(row['send_ms_mean'])} ms."
            )
        if row.get("receiver_sync_warning"):
            lines.append(
                f"- `{row['profile_name']}` has a large RGB-depth pairing delta "
                f"({_fmt(row['rgb_pair_delta_ms_mean'])} ms), so its visual/depth alignment should be treated with caution."
            )
        if row.get("fused_source_skew_ms_mean") is not None and float(row["fused_source_skew_ms_mean"]) > 80.0:
            lines.append(
                f"- `{row['profile_name']}` shows high matched RGB/depth source skew "
                f"({_fmt(row['fused_source_skew_ms_mean'])} ms), so the fused frame is temporally weak even if transport latency is low."
            )
        if row.get("receiver_stale_depth_drops", 0) and row["receiver_stale_depth_drops"] > 0:
            lines.append(
                f"- `{row['profile_name']}` dropped stale depth frames ({_fmt(row['receiver_stale_depth_drops'])}), "
                "which points to network congestion or receive-side mismatch."
            )
    return lines


def generate_caveats(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "- `latency_ms_*` is only trustworthy when sender clock sync succeeded; the current runs appear synced, but future unsynced runs should be treated as invalid.",
        "- `usable_fused_latency_ms_mean` is the new headline metric for fresh runs. It measures when the matched RGB+depth pair is actually ready on the receiver, using sender-side RGB metadata plus the depth packet timestamp.",
        "- `fused_source_skew_ms_mean` measures how far apart the sender timestamps of the matched RGB and depth samples are. High skew means the pair is available but temporally weak.",
        "- `practical_latency_ms_est` now prefers direct fused-ready latency. For older runs without RGB metadata it falls back to the earlier heuristic so historical comparisons still render.",
        "- `depth_payload_mbps_est` is computed from sender depth payload only; it does not include RTSP RGB overhead, so it is best used for compression comparisons rather than total link budgeting.",
    ]
    if any((row.get("receiver_sync_warning") for row in rows)):
        lines.append("- Some wireless runs show very large RGB-depth pairing deltas, indicating receiver-side timing mismatch even when one-way latency is low.")
    if any(row.get("practical_latency_basis") != "direct_fused_ready" for row in rows):
        lines.append("- Some rows still rely on fallback latency logic because they were recorded before RGB metadata was added; rerun those scenarios if you want apples-to-apples fused-latency comparisons.")
    if any(not row.get("sender_present") for row in rows if row["scenario_kind"] == "main"):
        lines.append("- At least one main-matrix run is missing sender artifacts, so throughput efficiency and sender-side cost analysis are incomplete for that row.")
    return lines


def rank_rows(rows: list[dict[str, Any]], *, weights: dict[str, float]) -> list[dict[str, Any]]:
    fps_vals = [row.get("receiver_depth_fps") for row in rows if row.get("receiver_depth_fps") is not None]
    practical_latency_vals = [row.get("practical_latency_ms_est") for row in rows if row.get("practical_latency_ms_est") is not None]
    bw_vals = [row.get("depth_payload_mbps_est") for row in rows if row.get("depth_payload_mbps_est") is not None]
    sync_vals = [row.get("fused_source_skew_ms_mean") for row in rows if row.get("fused_source_skew_ms_mean") is not None]

    fps_min, fps_max = _min_max(fps_vals)
    practical_latency_min, practical_latency_max = _min_max(practical_latency_vals)
    bw_min, bw_max = _min_max(bw_vals)
    sync_min, sync_max = _min_max(sync_vals)

    ranked: list[dict[str, Any]] = []
    for row in rows:
        score = 0.0
        score += weights.get("fps", 0.0) * _norm_high(row.get("receiver_depth_fps"), fps_min, fps_max)
        score += weights.get("practical_latency", 0.0) * _norm_low(
            row.get("practical_latency_ms_est"), practical_latency_min, practical_latency_max
        )
        score += weights.get("bandwidth", 0.0) * _norm_low(row.get("depth_payload_mbps_est"), bw_min, bw_max)
        score += weights.get("sync", 0.0) * _norm_low(row.get("fused_source_skew_ms_mean"), sync_min, sync_max)
        ranked.append({**row, "score": score})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked


def maybe_plot(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_paths: list[str] = []
    main_rows = [row for row in rows if row["scenario_kind"] == "main"]
    if not main_rows:
        return []

    sorted_rows = sorted(main_rows, key=lambda row: (row["network_label"], row["compression"]))
    labels = [f"{row['network_label']}/{row['compression']}" for row in sorted_rows]

    simple_metrics = [
        ("receiver_depth_fps", "Receiver Depth FPS", "receiver_depth_fps.png"),
        ("latency_ms_mean", "Latency Mean (ms)", "latency_ms_mean.png"),
        ("rgb_latency_ms_mean", "RGB Ready Latency Mean (ms)", "rgb_latency_ms_mean.png"),
        ("practical_latency_ms_est", "Usable Fused Latency Mean (ms)", "practical_latency_ms_est.png"),
        ("compression_ms_mean", "Sender Compression Mean (ms)", "compression_ms_mean.png"),
        ("send_ms_mean", "Sender Send Mean (ms)", "send_ms_mean.png"),
        ("depth_payload_mbps_est", "Estimated Depth Payload Mbps", "depth_payload_mbps_est.png"),
        ("packet_count_mean", "Mean UDP Packets Per Depth Frame", "packet_count_mean.png"),
        ("fused_source_skew_ms_mean", "Matched RGB/Depth Source Skew Mean (ms)", "fused_source_skew_ms_mean.png"),
    ]
    for key, title, filename in simple_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        values = [row.get(key) or 0.0 for row in sorted_rows]
        colors = ["#175676" if row["network_label"] == "wired" else "#d62828" for row in sorted_rows]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        output_paths.append(str(out_path))

    # Throughput vs bandwidth tradeoff.
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in sorted_rows:
        x = row.get("depth_payload_mbps_est") or 0.0
        y = row.get("receiver_depth_fps") or 0.0
        size = 80
        latency = row.get("practical_latency_ms_est") or row.get("latency_ms_mean")
        if latency is not None:
            size = max(50.0, min(300.0, 8.0 * float(latency)))
        color = "#175676" if row["network_label"] == "wired" else "#d62828"
        ax.scatter([x], [y], s=size, color=color, alpha=0.75)
        ax.annotate(f"{row['network_label']}/{row['compression']}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Estimated depth payload Mbps")
    ax.set_ylabel("Receiver depth FPS")
    ax.set_title("Throughput vs bandwidth (bubble size = usable fused latency)")
    fig.tight_layout()
    out_path = output_dir / "throughput_vs_bandwidth.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    output_paths.append(str(out_path))

    # Sender stage cost breakdown.
    fig, ax = plt.subplots(figsize=(10, 4))
    comp_vals = [row.get("compression_ms_mean") or 0.0 for row in sorted_rows]
    send_vals = [row.get("send_ms_mean") or 0.0 for row in sorted_rows]
    decomp_vals = [row.get("decompression_ms_mean") or 0.0 for row in sorted_rows]
    ax.bar(labels, comp_vals, label="compress", color="#457b9d")
    ax.bar(labels, send_vals, bottom=comp_vals, label="send", color="#e76f51")
    stacked = [c + s for c, s in zip(comp_vals, send_vals)]
    ax.bar(labels, decomp_vals, bottom=stacked, label="decompress", color="#2a9d8f")
    ax.set_ylabel("ms")
    ax.set_title("Pipeline stage time budget per depth frame")
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_path = output_dir / "stage_time_budget.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    output_paths.append(str(out_path))
    return output_paths


def collect_run_ids(roots: list[Path], explicit_run_id: str | None) -> list[str]:
    if explicit_run_id:
        return [explicit_run_id]
    run_ids: set[str] = set()
    for root in roots:
        if root.exists():
            run_ids.update(path.name for path in root.iterdir() if path.is_dir() and path.name != "reports")
    return sorted(run_ids)


def main() -> int:
    args = parse_args()
    primary_root = Path(args.runs_root).resolve()
    aux_roots = [Path(root).resolve() for root in args.aux_runs_root]
    roots = [primary_root] + aux_roots
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (primary_root / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = collect_run_ids(roots, args.run_id)
    merged_bundles: list[dict[str, Any]] = []
    for run_id in run_ids:
        merged_bundles.append(merge_bundles([load_run_bundle(root, run_id) for root in roots]))

    rows = [bundle_rows(bundle) for bundle in merged_bundles if bundle.get("sender") or bundle.get("receiver")]
    rows.sort(key=lambda row: (row["scenario_kind"], row["network_label"], row["compression"], row["run_id"]))

    markdown = generate_markdown(rows)
    md_path = output_dir / "comparison_summary.md"
    md_path.write_text(markdown, encoding="utf-8")

    plot_paths = maybe_plot(rows, output_dir)
    json_path = output_dir / "comparison_data.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Report written to {md_path}")
    print(f"Data written to {json_path}")
    for path in plot_paths:
        print(f"Plot written to {path}")
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0.0):
        return None
    return float(a) / float(b)


def _min_max(values: list[float | None]) -> tuple[float, float]:
    nums = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not nums:
        return 0.0, 1.0
    return min(nums), max(nums)


def _norm_high(value: float | None, lo: float, hi: float) -> float:
    if value is None:
        return 0.0
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def _norm_low(value: float | None, lo: float, hi: float) -> float:
    if value is None:
        return 0.0
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (hi - float(value)) / (hi - lo)))


if __name__ == "__main__":
    raise SystemExit(main())
