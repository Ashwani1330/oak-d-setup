#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_profiles import ARTIFACT_ROOT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate comparison tables and plots from benchmark run artifacts.")
    ap.add_argument("--runs-root", default=str(ARTIFACT_ROOT), help="Root folder containing benchmark run directories.")
    ap.add_argument("--output-dir", default=None, help="Where to place the generated report. Defaults to <runs-root>/reports.")
    ap.add_argument("--run-id", default=None, help="Optional single run_id filter.")
    return ap.parse_args()


def load_run_bundle(runs_root: Path, run_id: str) -> dict[str, Any]:
    bundle: dict[str, Any] = {"run_id": run_id, "sender": None, "receiver": None}
    run_root = runs_root / run_id
    for role in ("sender", "receiver"):
        summary_path = run_root / role / "summary.json"
        manifest_path = run_root / role / "manifest.json"
        if summary_path.exists():
            bundle[role] = {
                "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
            }
    return bundle


def bundle_rows(bundle: dict[str, Any]) -> dict[str, Any]:
    sender = bundle.get("sender") or {}
    receiver = bundle.get("receiver") or {}
    sender_summary = sender.get("summary", {})
    receiver_summary = receiver.get("summary", {})
    sender_manifest = sender.get("manifest", {})
    receiver_manifest = receiver.get("manifest", {})
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
    sender_metrics = sender_summary.get("sender", {})
    receiver_metrics = receiver_summary.get("receiver", {})
    sender_host = sender_summary.get("host", {})
    receiver_host = receiver_summary.get("host", {})
    return {
        "run_id": bundle["run_id"],
        "profile_name": profile,
        "network_label": network,
        "compression": sender_manifest.get("profile", {}).get("depth_comp")
        or receiver_manifest.get("profile", {}).get("depth_comp")
        or "unknown",
        "sender_depth_fps": sender_metrics.get("depth_send_fps"),
        "receiver_depth_fps": receiver_metrics.get("depth_receive_fps"),
        "latency_ms_mean": receiver_metrics.get("latency_ms_mean"),
        "latency_ms_p95": receiver_metrics.get("latency_ms_p95"),
        "rgb_pair_delta_ms_mean": receiver_metrics.get("rgb_pair_delta_ms_mean"),
        "compression_ms_mean": sender_metrics.get("compression_ms_mean"),
        "decompression_ms_mean": receiver_metrics.get("decompression_ms_mean"),
        "packet_count_mean": sender_metrics.get("packet_count_mean"),
        "compression_ratio_mean": sender_metrics.get("compression_ratio_mean"),
        "sender_cpu_mean": sender_host.get("cpu_percent_mean"),
        "receiver_cpu_mean": receiver_host.get("cpu_percent_mean"),
        "sender_gpu_mean": sender_host.get("gpu_util_percent_mean"),
        "receiver_gpu_mean": receiver_host.get("gpu_util_percent_mean"),
        "sender_net_tx_mbps_mean": sender_host.get("net_tx_mbps_mean"),
        "receiver_net_rx_mbps_mean": receiver_host.get("net_rx_mbps_mean"),
        "receiver_stale_depth_drops": receiver_metrics.get("stale_depth_drops"),
        "receiver_sequence_gaps": receiver_metrics.get("sequence_gaps"),
        "sender_bottleneck": sender_summary.get("bottleneck"),
        "receiver_bottleneck": receiver_summary.get("bottleneck"),
    }


def generate_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Camera-Edge Benchmark Report",
        "",
        f"Runs analyzed: {len(rows)}",
        "",
        "## Summary Table",
        "",
        "| run_id | profile | net | comp | sender depth fps | receiver depth fps | latency mean ms | comp ms | decomp ms | tx mbps | rx mbps | stale | gaps |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {profile_name} | {network_label} | {compression} | {sender_depth_fps} | "
            "{receiver_depth_fps} | {latency_ms_mean} | {compression_ms_mean} | {decompression_ms_mean} | "
            "{sender_net_tx_mbps_mean} | {receiver_net_rx_mbps_mean} | {receiver_stale_depth_drops} | {receiver_sequence_gaps} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )

    lines.extend(["", "## Wired vs Wireless Deltas", ""])
    by_profile = {(row["network_label"], row["compression"]): row for row in rows}
    for comp in ("none", "lz4", "zstd"):
        wired = by_profile.get(("wired", comp))
        wireless = by_profile.get(("wireless", comp))
        if wired is None or wireless is None:
            continue
        delta_fps = _delta(wired.get("receiver_depth_fps"), wireless.get("receiver_depth_fps"))
        delta_latency = _delta(wireless.get("latency_ms_mean"), wired.get("latency_ms_mean"))
        lines.append(
            f"- `{comp}`: wired-vs-wireless receiver FPS delta = {_fmt(delta_fps)}, "
            f"wireless extra latency = {_fmt(delta_latency)} ms."
        )
    return "\n".join(lines) + "\n"


def maybe_plot(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_paths: list[str] = []
    metrics = [
        ("receiver_depth_fps", "Receiver Depth FPS", "receiver_depth_fps.png"),
        ("latency_ms_mean", "Latency Mean (ms)", "latency_ms_mean.png"),
        ("compression_ms_mean", "Compression Mean (ms)", "compression_ms_mean.png"),
    ]
    labels = [f"{row['network_label']}/{row['compression']}" for row in rows]
    for key, title, filename in metrics:
        values = [row.get(key) or 0.0 for row in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, values, color="#2c7fb8")
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        output_paths.append(str(out_path))
    return output_paths


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (runs_root / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_ids = [args.run_id]
    elif runs_root.exists():
        run_ids = sorted(path.name for path in runs_root.iterdir() if path.is_dir())
    else:
        run_ids = []
    bundles = [load_run_bundle(runs_root, run_id) for run_id in run_ids]
    rows = [bundle_rows(bundle) for bundle in bundles if bundle.get("sender") or bundle.get("receiver")]
    rows.sort(key=lambda row: (row["network_label"], row["compression"], row["run_id"]))

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
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


if __name__ == "__main__":
    raise SystemExit(main())
