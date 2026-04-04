#!/usr/bin/env python3
# sam3_remote_test.py: Remote SAM3 capture test using RTSP RGB + UDP depth from an OAK-D edge sender.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from antropi_config import DEFAULT_SAM_RUNNER_CONFIG, SamRunnerConfig
from vision.geometry_utils import _compute_robust_pitch_yaw_from_samples, extract_motion_target_from_samples
from vision.oakd_remote import build_remote_oakd_pipeline, create_remote_oakd_queues
from vision.sam_runner import (
    Sam3Runtime,
    _candidate_overlays,
    _capture_segmented_samples,
    _release_sam_runtime,
    select_best_candidate,
)
from vision.viz import (
    CAPTURE_WINDOW_NAME,
    _show_result,
    destroy_capture_window,
    is_capture_quit_key,
    make_segmented_preview_bgr,
    show_capture_frame_key,
)


def add_bool_arg(ap: argparse.ArgumentParser, name: str, default: bool, help_txt: str) -> None:
    dest = name.replace("-", "_")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help_txt + f" (default={default})")
    g.add_argument(f"--no-{name}", dest=dest, action="store_false", help="Disable: " + help_txt)
    ap.set_defaults(**{dest: default})


def _build_config(args: argparse.Namespace) -> SamRunnerConfig:
    base = DEFAULT_SAM_RUNNER_CONFIG
    return SamRunnerConfig(
        prompt=str(args.prompt),
        valid_frames=int(args.valid_frames),
        min_points=int(args.min_points),
        orient_median_repeats=int(args.orient_median_repeats),
        voxel_scene_mm=float(base.voxel_scene_mm),
        voxel_obj_mm=float(base.voxel_obj_mm),
        max_viz_points=int(base.max_viz_points),
        tint_rgb_u8=tuple(int(v) for v in base.tint_rgb_u8),
        tint_alpha=float(base.tint_alpha),
        side_dot_abs_max=float(base.side_dot_abs_max),
        refine_axis_using_side=bool(base.refine_axis_using_side),
        axis_refine_iters=int(base.axis_refine_iters),
        min_side_points=int(base.min_side_points),
        refine_axis_from_normals=bool(base.refine_axis_from_normals),
        normal_radius_mult=float(base.normal_radius_mult),
        normal_max_nn=int(base.normal_max_nn),
    )


def _clear_latest_queue(queue_obj) -> None:
    clear = getattr(queue_obj, "clear", None)
    if callable(clear):
        clear()
        return
    while True:
        message = queue_obj.tryGet()
        if message is None:
            return


def _print_receiver_stats(pipeline) -> None:
    status = pipeline.get_status()
    sync_ms = status.get("last_sync_delta_ms", None)
    if sync_ms is not None:
        print(f"Last local RGB/depth pairing delta: {float(sync_ms):.1f} ms")
    print(f"Receiver stats: rgb_frames={status['rgb_frames']} pcl_frames={status['pcl_frames']}")


def _build_post_capture_ranked_preview(
    *,
    pipeline,
    pcl_q,
    sam_runtime,
    config: SamRunnerConfig,
    timeout_s: float = 1.5,
) -> tuple[np.ndarray | None, str]:
    latest_pcl = None
    deadline = time.monotonic() + max(0.05, float(timeout_s))
    while time.monotonic() < deadline:
        if not pipeline.isRunning():
            break
        message = pcl_q.tryGet()
        if message is not None:
            latest_pcl = message
            break
        time.sleep(0.01)

    if latest_pcl is None:
        return None, "Final candidate preview unavailable"

    frame_bgr = latest_pcl.getCvFrame()
    points, colors = latest_pcl.getPointsRGB()

    sam_runtime.clear()
    sam_runtime.resume()
    try:
        sam_runtime.submit_frame(frame_bgr)
        latest_instances: list[np.ndarray] = []
        while time.monotonic() < deadline:
            latest_instances, _sam_fps = sam_runtime.get_latest_instances()
            if latest_instances:
                break
            time.sleep(0.01)
    finally:
        sam_runtime.pause()

    if not latest_instances:
        return np.asarray(frame_bgr, dtype=np.uint8).copy(), "Final candidate preview (no detections)"

    ranked_candidates, selected_candidate = select_best_candidate(
        latest_instances,
        points,
        colors,
        min_points=int(config.min_points),
        config=config,
    )
    if not ranked_candidates:
        return np.asarray(frame_bgr, dtype=np.uint8).copy(), "Final candidate preview (no valid candidates)"

    overlays = _candidate_overlays(
        ranked_candidates,
        selected=selected_candidate,
        rgb_hw=frame_bgr.shape[:2],
    )
    preview_bgr = make_segmented_preview_bgr(
        frame_bgr,
        mask_flat=None,
        instance_overlays=overlays,
    )
    return preview_bgr, f"Final candidate preview ({len(overlays)} candidates)"


def _handle_capture_result(
    *,
    samples,
    preview_bgr,
    config: SamRunnerConfig,
    pipeline,
    show_result: bool,
) -> np.ndarray | None:
    robust_pitch_deg, robust_yaw_deg = _compute_robust_pitch_yaw_from_samples(
        samples,
        repeats=int(config.orient_median_repeats),
        frames_per_batch=int(config.valid_frames),
        config=config,
    )
    target = extract_motion_target_from_samples(
        samples,
        robust_pitch_deg=robust_pitch_deg,
        robust_yaw_deg=robust_yaw_deg,
        hardcoded_roll_deg=180.0,
        config=config,
    )

    print(f"Captured {len(samples)} valid remote segmented frames.")
    if target is None:
        print("Target extraction failed for the captured remote samples.")
    else:
        cx, cy, cz = np.asarray(target.centroid_cam_mm, dtype=np.float64)
        print(
            "Remote target:"
            f" centroid_mm=({cx:.1f}, {cy:.1f}, {cz:.1f})"
            f" pitch={target.pitch_cam_deg:.2f}"
            f" yaw={target.yaw_cam_deg:.2f}"
            f" roll={target.roll_cam_deg:.2f}"
        )
        if target.minor_axis_mm is not None:
            print(f"  minor_axis_mm={float(target.minor_axis_mm):.1f}")
        if robust_pitch_deg is not None and robust_yaw_deg is not None:
            print(f"  robust_pitch_yaw_deg=({float(robust_pitch_deg):.2f}, {float(robust_yaw_deg):.2f})")

    if bool(show_result):
        merged_full_points = np.concatenate([sample.full_points_mm for sample in samples], axis=0)
        merged_full_colors = np.concatenate([sample.full_colors_rgb_u8 for sample in samples], axis=0)
        merged_seg_points = np.concatenate([sample.seg_points_mm for sample in samples], axis=0)
        merged_seg_colors = np.concatenate([sample.seg_colors_rgb_u8 for sample in samples], axis=0)

        _show_result(
            full_points_mm=merged_full_points,
            full_colors_rgb_u8=merged_full_colors,
            seg_points_mm=merged_seg_points,
            seg_colors_rgb_u8=merged_seg_colors,
            tint_rgb_u8=np.asarray(config.tint_rgb_u8, dtype=np.uint8),
            tint_alpha=float(config.tint_alpha),
            voxel_scene_mm=float(config.voxel_scene_mm),
            voxel_obj_mm=float(config.voxel_obj_mm),
            max_viz_points=int(config.max_viz_points),
            segmented_preview_bgr=preview_bgr,
            robust_pitch_deg=robust_pitch_deg,
            robust_yaw_deg=robust_yaw_deg,
            side_dot_abs_max=float(config.side_dot_abs_max),
            refine_axis_using_side=bool(config.refine_axis_using_side),
            axis_refine_iters=int(config.axis_refine_iters),
            min_side_points=int(config.min_side_points),
            refine_axis_from_normals=bool(config.refine_axis_from_normals),
            normal_radius_mult=float(config.normal_radius_mult),
            normal_max_nn=int(config.normal_max_nn),
        )

    _print_receiver_stats(pipeline)
    return None if preview_bgr is None else np.asarray(preview_bgr, dtype=np.uint8).copy()


def _run_interactive_remote_capture(
    *,
    pipeline,
    rgb_q,
    pcl_q,
    config: SamRunnerConfig,
    show_result: bool,
) -> int:
    if Sam3Runtime is None:
        raise RuntimeError("SAM3 runtime is unavailable. Ensure SAM3 dependencies are installed.")

    total_valid_frames = int(config.valid_frames) * int(config.orient_median_repeats)
    sam_runtime = Sam3Runtime(prompt=str(config.prompt), multi_instance=True)
    sam_runtime.start()
    sam_runtime.pause()

    preview_bgr: np.ndarray | None = None
    preview_status_text = "Live preview"
    preview_state_text = "STATE: READY"
    last_fps_t = time.monotonic()
    cam_fps = 0.0
    frame_ctr = 0

    print("Remote receiver ready. Live preview started.")
    print(f"Window controls ({CAPTURE_WINDOW_NAME}): T/C=capture, Q/ESC=quit")
    try:
        while pipeline.isRunning():
            in_rgb = rgb_q.tryGet()
            if in_rgb is None:
                time.sleep(0.002)
                continue

            frame_ctr += 1
            now_t = time.monotonic()
            dt = now_t - last_fps_t
            if dt >= 0.5:
                cam_fps = float(frame_ctr / max(dt, 1e-6))
                frame_ctr = 0
                last_fps_t = now_t

            live_frame_bgr = in_rgb.getCvFrame()
            showing_preview = preview_bgr is not None
            controls_text = "T/C: recapture  Q/ESC: quit" if showing_preview else "T/C: capture  Q/ESC: quit"
            key = show_capture_frame_key(
                frame_bgr=preview_bgr if showing_preview else live_frame_bgr,
                mask_flat=None,
                cam_fps=cam_fps,
                sam_fps=0.0,
                score=None,
                state_text=preview_state_text if showing_preview else "STATE: READY",
                status_text=preview_status_text if showing_preview else "Live preview",
                controls_text=controls_text,
                window_name=CAPTURE_WINDOW_NAME,
            )
            if is_capture_quit_key(key):
                print("Exiting interactive remote capture.")
                return 0
            if key not in (ord("t"), ord("T"), ord("c"), ord("C")):
                continue

            print("Starting remote SAM3 capture.")
            print(
                f"Capture plan: {config.orient_median_repeats} x {config.valid_frames} valid frames "
                f"({total_valid_frames} total)"
            )
            preview_bgr = None
            preview_status_text = "Live preview"
            preview_state_text = "STATE: READY"
            _clear_latest_queue(rgb_q)
            _clear_latest_queue(pcl_q)

            action, samples, preview_bgr = _capture_segmented_samples(
                pipeline=pipeline,
                rgb_q=rgb_q,
                pcl_q=pcl_q,
                prompt=str(config.prompt),
                valid_frames_target=total_valid_frames,
                min_points=int(config.min_points),
                config=config,
                sam_runtime=sam_runtime,
                destroy_capture_window_flag=False,
            )

            if action == "quit":
                print("Capture cancelled by user.")
                return 0
            if not samples:
                print("No valid remote RGB-D samples were captured.")
                preview_status_text = "Capture preview"
                preview_state_text = "STATE: READY"
                preview_bgr = None if preview_bgr is None else np.asarray(preview_bgr, dtype=np.uint8).copy()
                print(f"Window controls ({CAPTURE_WINDOW_NAME}): T/C=capture, Q/ESC=quit")
                continue

            _handle_capture_result(
                samples=samples,
                preview_bgr=preview_bgr,
                config=config,
                pipeline=pipeline,
                show_result=show_result,
            )
            preview_bgr, preview_status_text = _build_post_capture_ranked_preview(
                pipeline=pipeline,
                pcl_q=pcl_q,
                sam_runtime=sam_runtime,
                config=config,
            )
            preview_state_text = "STATE: COMPLETE" if preview_bgr is not None else "STATE: READY"
            print(f"Window controls ({CAPTURE_WINDOW_NAME}): T/C=recapture, Q/ESC=quit")

        status = pipeline.get_status()
        fatal_error = status.get("fatal_error")
        if fatal_error:
            print(f"Remote receiver stopped: {fatal_error}")
            return 1
        print("Remote receiver stopped.")
        return 0
    finally:
        destroy_capture_window(CAPTURE_WINDOW_NAME)
        _release_sam_runtime(sam_runtime)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remote SAM3 validation using RTSP RGB + UDP depth, matched to the wired OAK-D capture flow."
    )

    ap.add_argument("--rtsp-url", required=True, help="RTSP URL from the edge sender, e.g. rtsp://<ip>:8554/oak")
    ap.add_argument("--rtsp-transport", choices=["udp", "tcp"], default="udp")
    ap.add_argument("--bind", default="0.0.0.0", help="UDP bind IP for incoming depth packets")
    ap.add_argument("--port", type=int, default=9000, help="UDP port for incoming depth packets")
    ap.add_argument("--startup-timeout", type=float, default=15.0, help="Seconds to wait for RGB + CAL0 readiness")
    ap.add_argument("--stale-depth-ms", type=int, default=120, help="Drop partial depth frames older than this")
    ap.add_argument("--socket-rcvbuf", type=int, default=8_388_608)
    ap.add_argument("--max-rgb-buffer", type=int, default=8, help="Buffered RGB frames used for depth pairing")
    ap.add_argument(
        "--max-rgb-depth-delta-ms",
        type=float,
        default=250.0,
        help="Warn when the locally paired RGB/depth arrival gap exceeds this value",
    )
    ap.add_argument("--min-mm", type=int, default=120, help="Minimum usable depth in millimeters")
    ap.add_argument("--max-mm", type=int, default=2500, help="Maximum usable depth in millimeters")
    ap.add_argument("--prompt", type=str, default=DEFAULT_SAM_RUNNER_CONFIG.prompt)
    ap.add_argument("--valid-frames", type=int, default=DEFAULT_SAM_RUNNER_CONFIG.valid_frames)
    ap.add_argument("--orient-median-repeats", type=int, default=DEFAULT_SAM_RUNNER_CONFIG.orient_median_repeats)
    ap.add_argument("--min-points", type=int, default=DEFAULT_SAM_RUNNER_CONFIG.min_points)
    add_bool_arg(ap, "show-result", True, "Open the final Open3D/OpenCV result windows after capture")
    add_bool_arg(ap, "verbose", True, "Print receiver diagnostics")
    args = ap.parse_args()

    config = _build_config(args)
    config.validate()
    total_valid_frames = int(config.valid_frames) * int(config.orient_median_repeats)

    pipeline = build_remote_oakd_pipeline(
        rtsp_url=args.rtsp_url,
        depth_bind_ip=args.bind,
        depth_port=args.port,
        rtsp_transport=args.rtsp_transport,
        stale_depth_ms=args.stale_depth_ms,
        socket_rcvbuf=args.socket_rcvbuf,
        max_rgb_buffer=args.max_rgb_buffer,
        max_rgb_depth_delta_ms=args.max_rgb_depth_delta_ms,
        min_mm=args.min_mm,
        max_mm=args.max_mm,
        verbose=bool(args.verbose),
    )
    rgb_q, pcl_q = create_remote_oakd_queues(pipeline)

    print("Remote SAM3 test starting.")
    print(f"  RTSP: {args.rtsp_url} ({args.rtsp_transport})")
    print(f"  UDP depth: {args.bind}:{args.port}")
    print(f"  Prompt: {config.prompt}")
    print(
        f"  Capture plan: {config.orient_median_repeats} x {config.valid_frames} valid frames "
        f"({total_valid_frames} total)"
    )

    pipeline.start()
    try:
        if not pipeline.wait_until_ready(timeout_s=float(args.startup_timeout)):
            status = pipeline.get_status()
            print("Remote receiver did not become ready in time.")
            print(f"Status: {status}")
            return 1

        return _run_interactive_remote_capture(
            pipeline=pipeline,
            rgb_q=rgb_q,
            pcl_q=pcl_q,
            config=config,
            show_result=bool(args.show_result),
        )
    finally:
        pipeline.stop()
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
