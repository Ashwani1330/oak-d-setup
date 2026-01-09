import os
import sys
import time
import requests
from dataclasses import dataclass

# --- CRITICAL: set HF cache before any HF/model loads ---
os.environ["HF_HOME"] = r"D:\sam3\huggingface_cache"

import cv2
import torch
import numpy as np
from PIL import Image
import depthai as dai  # DepthAI v3

# =========================
# OPTIONAL: Open3D viewer
# =========================
try:
    import open3d as o3d
    OPEN3D_OK = True
except Exception:
    o3d = None
    OPEN3D_OK = False

# =========================
# CONFIG
# =========================
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"

# Recommended for OAK-D stereo: 640x400 (native-ish for mono cams)
# You can switch to 640x480 if you prefer, but depth will be scaled/aligned accordingly.
RGB_RES    = (640, 400)   # (w, h) what SAM3 sees
STEREO_RES = (640, 400)   # (w, h) what StereoDepth consumes

FPS_RGB   = 30
FPS_DEPTH = 30

INFERENCE_INTERVAL = 1            # run SAM3 every N frames
CONFIDENCE_THRESH = 0.20          # per-mask threshold
MASK_ALPHA = 0.45

# Depth filtering (mm)
DEPTH_MIN_MM = 120
DEPTH_MAX_MM = 2500

# Sphere/ring visualization radius in meters -> converted to pixels based on Z
SPHERE_RADIUS_M = 0.05  # 5 cm

# Point cloud
ENABLE_POINTCLOUD = True          # set False to skip Open3D
CLOUD_UPDATE_INTERVAL = 2         # update Open3D every N frames
SCENE_STRIDE = 4                  # downsample scene cloud (higher = faster)
MAX_SCENE_POINTS = 120_000        # safety cap
MAX_OBJ_POINTS   = 40_000         # safety cap

# Stereo tuning
EXTENDED_DISPARITY = False        # improves close range but changes tradeoffs
SUBPIXEL = False
LR_CHECK = True

# Misc
DISPLAY_SCALE = 1.0
SHOW_DEPTH_DEBUG_DEFAULT = False

# =========================
# SAM3 repo path setup
# =========================
current_file_path = os.path.abspath(__file__)
sam3_repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "..", ".."))
if sam3_repo_root not in sys.path:
    sys.path.append(sam3_repo_root)

# Ensure tokenizer exists
assets_dir = os.path.join(sam3_repo_root, "assets")
bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    print(f"[AUTO-FIX] Downloading tokenizer to {bpe_path} ...")
    os.makedirs(assets_dir, exist_ok=True)
    r = requests.get(TOKENIZER_URL, timeout=30)
    r.raise_for_status()
    with open(bpe_path, "wb") as f:
        f.write(r.content)

# SAM3 imports (your repo)
try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# =========================
# Helpers
# =========================

# BGR palette (repeat if more objects)
PALETTE = [
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (0, 165, 255),   # orange
    (128, 0, 255),   # purple-ish
]

@dataclass
class DetectedObject:
    idx: int
    score: float
    mask_bool: np.ndarray               # (H,W) bool
    centroid_uv: tuple[int, int] | None # (u,v)
    centroid_xyz_m: np.ndarray | None   # (3,) meters (X,Y,Z)
    points_xyz_m: np.ndarray | None     # (N,3) meters


def read_rgb_intrinsics_for_res(w: int, h: int):
    """
    Reads RGB intrinsics (fx, fy, cx, cy) for CAM_A at resolution (w,h).
    """
    with dai.Device() as device:
        calib = device.readCalibration()
        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return fx, fy, cx, cy


def compute_mask_centroid_uv(mask_bool: np.ndarray):
    m = (mask_bool.astype(np.uint8) * 255)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] <= 0:
        return None
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return (u, v)


def backproject_points_from_mask(mask_bool: np.ndarray,
                                 depth_mm: np.ndarray,
                                 fx: float, fy: float, cx: float, cy: float,
                                 max_points: int):
    """
    Returns (centroid_xyz_m, points_xyz_m) for this mask using aligned depth.
    """
    valid = mask_bool & (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM) & (depth_mm > 0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None, None

    if xs.size > max_points:
        pick = np.random.choice(xs.size, size=max_points, replace=False)
        xs = xs[pick]
        ys = ys[pick]

    z_m = depth_mm[ys, xs].astype(np.float32) / 1000.0
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)

    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    pts = np.stack([X, Y, z_m], axis=1)  # camera frame: X right, Y down, Z forward

    centroid = pts.mean(axis=0)
    return centroid, pts


def overlay_masks_and_centroids(frame_bgr: np.ndarray,
                                detections: list[DetectedObject],
                                fx: float,
                                sphere_radius_m: float,
                                alpha: float):
    """
    Overlays per-object mask with unique color and draws per-object "sphere" ring at centroid.
    """
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    for k, det in enumerate(detections):
        color = PALETTE[k % len(PALETTE)]
        mask = det.mask_bool
        if mask is None or not np.any(mask):
            continue

        # mask overlay
        color_layer = np.zeros_like(vis, dtype=np.uint8)
        color_layer[:, :] = color
        roi = vis[mask]
        vis[mask] = cv2.addWeighted(roi, 1.0 - alpha, color_layer[mask], alpha, 0)

        # contour
        m255 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

        # centroid + sphere
        if det.centroid_uv is not None:
            u, v = det.centroid_uv
            if 0 <= u < W and 0 <= v < H:
                if det.centroid_xyz_m is not None:
                    Z = float(det.centroid_xyz_m[2])
                    r_px = int(max(3, min(220, (fx * sphere_radius_m) / max(Z, 1e-3))))
                else:
                    r_px = 18

                cv2.circle(vis, (u, v), r_px, color, 2, cv2.LINE_AA)
                cv2.circle(vis, (u, v), 3, color, -1, cv2.LINE_AA)

                label = f"#{k} s={det.score:.2f}"
                if det.centroid_xyz_m is not None:
                    X, Y, Z = det.centroid_xyz_m
                    label += f" Z={Z:.2f}m"
                cv2.putText(vis, label, (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def depth_colormap(depth_mm: np.ndarray):
    d = depth_mm.copy()
    d[(d < DEPTH_MIN_MM) | (d > DEPTH_MAX_MM)] = 0
    if d.max() <= 0:
        return None
    d8 = np.clip((d.astype(np.float32) - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM), 0, 1)
    d8 = (d8 * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def make_scene_cloud(depth_mm: np.ndarray,
                     rgb_bgr: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     stride: int,
                     max_points: int):
    """
    Builds a downsampled colored point cloud of the entire depth frame.
    Returns (pts_xyz_m, colors_rgb_0to1)
    """
    H, W = depth_mm.shape[:2]

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.reshape(-1)
    y = grid_y.reshape(-1)

    z = depth_mm[y, x].astype(np.float32)
    valid = (z >= DEPTH_MIN_MM) & (z <= DEPTH_MAX_MM) & (z > 0)
    x = x[valid]
    y = y[valid]
    z = z[valid] / 1000.0

    if x.size == 0:
        return None, None

    if x.size > max_points:
        pick = np.random.choice(x.size, size=max_points, replace=False)
        x, y, z = x[pick], y[pick], z[pick]

    u = x.astype(np.float32)
    v = y.astype(np.float32)

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    pts = np.stack([X, Y, z], axis=1)

    # colors from RGB
    cols = rgb_bgr[y, x][:, ::-1].astype(np.float32) / 255.0  # BGR->RGB

    return pts, cols


class Open3DViewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("OAK-D PointCloud (Scene + Segments)", width=1000, height=700)

        self.pcd_scene = o3d.geometry.PointCloud()
        self.pcd_obj = o3d.geometry.PointCloud()
        self.pcd_centroids = o3d.geometry.PointCloud()

        self.vis.add_geometry(self.pcd_scene)
        self.vis.add_geometry(self.pcd_obj)
        self.vis.add_geometry(self.pcd_centroids)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(axis)

        # Improve initial viewpoint a bit
        self._inited = False

    def update(self, scene_pts, scene_cols, obj_pts, obj_cols, centroid_pts):
        if scene_pts is not None:
            self.pcd_scene.points = o3d.utility.Vector3dVector(scene_pts.astype(np.float64))
            self.pcd_scene.colors = o3d.utility.Vector3dVector(scene_cols.astype(np.float64))

        if obj_pts is not None and len(obj_pts) > 0:
            self.pcd_obj.points = o3d.utility.Vector3dVector(obj_pts.astype(np.float64))
            self.pcd_obj.colors = o3d.utility.Vector3dVector(obj_cols.astype(np.float64))
        else:
            self.pcd_obj.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.pcd_obj.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if centroid_pts is not None and len(centroid_pts) > 0:
            self.pcd_centroids.points = o3d.utility.Vector3dVector(centroid_pts.astype(np.float64))
            self.pcd_centroids.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(centroid_pts), 1))
            )
        else:
            self.pcd_centroids.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.pcd_centroids.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        self.vis.update_geometry(self.pcd_scene)
        self.vis.update_geometry(self.pcd_obj)
        self.vis.update_geometry(self.pcd_centroids)

        # Poll + render
        self.vis.poll_events()
        self.vis.update_renderer()

        if not self._inited:
            try:
                self.vis.get_view_control().set_zoom(0.6)
            except Exception:
                pass
            self._inited = True

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


def main():
    print("--- SAM3 + OAK-D: Multi-object centroids + point cloud ---")
    print(f"HF cache: {os.environ.get('HF_HOME')}")

    device_torch = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM3 device: {device_torch}")

    if device_torch == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Intrinsics for CAM_A at RGB_RES
    fx, fy, cx, cy = read_rgb_intrinsics_for_res(RGB_RES[0], RGB_RES[1])
    print(f"RGB intrinsics @ {RGB_RES}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Load SAM3
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=None).to(device_torch)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESH)
    print("SAM3 loaded.")

    text_prompt = input("\nObject to segment (e.g. person, bottle): ").strip() or "person"
    print("Controls: q=quit, t=change prompt, d=toggle depth debug window")

    show_depth_debug = SHOW_DEPTH_DEBUG_DEFAULT

    # Open3D viewer
    viewer = None
    if ENABLE_POINTCLOUD and OPEN3D_OK:
        viewer = Open3DViewer()
        print("[Open3D] Viewer started.")
    elif ENABLE_POINTCLOUD and not OPEN3D_OK:
        print("[Open3D] Not installed. Run: pip install open3d  (continuing without 3D view)")

    # DepthAI v3 pipeline
    with dai.Pipeline() as pipeline:
        # RGB camera
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # IMPORTANT:
        # For 3D math, it's usually safer to NOT undistort on-host unless you also use undistorted intrinsics.
        enable_undistortion = False

        rgb_out = cam_rgb.requestOutput(
            RGB_RES,
            dai.ImgFrame.Type.BGR888p,
            dai.ImgResizeMode.LETTERBOX,
            FPS_RGB,
            enable_undistortion,
        )

        # Stereo cameras
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        left_out = cam_left.requestOutput(STEREO_RES, fps=FPS_DEPTH)
        right_out = cam_right.requestOutput(STEREO_RES, fps=FPS_DEPTH)

        stereo = pipeline.create(dai.node.StereoDepth)

        # Stereo knobs
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
        except Exception:
            pass

        try:
            stereo.setLeftRightCheck(bool(LR_CHECK))
        except Exception:
            pass

        if EXTENDED_DISPARITY:
            try:
                stereo.setExtendedDisparity(True)
            except Exception:
                pass

        try:
            stereo.setSubpixel(bool(SUBPIXEL))
        except Exception:
            pass

        # Align depth to RGB camera and output at RGB_RES
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(RGB_RES[0], RGB_RES[1])

        # Range gating
        try:
            cfg = stereo.initialConfig
            cfg.postProcessing.thresholdFilter.minRange = int(DEPTH_MIN_MM)
            cfg.postProcessing.thresholdFilter.maxRange = int(DEPTH_MAX_MM)
        except Exception:
            pass

        left_out.link(stereo.left)
        right_out.link(stereo.right)

        # Queues
        try:
            q_rgb = rgb_out.createOutputQueue(maxSize=1, blocking=False)
        except TypeError:
            q_rgb = rgb_out.createOutputQueue()

        try:
            q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
        except TypeError:
            q_depth = stereo.depth.createOutputQueue()

        pipeline.start()

        frame_count = 0
        last_depth_mm = None
        last_detections: list[DetectedObject] = []

        fps_vis = 0.0
        t_prev = time.time()

        use_amp = (device_torch == "cuda")

        while pipeline.isRunning():
            rgb_msg = q_rgb.tryGet() if hasattr(q_rgb, "tryGet") else q_rgb.get()
            if rgb_msg is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            frame_bgr = rgb_msg.getCvFrame()

            dmsg = q_depth.tryGet() if hasattr(q_depth, "tryGet") else None
            if dmsg is not None:
                last_depth_mm = dmsg.getFrame()
                if last_depth_mm.shape[:2] != frame_bgr.shape[:2]:
                    last_depth_mm = cv2.resize(
                        last_depth_mm,
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

            # ---- Run SAM3 every N frames
            if frame_count % INFERENCE_INTERVAL == 0:
                try:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb)

                    if use_amp:
                        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                    else:
                        amp_ctx = torch.autocast(device_type="cpu", enabled=False)

                    with torch.inference_mode(), amp_ctx:
                        state = processor.set_image(pil_image)
                        processor.reset_all_prompts(state)
                        result = processor.set_text_prompt(state=state, prompt=text_prompt)

                    scores = result.get("scores")
                    masks = result.get("masks")

                    detections: list[DetectedObject] = []
                    if scores is not None and masks is not None and len(scores) > 0:
                        scores_f = scores.flatten()
                        valid_idx = torch.where(scores_f > CONFIDENCE_THRESH)[0]

                        for out_i, mi in enumerate(valid_idx.tolist()):
                            score = float(scores_f[mi].item())
                            m = masks[mi].detach().cpu().numpy()
                            if m.ndim == 3:
                                m = m[0]

                            # Resize mask to match frame if needed
                            if m.shape != frame_bgr.shape[:2]:
                                m = cv2.resize(m.astype(np.float32), (frame_bgr.shape[1], frame_bgr.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)

                            mask_bool = (m > 0)
                            centroid_uv = compute_mask_centroid_uv(mask_bool)

                            centroid_xyz = None
                            pts_xyz = None
                            if last_depth_mm is not None and np.any(mask_bool):
                                centroid_xyz, pts_xyz = backproject_points_from_mask(
                                    mask_bool, last_depth_mm, fx, fy, cx, cy, max_points=MAX_OBJ_POINTS
                                )

                            detections.append(DetectedObject(
                                idx=out_i,
                                score=score,
                                mask_bool=mask_bool,
                                centroid_uv=centroid_uv,
                                centroid_xyz_m=centroid_xyz,
                                points_xyz_m=pts_xyz
                            ))

                    last_detections = detections

                except Exception:
                    # keep previous detections if SAM3 hiccups
                    pass

            # ---- Draw overlays
            vis = overlay_masks_and_centroids(frame_bgr, last_detections, fx, SPHERE_RADIUS_M, MASK_ALPHA)

            # ---- FPS
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_vis = 0.9 * fps_vis + 0.1 * (1.0 / dt)

            cv2.putText(
                vis,
                f"Prompt: {text_prompt} | FPS: {fps_vis:.1f} | objs: {len(last_detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ---- Depth debug windows
            if show_depth_debug and last_depth_mm is not None:
                cmap = depth_colormap(last_depth_mm)
                if cmap is not None:
                    cv2.imshow("Depth (aligned)", cmap)

                if len(last_detections) > 0:
                    # show masked depth for first object as quick sanity check
                    obj_depth = last_depth_mm.copy()
                    obj_depth[~last_detections[0].mask_bool] = 0
                    cmap_obj = depth_colormap(obj_depth)
                    if cmap_obj is not None:
                        cv2.imshow("Depth (masked obj #0)", cmap_obj)

            # ---- Open3D point cloud update (separate visualization)
            if viewer is not None and last_depth_mm is not None and (frame_count % CLOUD_UPDATE_INTERVAL == 0):
                try:
                    scene_pts, scene_cols = make_scene_cloud(
                        last_depth_mm, frame_bgr, fx, fy, cx, cy,
                        stride=SCENE_STRIDE,
                        max_points=MAX_SCENE_POINTS
                    )

                    # Merge all object points; color them per object
                    obj_pts_list = []
                    obj_col_list = []
                    centroid_list = []

                    for k, det in enumerate(last_detections):
                        if det.points_xyz_m is not None and len(det.points_xyz_m) > 0:
                            pts = det.points_xyz_m
                            # color per object (Open3D expects RGB 0..1)
                            bgr = np.array(PALETTE[k % len(PALETTE)], dtype=np.float32) / 255.0
                            rgb_col = bgr[::-1]  # to RGB
                            cols = np.tile(rgb_col[None, :], (len(pts), 1))
                            obj_pts_list.append(pts)
                            obj_col_list.append(cols)

                        if det.centroid_xyz_m is not None:
                            centroid_list.append(det.centroid_xyz_m[None, :])

                    obj_pts = np.concatenate(obj_pts_list, axis=0) if len(obj_pts_list) else None
                    obj_cols = np.concatenate(obj_col_list, axis=0) if len(obj_col_list) else None
                    cent_pts = np.concatenate(centroid_list, axis=0) if len(centroid_list) else None

                    viewer.update(scene_pts, scene_cols, obj_pts, obj_cols, cent_pts)
                except Exception:
                    pass

            # ---- Display RGB
            if DISPLAY_SCALE != 1.0:
                vis = cv2.resize(vis, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("SAM3 + OAK-D (Multi centroids)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                new_prompt = input("\nNew prompt: ").strip()
                if new_prompt:
                    text_prompt = new_prompt
                    print(f"Updated prompt -> '{text_prompt}'")
            if key == ord("d"):
                show_depth_debug = not show_depth_debug
                if not show_depth_debug:
                    for wname in ["Depth (aligned)", "Depth (masked obj #0)"]:
                        try:
                            cv2.destroyWindow(wname)
                        except Exception:
                            pass

            frame_count += 1

    if viewer is not None:
        viewer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
