import os
import sys
import time
import threading
import requests
from contextlib import nullcontext
from datetime import timedelta

# --- CRITICAL: set HF cache before any HF/model loads ---
os.environ["HF_HOME"] = r"D:\sam3\huggingface_cache"

import cv2
import torch
import numpy as np
from PIL import Image
import depthai as dai

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
# CONFIG (keep yours)
# =========================
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"

RGB_RES    = (640, 400)
STEREO_RES = (640, 400)
FPS_RGB    = 30
FPS_DEPTH  = 30

TARGET_INFER_FPS   = 6.0
SAM_INPUT_RES      = (512, 320)    # keep same aspect ratio as 640x400

CONFIDENCE_THRESH  = 0.20
MAX_DETECTIONS     = 6
NMS_IOU_THRESH     = 0.55
MISS_TOLERANCE     = 2

DEPTH_MIN_MM = 120
DEPTH_MAX_MM = 2500

KEEP_LARGEST_COMPONENT = True
MIN_MASK_AREA_PX       = 120

# Pick rule:
# If camera is ABOVE tray, "highest" object is closest to camera => MIN z
PICK_RULE = "min_z"  # "min_z" (closest) or "max_z" (farthest)

# Pointcloud
ENABLE_POINTCLOUD     = True
SHOW_BG_CLOUD         = False
O3D_POINT_SIZE        = 5.0
CLOUD_UPDATE_HZ       = 10.0

SCENE_STRIDE      = 10
MAX_SCENE_POINTS  = 60_000
MAX_OBJ_POINTS    = 40_000
MIN_OBJ_DEPTH_PTS = 120

# centroid marker
CENTROID_SPHERE_R_M   = 0.008
CENTROID_SPHERE_RES   = 14

# Palette (BGR) for 2D drawing
PALETTE_BGR = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 165, 255),
    (128, 0, 255),
]

# =========================
# SAM3 repo path setup
# =========================
current_file_path = os.path.abspath(__file__)
sam3_repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "..", ".."))
if sam3_repo_root not in sys.path:
    sys.path.append(sam3_repo_root)

assets_dir = os.path.join(sam3_repo_root, "assets")
bpe_path = os.path.join(assets_dir, "bpe_simple_vocab_16e6.txt.gz")
if not os.path.exists(bpe_path):
    print(f"[AUTO-FIX] Downloading tokenizer to {bpe_path} ...")
    os.makedirs(assets_dir, exist_ok=True)
    r = requests.get(TOKENIZER_URL, timeout=30)
    r.raise_for_status()
    with open(bpe_path, "wb") as f:
        f.write(r.content)

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# =========================
# Helpers
# =========================
def read_rgb_intrinsics_for_res(w: int, h: int):
    with dai.Device() as device:
        calib = device.readCalibration()
        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h), dtype=np.float32)
    fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])
    return fx, fy, cx, cy

def clean_mask_keep_largest(mask_bool: np.ndarray):
    if not KEEP_LARGEST_COMPONENT:
        return mask_bool
    m = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_bool
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    return (labels == k)

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)

def select_topk_with_nms(scores_f, masks, out_hw):
    """
    Takes SAM3 outputs at SAM_INPUT_RES size and returns a list of boolean masks at full RGB size.
    """
    H, W = out_hw
    idxs = torch.where(scores_f > CONFIDENCE_THRESH)[0].tolist()
    idxs.sort(key=lambda i: float(scores_f[i].item()), reverse=True)

    picked = []
    for mi in idxs:
        m = masks[mi].detach().cpu().numpy()
        if m.ndim == 3:
            m = m[0]  # [1,H,W] -> [H,W]

        # resize to full output size
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        mask_bool = (m > 0)
        if int(mask_bool.sum()) < MIN_MASK_AREA_PX:
            continue

        if KEEP_LARGEST_COMPONENT:
            mask_bool = clean_mask_keep_largest(mask_bool)
            if int(mask_bool.sum()) < MIN_MASK_AREA_PX:
                continue

        # NMS (remove duplicates)
        dup = False
        for kept in picked:
            if mask_iou(mask_bool, kept) >= NMS_IOU_THRESH:
                dup = True
                break
        if dup:
            continue

        picked.append(mask_bool)
        if len(picked) >= MAX_DETECTIONS:
            break

    return picked

def stable_subsample(xs, ys, max_points: int):
    n = xs.size
    if n <= max_points:
        return xs, ys
    step = int(np.ceil(n / max_points))
    return xs[::step], ys[::step]

def backproject_from_pixels(xs, ys, depth_mm, fx, fy, cx, cy):
    z_m = depth_mm[ys, xs].astype(np.float32) / 1000.0
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)
    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    return np.stack([X, Y, z_m], axis=1)

def make_scene_cloud(depth_mm, fx, fy, cx, cy, stride, max_points):
    H, W = depth_mm.shape[:2]
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    gx, gy = np.meshgrid(xs, ys)
    x = gx.reshape(-1)
    y = gy.reshape(-1)

    z = depth_mm[y, x].astype(np.float32)
    valid = (z >= DEPTH_MIN_MM) & (z <= DEPTH_MAX_MM) & (z > 0)
    x = x[valid]; y = y[valid]; z = z[valid]

    if x.size == 0:
        return None, None

    if x.size > max_points:
        step = int(np.ceil(x.size / max_points))
        x = x[::step]; y = y[::step]; z = z[::step]

    z_m = z / 1000.0
    X = (x.astype(np.float32) - cx) * z_m / fx
    Y = (y.astype(np.float32) - cy) * z_m / fy
    pts = np.stack([X, Y, z_m], axis=1)

    cols = depth_mm_to_rgb01(z)  # depth-colored
    return pts, cols

def make_object_cloud(mask_bool, depth_mm, fx, fy, cx, cy):
    valid = mask_bool & (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM) & (depth_mm > 0)
    ys, xs = np.nonzero(valid)
    if xs.size < MIN_OBJ_DEPTH_PTS:
        return None, None, None

    xs, ys = stable_subsample(xs, ys, max_points=MAX_OBJ_POINTS)
    pts = backproject_from_pixels(xs, ys, depth_mm, fx, fy, cx, cy)
    centroid = pts.mean(axis=0)

    # color points by depth (z)
    z_mm = (pts[:, 2] * 1000.0).astype(np.float32)
    cols = depth_mm_to_rgb01(z_mm)
    return pts, cols, centroid

def pca_major_minor_xy(pts_xyz: np.ndarray):
    """
    Very simple axis estimate:
    - Take XY from 3D points
    - PCA -> major axis (v1), minor axis (v2)
    - Also estimate lengths along those axes
    """
    xy = pts_xyz[:, :2].astype(np.float32)
    c = xy.mean(axis=0)
    xyc = xy - c[None, :]

    C = (xyc.T @ xyc) / max(1, (xyc.shape[0] - 1))
    evals, evecs = np.linalg.eigh(C)  # sorted small->large
    v_major = evecs[:, 1]
    v_minor = evecs[:, 0]

    # lengths: robust spread using percentiles
    proj_major = xyc @ v_major
    proj_minor = xyc @ v_minor
    maj_len = float(np.percentile(proj_major, 95) - np.percentile(proj_major, 5))
    min_len = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))

    yaw_rad = float(np.arctan2(v_major[1], v_major[0]))  # in camera XY plane
    return c, v_major, v_minor, maj_len, min_len, yaw_rad

# -------------------------
# Depth colormap for pointcloud colors (fast LUT)
# -------------------------
_depth_lut_rgb01 = None

def _build_depth_lut():
    # 256x1 grayscale -> applyColorMap -> RGB
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(256, 3)
    rgb = bgr[:, ::-1].astype(np.float32) / 255.0
    return rgb

def depth_mm_to_rgb01(depth_mm_values):
    """
    depth_mm_values: array-like float/int in millimeters
    returns Nx3 float colors in [0,1]
    """
    global _depth_lut_rgb01
    if _depth_lut_rgb01 is None:
        _depth_lut_rgb01 = _build_depth_lut()

    d = np.asarray(depth_mm_values, dtype=np.float32)
    t = (d - DEPTH_MIN_MM) / max(1e-6, (DEPTH_MAX_MM - DEPTH_MIN_MM))
    t = np.clip(t, 0.0, 1.0)
    idx = (t * 255.0).astype(np.uint8)
    return _depth_lut_rgb01[idx]

# =========================
# Open3D viewer (same idea as yours)
# =========================
class MultiObjectOpen3DViewer:
    def __init__(self, max_centroids: int):
        self.max_centroids = max_centroids
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("OAK-D PointCloud (depth-colored)", width=1200, height=800)

        opt = self.vis.get_render_option()
        opt.background_color = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        opt.point_size = float(O3D_POINT_SIZE)

        self.pcd_scene = o3d.geometry.PointCloud()
        self.pcd_obj   = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_scene, reset_bounding_box=True)
        self.vis.add_geometry(self.pcd_obj,   reset_bounding_box=False)

        # fixed-radius spheres
        self.spheres = []
        self.base_verts = []
        for _ in range(max_centroids):
            s = o3d.geometry.TriangleMesh.create_sphere(
                radius=float(CENTROID_SPHERE_R_M),
                resolution=int(CENTROID_SPHERE_RES)
            )
            s.compute_vertex_normals()
            s.paint_uniform_color([0.0, 0.0, 0.0])
            self.spheres.append(s)
            self.base_verts.append(np.asarray(s.vertices).copy())
            self.vis.add_geometry(s, reset_bounding_box=False)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(axis, reset_bounding_box=False)

        self._did_fit = False

    def set_point_size(self, ps: float):
        self.vis.get_render_option().point_size = float(ps)

    def reset_view(self):
        self.vis.reset_view_point(True)
        self._did_fit = True

    def _set_sphere(self, i: int, center_xyz, color_rgb):
        s = self.spheres[i]
        base = self.base_verts[i]
        if center_xyz is None:
            s.paint_uniform_color([0.0, 0.0, 0.0])
            s.vertices = o3d.utility.Vector3dVector(base.astype(np.float64))
            return
        v = base.astype(np.float64) + center_xyz[None, :].astype(np.float64)
        s.vertices = o3d.utility.Vector3dVector(v)
        s.paint_uniform_color(color_rgb.tolist())
        s.compute_vertex_normals()

    def update(self, scene_pts, scene_cols, obj_pts, obj_cols, centroids_xyz, centroid_cols):
        # scene
        if scene_pts is not None and len(scene_pts) > 0:
            self.pcd_scene.points = o3d.utility.Vector3dVector(scene_pts.astype(np.float64))
            self.pcd_scene.colors = o3d.utility.Vector3dVector(scene_cols.astype(np.float64))
        else:
            self.pcd_scene.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.pcd_scene.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        self.vis.update_geometry(self.pcd_scene)

        # objects
        if obj_pts is not None and len(obj_pts) > 0:
            self.pcd_obj.points = o3d.utility.Vector3dVector(obj_pts.astype(np.float64))
            self.pcd_obj.colors = o3d.utility.Vector3dVector(obj_cols.astype(np.float64))
        else:
            self.pcd_obj.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.pcd_obj.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        self.vis.update_geometry(self.pcd_obj)

        n = 0 if centroids_xyz is None else min(len(centroids_xyz), self.max_centroids)
        for i in range(self.max_centroids):
            if i < n:
                self._set_sphere(i, centroids_xyz[i], centroid_cols[i])
            else:
                self._set_sphere(i, None, None)
            self.vis.update_geometry(self.spheres[i])

        if not self._did_fit and (
            (obj_pts is not None and len(obj_pts) > 0) or (scene_pts is not None and len(scene_pts) > 0)
        ):
            self.reset_view()

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass

# =========================
# Shared state
# =========================
class Shared:
    def __init__(self):
        self.lock = threading.Lock()

        # latest inputs
        self.frame_bgr = None
        self.depth_mm  = None
        self.frame_id  = 0

        self.prompt = "pipe"
        self.prompt_ver = 0

        # latest outputs
        self.contours = []          # list[list[np.ndarray]]
        self.centroids_uv = []      # list[(u,v)]
        self.centroids_xyz = None   # Nx3 (meters)
        self.colors_bgr = []        # for 2D drawing

        # for Open3D
        self.obj_pts = None
        self.obj_cols = None
        self.centroid_cols = None

        # chosen object info
        self.pick_index = -1
        self.pick_yaw_deg = 0.0
        self.pick_major_m = 0.0
        self.pick_minor_m = 0.0

        self.last_infer_ms = 0.0
        self.num_objs = 0

# =========================
# Inference worker
# =========================
class InferWorker(threading.Thread):
    def __init__(self, shared: Shared, processor: Sam3Processor, device_torch: str, fx, fy, cx, cy):
        super().__init__(daemon=True)
        self.shared = shared
        self.processor = processor
        self.device_torch = device_torch
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        self.stop_evt = threading.Event()
        self.miss_count = 0
        self.period = 1.0 / max(1e-6, TARGET_INFER_FPS)

        self.use_amp = (device_torch == "cuda")
        self.amp_ctx = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

        self.last_seen_frame_id = -1

    def stop(self):
        self.stop_evt.set()

    def run(self):
        next_t = time.perf_counter()

        while not self.stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(0.001)
                continue
            next_t = now + self.period

            with self.shared.lock:
                fid = self.shared.frame_id
                frame_bgr = self.shared.frame_bgr
                depth_mm  = self.shared.depth_mm
                prompt    = self.shared.prompt

            if frame_bgr is None or depth_mm is None:
                continue

            # only run if a newer frame arrived
            if fid == self.last_seen_frame_id:
                continue
            self.last_seen_frame_id = fid

            t0 = time.perf_counter()

            # downscale for SAM
            frame_small = cv2.resize(frame_bgr, SAM_INPUT_RES, interpolation=cv2.INTER_AREA)
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_small)

            outH, outW = frame_bgr.shape[:2]

            try:
                # SAM3 inference (simple use: set_image -> set_text_prompt)
                with torch.inference_mode(), self.amp_ctx:
                    state = self.processor.set_image(pil_image)
                    result = self.processor.set_text_prompt(state=state, prompt=prompt)

                scores = result.get("scores")
                masks  = result.get("masks")

                contours_out = []
                cent_uv_out  = []
                colors_out   = []

                obj_pts_list = []
                obj_cols_list = []
                cent_xyz_list = []
                cent_col_list = []

                pick_index = -1
                pick_yaw_deg = 0.0
                pick_major_m = 0.0
                pick_minor_m = 0.0

                # collect candidates with their Z so we can choose best
                cand = []  # (z_m, idx)

                if scores is not None and masks is not None and len(scores) > 0:
                    scores_f = scores.flatten()
                    picked_masks = select_topk_with_nms(scores_f, masks, (outH, outW))

                    # sort left->right (simple, reduces ID swapping)
                    items = []
                    for i, mask_bool in enumerate(picked_masks):
                        if int(mask_bool.sum()) < MIN_MASK_AREA_PX:
                            continue

                        # 2D centroid (fast moment) just for drawing;
                        # 3D centroid comes from depth points.
                        m255 = (mask_bool.astype(np.uint8) * 255)
                        M = cv2.moments(m255, binaryImage=True)
                        if M["m00"] <= 0:
                            continue
                        u = int(M["m10"] / M["m00"])
                        v = int(M["m01"] / M["m00"])

                        bgr = PALETTE_BGR[i % len(PALETTE_BGR)]
                        items.append((u, mask_bool, (u, v), bgr))

                    items.sort(key=lambda x: x[0])

                    for obj_i, (u0, mask_bool, c_uv, bgr) in enumerate(items[:MAX_DETECTIONS]):
                        # contours
                        m255 = (mask_bool.astype(np.uint8) * 255)
                        cnts, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        contours_out.append(cnts)
                        cent_uv_out.append(c_uv)
                        colors_out.append(bgr)

                        # 3D points + centroid (meters)
                        pts, cols, cxyz = make_object_cloud(mask_bool, depth_mm, self.fx, self.fy, self.cx, self.cy)
                        if pts is not None:
                            obj_pts_list.append(pts)
                            obj_cols_list.append(cols)

                            cent_xyz_list.append(cxyz[None, :])
                            # centroid color: use a fixed bright-ish depth color based on its depth
                            cz_mm = float(cxyz[2] * 1000.0)
                            cc = depth_mm_to_rgb01(np.array([cz_mm], dtype=np.float32))[0]
                            cent_col_list.append(cc[None, :])

                            cand.append((float(cxyz[2]), obj_i))  # z in meters

                got_any = (len(contours_out) > 0)

                # choose best object to pick
                if got_any and len(cand) > 0:
                    if PICK_RULE == "max_z":
                        _, pick_index = max(cand, key=lambda x: x[0])
                    else:
                        _, pick_index = min(cand, key=lambda x: x[0])

                    # compute major/minor axis ONLY for chosen object (keeps it light)
                    # We use the merged point list; but we need the chosen object's pts.
                    # easiest: rebuild chosen pts quickly from stored lists index
                    if 0 <= pick_index < len(obj_pts_list):
                        pts_pick = obj_pts_list[pick_index]
                        c_xy, vmaj, vmin, maj_len, min_len, yaw = pca_major_minor_xy(pts_pick)
                        pick_yaw_deg = float(np.degrees(yaw))
                        pick_major_m = maj_len
                        pick_minor_m = min_len

                infer_ms = (time.perf_counter() - t0) * 1000.0

                with self.shared.lock:
                    self.shared.last_infer_ms = infer_ms

                    if got_any:
                        self.miss_count = 0
                        self.shared.contours = contours_out
                        self.shared.centroids_uv = cent_uv_out
                        self.shared.colors_bgr = colors_out
                        self.shared.num_objs = len(contours_out)

                        if obj_pts_list:
                            self.shared.obj_pts = np.concatenate(obj_pts_list, axis=0)
                            self.shared.obj_cols = np.concatenate(obj_cols_list, axis=0)
                            self.shared.centroids_xyz = np.concatenate(cent_xyz_list, axis=0) if cent_xyz_list else None
                            self.shared.centroid_cols = np.concatenate(cent_col_list, axis=0) if cent_col_list else None
                        else:
                            self.shared.obj_pts = None
                            self.shared.obj_cols = None
                            self.shared.centroids_xyz = None
                            self.shared.centroid_cols = None

                        self.shared.pick_index = int(pick_index)
                        self.shared.pick_yaw_deg = float(pick_yaw_deg)
                        self.shared.pick_major_m = float(pick_major_m)
                        self.shared.pick_minor_m = float(pick_minor_m)
                    else:
                        self.miss_count += 1
                        if self.miss_count >= MISS_TOLERANCE:
                            self.shared.contours = []
                            self.shared.centroids_uv = []
                            self.shared.colors_bgr = []
                            self.shared.num_objs = 0
                            self.shared.obj_pts = None
                            self.shared.obj_cols = None
                            self.shared.centroids_xyz = None
                            self.shared.centroid_cols = None
                            self.shared.pick_index = -1

            except Exception:
                self.miss_count += 1
                if self.miss_count >= MISS_TOLERANCE:
                    with self.shared.lock:
                        self.shared.contours = []
                        self.shared.centroids_uv = []
                        self.shared.colors_bgr = []
                        self.shared.num_objs = 0
                        self.shared.obj_pts = None
                        self.shared.obj_cols = None
                        self.shared.centroids_xyz = None
                        self.shared.centroid_cols = None
                        self.shared.pick_index = -1

# =========================
# 2D overlay
# =========================
def overlay_2d(frame_bgr, contours_list, centroids_uv, colors_bgr, pick_index, pick_yaw_deg):
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    for i in range(len(contours_list)):
        bgr = colors_bgr[i]
        cv2.drawContours(vis, contours_list[i], -1, bgr, 2)

        u, v = centroids_uv[i]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, f"{i}", (u + 6, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # mark chosen object
        if i == pick_index:
            cv2.putText(vis, f"PICK (yaw {pick_yaw_deg:.1f} deg)", (u + 12, v + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return vis

# =========================
# Main
# =========================
def main():
    global SHOW_BG_CLOUD, O3D_POINT_SIZE

    cv2.setUseOptimized(True)

    print("--- SAM3 + OAK-D (DepthAI v3 sync + aligned depth) | multi-object + 3D centroid + pick + axes ---")

    device_torch = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM3 device: {device_torch}")

    if device_torch == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    fx, fy, cx, cy = read_rgb_intrinsics_for_res(RGB_RES[0], RGB_RES[1])
    print(f"RGB intrinsics @ {RGB_RES}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # load SAM3 (repo API)
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=None).to(device_torch)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESH)
    print("SAM3 loaded.")

    shared = Shared()
    text_prompt = input("\nObject to segment (try: 'pipe' / 'cylinder' / 'hex adaptor'): ").strip() or "pipe"
    with shared.lock:
        shared.prompt = text_prompt
        shared.prompt_ver += 1

    print("\nControls:")
    print("  q = quit")
    print("  t = change prompt")
    print("  b = toggle background cloud (3D)")
    print("  r = reset 3D view")
    print("  [ / ] = decrease/increase point size")

    viewer = None
    if ENABLE_POINTCLOUD and OPEN3D_OK:
        viewer = MultiObjectOpen3DViewer(max_centroids=MAX_DETECTIONS)
        print("[Open3D] started.")
    elif ENABLE_POINTCLOUD and not OPEN3D_OK:
        print("[Open3D] not installed -> pip install open3d (continuing without 3D)")

    worker = InferWorker(shared, processor, device_torch, fx, fy, cx, cy)
    worker.start()

    last_o3d_t = 0.0
    o3d_period = 1.0 / max(1e-6, CLOUD_UPDATE_HZ)

    # -------------------------
    # DepthAI v3: RGB + aligned depth using Sync + inputAlignTo/ImageAlign
    # This matches Luxonis v3 Depth Align example. :contentReference[oaicite:3]{index=3}
    # -------------------------
    pipeline = dai.Pipeline()
    platform = pipeline.getDefaultDevice().getPlatform()

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    sync   = pipeline.create(dai.node.Sync)

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
    else:
        align = None

    # tuning (simple)
    try:
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    except Exception:
        pass
    try:
        stereo.setLeftRightCheck(True)
    except Exception:
        pass

    sync.setSyncThreshold(timedelta(seconds=1/(2*FPS_RGB)))

    rgbOut   = camRgb.requestOutput(size=RGB_RES, fps=FPS_RGB, enableUndistortion=True)
    leftOut  = left.requestOutput(size=STEREO_RES, fps=FPS_DEPTH)
    rightOut = right.requestOutput(size=STEREO_RES, fps=FPS_DEPTH)

    rgbOut.link(sync.inputs["rgb"])
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    if platform == dai.Platform.RVC4:
        stereo.depth.link(align.input)
        rgbOut.link(align.inputAlignTo)
        align.outputAligned.link(sync.inputs["depth_aligned"])
    else:
        stereo.depth.link(sync.inputs["depth_aligned"])
        rgbOut.link(stereo.inputAlignTo)

    q = sync.out.createOutputQueue(maxSize=4, blocking=False)

    with pipeline:
        pipeline.start()

        fps_vis = 0.0
        t_prev = time.perf_counter()

        while pipeline.isRunning():
            msg = q.tryGet()
            if msg is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            frameRgb = msg["rgb"]
            frameDepth = msg["depth_aligned"]
            frame_bgr = frameRgb.getCvFrame()
            depth_mm = frameDepth.getFrame()

            # publish latest frame/depth to worker
            with shared.lock:
                shared.frame_bgr = frame_bgr
                shared.depth_mm = depth_mm
                shared.frame_id += 1

            # read results
            with shared.lock:
                contours = list(shared.contours)
                cent_uv  = list(shared.centroids_uv)
                cols_bgr = list(shared.colors_bgr)
                n_objs   = int(shared.num_objs)
                infer_ms = float(shared.last_infer_ms)

                obj_pts = shared.obj_pts
                obj_cols = shared.obj_cols
                cent_xyz = shared.centroids_xyz
                cent_cols = shared.centroid_cols

                pick_index = int(shared.pick_index)
                pick_yaw_deg = float(shared.pick_yaw_deg)
                pick_major_m = float(shared.pick_major_m)
                pick_minor_m = float(shared.pick_minor_m)

                text_prompt = shared.prompt

            vis2d = overlay_2d(frame_bgr, contours, cent_uv, cols_bgr, pick_index, pick_yaw_deg)

            # UI FPS
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_vis = 0.9 * fps_vis + 0.1 * (1.0 / dt)

            cv2.putText(vis2d, f"prompt: {text_prompt} | objs: {n_objs}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis2d, f"UI FPS: {fps_vis:.1f} | infer: {infer_ms:.0f} ms @ {TARGET_INFER_FPS:.0f}Hz",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # show chosen centroid xyz
            if cent_xyz is not None and 0 <= pick_index < cent_xyz.shape[0]:
                cxm, cym, czm = cent_xyz[pick_index]
                cv2.putText(vis2d, f"PICK xyz(m): {cxm:.3f}, {cym:.3f}, {czm:.3f}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis2d, f"major={pick_major_m:.3f}m minor={pick_minor_m:.3f}m yaw={pick_yaw_deg:.1f}deg",
                            (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("2D Preview", vis2d)

            # Open3D update (throttled)
            if viewer is not None:
                now = time.perf_counter()
                if now - last_o3d_t >= o3d_period:
                    last_o3d_t = now
                    if SHOW_BG_CLOUD:
                        scene_pts, scene_cols = make_scene_cloud(
                            depth_mm, fx, fy, cx, cy, stride=SCENE_STRIDE, max_points=MAX_SCENE_POINTS
                        )
                    else:
                        scene_pts, scene_cols = None, None

                    viewer.update(scene_pts, scene_cols, obj_pts, obj_cols, cent_xyz, cent_cols)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                new_prompt = input("\nNew prompt: ").strip()
                if new_prompt:
                    with shared.lock:
                        shared.prompt = new_prompt
                        shared.prompt_ver += 1
            if key == ord("b"):
                SHOW_BG_CLOUD = not SHOW_BG_CLOUD
                print(f"[3D] Background cloud: {SHOW_BG_CLOUD}")
            if key == ord("r") and viewer is not None:
                viewer.reset_view()
            if key == ord("[") and viewer is not None:
                O3D_POINT_SIZE = max(1.0, O3D_POINT_SIZE - 0.5)
                viewer.set_point_size(O3D_POINT_SIZE)
            if key == ord("]") and viewer is not None:
                O3D_POINT_SIZE = min(12.0, O3D_POINT_SIZE + 0.5)
                viewer.set_point_size(O3D_POINT_SIZE)

    worker.stop()
    worker.join(timeout=1.0)

    if viewer is not None:
        viewer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
