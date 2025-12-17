#!/usr/bin/env python3
import argparse, socket, struct, time, threading, zlib
import numpy as np

import cv2
import open3d as o3d

MAGIC_DPT = b"DPT0"
HDR_DPT = struct.Struct("<4sIHHQHHBH")  # magic seq idx cnt ts_ns w h comp n

MAGIC_CAL = b"CAL0"
HDR_CAL  = struct.Struct("<4sBQHH")     # magic ver ts_ns w h
CAL_FLOATS = struct.Struct("<30fB")     # 30 floats + units

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    HAS_ZSTD = False

try:
    import lz4.frame
    HAS_LZ4 = True
except Exception:
    HAS_LZ4 = False


def decompress(payload: bytes, comp: int, expected: int) -> bytes:
    if comp == COMP_NONE:
        return payload
    if comp == COMP_ZSTD and HAS_ZSTD:
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=expected)
    if comp == COMP_LZ4 and HAS_LZ4:
        return lz4.frame.decompress(payload)
    if comp == COMP_ZLIB:
        return zlib.decompress(payload)
    return payload


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.K = None
        self.depth = None
        self.lat_ms = 0.0
        self.seq = 0
        self.depth_frames = 0

        self.rgb = None
        self.rgb_frames = 0


class UdpReceiver(threading.Thread):
    def __init__(self, bind: str, port: int, stale_ms: int, shared: Shared):
        super().__init__(daemon=True)
        self.shared = shared
        self.stale_ms = stale_ms
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.25)
        self.pending = {}

    def run(self):
        while True:
            now = time.monotonic()
            for seq in list(self.pending.keys()):
                if (now - self.pending[seq]["t0"]) * 1000.0 > self.stale_ms:
                    self.pending.pop(seq, None)

            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue

            if len(data) < 4:
                continue

            magic = data[:4]

            if magic == MAGIC_CAL:
                if len(data) < HDR_CAL.size + CAL_FLOATS.size:
                    continue
                _, ver, ts_ns, w, h = HDR_CAL.unpack_from(data, 0)
                floats_units = CAL_FLOATS.unpack_from(data, HDR_CAL.size)
                floats = floats_units[:-1]
                K_rgb = np.array(floats[0:9], dtype=np.float32).reshape(3, 3)

                with self.shared.lock:
                    self.shared.K = K_rgb

                print(f"\n[CAL0] ver={ver} size={w}x{h}\nK_rgb=\n{K_rgb}\n", flush=True)
                continue

            if magic != MAGIC_DPT or len(data) < HDR_DPT.size:
                continue

            _, seq, idx, cnt, ts_ns, w, h, comp, n = HDR_DPT.unpack_from(data, 0)
            payload = data[HDR_DPT.size:HDR_DPT.size+n]
            if len(payload) != n or idx >= cnt:
                continue

            st = self.pending.get(seq)
            if st is None:
                st = {
                    "t0": time.monotonic(),
                    "w": w, "h": h, "comp": comp, "cnt": cnt,
                    "parts": [None] * cnt,
                    "ts": ts_ns,
                }
                self.pending[seq] = st

            st["parts"][idx] = payload

            if all(p is not None for p in st["parts"]):
                blob = b"".join(st["parts"])
                self.pending.pop(seq, None)

                raw = decompress(blob, st["comp"], expected=st["w"] * st["h"] * 2)
                if len(raw) < st["w"] * st["h"] * 2:
                    continue

                depth = np.frombuffer(raw[:st["w"] * st["h"] * 2], np.uint16).reshape((st["h"], st["w"]))
                lat_ms = (time.time_ns() - st["ts"]) / 1e6

                with self.shared.lock:
                    self.shared.depth = depth
                    self.shared.lat_ms = lat_ms
                    self.shared.seq = seq
                    self.shared.depth_frames += 1


class RtspReader(threading.Thread):
    def __init__(self, url: str, shared: Shared):
        super().__init__(daemon=True)
        self.url = url
        self.shared = shared

    def run(self):
        cap = cv2.VideoCapture(self.url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.shared.lock:
                self.shared.rgb = frame
                self.shared.rgb_frames += 1


def build_points(depth_u16: np.ndarray, K: np.ndarray, stride: int, min_mm: int, max_mm: int):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    d = depth_u16[::stride, ::stride].astype(np.float32)
    v, u = np.indices(d.shape, dtype=np.float32)
    u = u * stride
    v = v * stride

    valid = (d >= min_mm) & (d <= max_mm)
    z = d * 0.001  # meters

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, -y, z], axis=-1)     # (H,W,3)
    uv  = np.stack([u, v], axis=-1)         # (H,W,2)

    return pts[valid], uv[valid], d[valid]


def depth_colormap_rgb(dmm_valid: np.ndarray, min_mm: int, max_mm: int) -> np.ndarray:
    """Return (N,3) float32 RGB in [0,1]."""
    denom = max(1, (max_mm - min_mm))
    zn = np.clip((dmm_valid - min_mm) / denom, 0.0, 1.0)

    zn8 = (zn * 255.0).astype(np.uint8).reshape(-1, 1)     # (N,1)
    col = cv2.applyColorMap(zn8, cv2.COLORMAP_TURBO)       # (N,1,3) BGR
    col = col.reshape(-1, 3)                               # (N,3) BGR

    rgb = col[:, ::-1].astype(np.float32) / 255.0          # (N,3) RGB
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--stale-ms", type=int, default=150)

    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-mm", type=int, default=300)
    ap.add_argument("--max-mm", type=int, default=4000)

    ap.add_argument("--voxel", type=float, default=0.01)  # meters; 0 disables
    ap.add_argument("--sor-nb", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)

    ap.add_argument("--rtsp", default="", help="RTSP URL for RGB coloring (optional)")
    ap.add_argument("--fx", type=float, default=0.0, help="fallback if CAL0 missing")
    ap.add_argument("--fy", type=float, default=0.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)
    args = ap.parse_args()

    shared = Shared()
    UdpReceiver(args.bind, args.port, args.stale_ms, shared).start()
    if args.rtsp:
        RtspReader(args.rtsp, shared).start()

    if args.fx > 0 and args.fy > 0:
        K = np.array([[args.fx, 0, args.cx],
                      [0, args.fy, args.cy],
                      [0, 0, 1]], dtype=np.float32)
        with shared.lock:
            shared.K = K
        print(f"[K fallback] Using fx/fy/cx/cy = {args.fx},{args.fy},{args.cx},{args.cy}")

    vis = o3d.visualization.Visualizer()
    vis.create_window("UDP PointCloud (Open3D) - interactive", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(coord)
    vis.add_geometry(pcd)

    ropt = vis.get_render_option()
    ropt.point_size = 2.0
    ropt.background_color = np.asarray([0.05, 0.05, 0.05])

    last_seq = None
    print("Tip: start this viewer FIRST, then restart the edge sender so CAL0 arrives.", flush=True)

    while True:
        if not vis.poll_events():
            break

        with shared.lock:
            depth = None if shared.depth is None else shared.depth.copy()
            K = None if shared.K is None else shared.K.copy()
            lat = shared.lat_ms
            seq = shared.seq
            rgb = None if shared.rgb is None else shared.rgb.copy()
            df = shared.depth_frames
            rf = shared.rgb_frames

        if depth is None or K is None:
            vis.update_renderer()
            time.sleep(0.01)
            continue

        if seq == last_seq:
            vis.update_renderer()
            time.sleep(0.005)
            continue

        try:
            depth = cv2.medianBlur(depth, 3)
        except Exception:
            pass

        pts, uv, dmm = build_points(depth, K, args.stride, args.min_mm, args.max_mm)

        if pts.shape[0] == 0:
            print(f"\rseq={seq} depth_frames={df} rgb_frames={rf} udp_lat={lat:.1f}ms VALID_PTS=0 (tune min/max)", end="")
            last_seq = seq
            vis.update_renderer()
            continue

        # Colors: from RTSP if perfectly aligned/resolution-matched; else colormap
        if rgb is not None and rgb.shape[:2] == depth.shape:
            u = np.clip(uv[:, 0].astype(np.int32), 0, rgb.shape[1]-1)
            v = np.clip(uv[:, 1].astype(np.int32), 0, rgb.shape[0]-1)
            bgr = rgb[v, u, :]
            colors = (bgr[:, ::-1].astype(np.float32) / 255.0)   # (N,3)
        else:
            colors = depth_colormap_rgb(dmm, args.min_mm, args.max_mm)

        # Safety: shapes must match
        if colors.shape[0] != pts.shape[0] or colors.shape[1] != 3:
            print(f"\nBAD colors shape={colors.shape}, pts shape={pts.shape}, skipping frame", flush=True)
            last_seq = seq
            continue

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
        cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))

        if args.voxel and args.voxel > 0:
            cloud = cloud.voxel_down_sample(args.voxel)

        if len(cloud.points) > 0 and args.sor_nb > 0:
            cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=args.sor_nb, std_ratio=args.sor_std)

        pcd.points = cloud.points
        pcd.colors = cloud.colors
        vis.update_geometry(pcd)

        print(f"\rseq={seq} depth_frames={df} rgb_frames={rf} udp_lat={lat:5.1f}ms pts={len(pcd.points):7d}", end="")
        last_seq = seq
        vis.update_renderer()
        time.sleep(0.002)

    vis.destroy_window()
    print()


if __name__ == "__main__":
    main()
