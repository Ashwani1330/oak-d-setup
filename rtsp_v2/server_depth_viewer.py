#!/usr/bin/env python3
import argparse, socket, struct, time, zlib
import numpy as np
import cv2

# ---- DPT0 ----
MAGIC_DPT = b"DPT0"
HDR_DPT = struct.Struct("<4sIHHQHHBH")  # magic, seq, idx, cnt, ts_ns, w, h, comp, n

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

# ---- CAL0 ----
MAGIC_CAL = b"CAL0"
HDR_CAL = struct.Struct("<4sBQHH")      # magic, ver(u8), ts_ns(u64), w(u16), h(u16)
CAL_FLOATS = struct.Struct("<30fB")     # 30 floats + units(u8)
UNITS_CM = 1
UNITS_M  = 2

def make_decompressor():
    zstd_d = None
    lz4mod = None
    try:
        import zstandard as zstd
        zstd_d = zstd.ZstdDecompressor()
    except Exception:
        pass
    try:
        import lz4.frame
        lz4mod = lz4.frame
    except Exception:
        pass
    return zstd_d, lz4mod

def decompress(blob: bytes, comp: int, expected: int, zstd_d, lz4mod) -> bytes:
    if comp == COMP_NONE:
        return blob
    if comp == COMP_ZSTD and zstd_d is not None:
        return zstd_d.decompress(blob, max_output_size=expected)
    if comp == COMP_LZ4 and lz4mod is not None:
        return lz4mod.decompress(blob)
    if comp == COMP_ZLIB:
        return zlib.decompress(blob)
    raise RuntimeError(f"Unsupported compression {comp} (missing lib?)")

def depth_to_points(depth_u16: np.ndarray, K: np.ndarray, stride: int, min_mm: int, max_mm: int):
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    d = depth_u16[::stride, ::stride].astype(np.float32)
    v, u = np.indices(d.shape, dtype=np.float32)
    u *= stride
    v *= stride

    z_mm = d
    valid = (z_mm > 0) & (z_mm >= min_mm) & (z_mm <= max_mm)
    if not np.any(valid):
        return np.empty((0,3), np.float32)

    z = z_mm * 0.001  # m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1)
    return pts[valid].astype(np.float32)

def parse_roi(s: str):
    # "x0,y0,x1,y1"
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1")
    return tuple(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--rcvbuf", type=int, default=8_388_608)
    ap.add_argument("--stale-ms", type=int, default=120)

    ap.add_argument("--ui", action="store_true", help="OpenCV depth window")
    ap.add_argument("--o3d", action="store_true", help="Open3D point cloud window")
    ap.add_argument("--o3d-every", type=int, default=1, help="update Open3D every N frames")

    ap.add_argument("--min-mm", type=int, default=300)
    ap.add_argument("--max-mm", type=int, default=2000)
    ap.add_argument("--pcd-stride", type=int, default=6)

    ap.add_argument("--voxel", type=float, default=0.01, help="meters (0 disables)")
    ap.add_argument("--remove-plane", action="store_true", help="RANSAC remove dominant plane (wall/floor)")
    ap.add_argument("--roi", type=str, default="", help="crop ROI: x0,y0,x1,y1")

    args = ap.parse_args()

    roi = None
    if args.roi:
        roi = parse_roi(args.roi)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    zstd_d, lz4mod = make_decompressor()

    # pending frames: seq -> state
    pending = {}

    # calibration state
    K_rgb = None
    K_right = None
    T_right_to_rgb = None
    units = None

    # Open3D setup (optional)
    HAS_O3D = False
    if args.o3d:
        try:
            import open3d as o3d
            HAS_O3D = True
            vis = o3d.visualization.Visualizer()
            vis.create_window("OAK-D Depth PointCloud", 1280, 720)
            pcd = o3d.geometry.PointCloud()
            added = False
        except Exception:
            print("[WARN] Open3D not available. pip install open3d")
            args.o3d = False

    ok = 0
    last_latency_ms = 0.0
    last_print = time.monotonic()

    print(f"Listening UDP on {args.bind}:{args.port} (CAL0 + DPT0)")

    while True:
        # cleanup stale depth frames
        now = time.monotonic()
        for seq in list(pending.keys()):
            if (now - pending[seq]["t0"]) * 1000.0 > args.stale_ms:
                pending.pop(seq, None)

        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue

        if len(data) < 4:
            continue

        magic = data[:4]

        # ---- CAL0 ----
        if magic == MAGIC_CAL:
            if len(data) < HDR_CAL.size + CAL_FLOATS.size:
                continue
            _, ver, ts_ns, w, h = HDR_CAL.unpack_from(data, 0)
            floats_units = CAL_FLOATS.unpack_from(data, HDR_CAL.size)
            floats = floats_units[:-1]
            units = floats_units[-1]

            K_rgb = np.array(floats[0:9], dtype=np.float32).reshape(3,3)
            K_right = np.array(floats[9:18], dtype=np.float32).reshape(3,3)
            T_right_to_rgb = np.array(floats[18:30], dtype=np.float32).reshape(3,4)

            print("Received CAL0:")
            print(f"  ver={ver} size={w}x{h} units={'cm' if units==UNITS_CM else 'm' if units==UNITS_M else units}")
            continue

        # ---- DPT0 ----
        if magic != MAGIC_DPT or len(data) < HDR_DPT.size:
            continue

        _, seq, idx, count, ts_ns, w, h, comp, n = HDR_DPT.unpack_from(data, 0)
        payload = data[HDR_DPT.size:HDR_DPT.size+n]
        if len(payload) != n or idx >= count:
            continue

        st = pending.get(seq)
        if st is None:
            st = {
                "t0": time.monotonic(),
                "w": w, "h": h, "comp": comp,
                "count": count,
                "got": 0,
                "parts": [None] * count,
                "ts_ns": ts_ns,
            }
            pending[seq] = st

        if st["parts"][idx] is None:
            st["parts"][idx] = payload
            st["got"] += 1

        if st["got"] == st["count"]:
            blob = b"".join(st["parts"])
            pending.pop(seq, None)

            expected = st["w"] * st["h"] * 2
            try:
                raw = decompress(blob, st["comp"], expected=expected, zstd_d=zstd_d, lz4mod=lz4mod)
            except Exception:
                continue
            if len(raw) < expected:
                continue

            depth = np.frombuffer(raw[:expected], np.uint16).reshape((st["h"], st["w"]))
            ok += 1
            last_latency_ms = (time.time_ns() - st["ts_ns"]) / 1e6

            # ROI crop (optional)
            if roi is not None:
                x0,y0,x1,y1 = roi
                x0 = max(0, min(depth.shape[1]-1, x0))
                x1 = max(0, min(depth.shape[1],   x1))
                y0 = max(0, min(depth.shape[0]-1, y0))
                y1 = max(0, min(depth.shape[0],   y1))
                if x1 > x0 and y1 > y0:
                    depth_roi = depth[y0:y1, x0:x1]
                else:
                    depth_roi = depth
            else:
                depth_roi = depth

            # ---- OpenCV depth visualization ----
            if args.ui:
                d = depth_roi
                valid = (d > 0) & (d >= args.min_mm) & (d <= args.max_mm)
                vis8 = np.zeros(d.shape, np.uint8)
                if valid.any():
                    lo = np.percentile(d[valid], 3)
                    hi = np.percentile(d[valid], 97)
                    vis8 = np.clip((d.astype(np.float32) - lo) * 255.0 / max(1.0, (hi - lo)), 0, 255).astype(np.uint8)

                dc = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
                dc[~valid] = 0
                cv2.putText(dc, f"depth {st['w']}x{st['h']}  {last_latency_ms:.1f}ms  ok {ok}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Depth UDP", dc)
                if cv2.waitKey(1) == ord("q"):
                    break

            # ---- Open3D point cloud ----
            if args.o3d and HAS_O3D and (ok % max(1, args.o3d_every) == 0):
                if K_rgb is not None:
                    pts = depth_to_points(depth_roi, K_rgb, stride=max(1,args.pcd_stride),
                                          min_mm=args.min_mm, max_mm=args.max_mm)
                    if pts.shape[0] > 0:
                        import open3d as o3d
                        p = o3d.geometry.PointCloud()
                        p.points = o3d.utility.Vector3dVector(pts)

                        if args.voxel and args.voxel > 0:
                            p = p.voxel_down_sample(args.voxel)

                        if args.remove_plane and len(p.points) > 2000:
                            plane_model, inliers = p.segment_plane(
                                distance_threshold=0.01, ransac_n=3, num_iterations=200
                            )
                            p = p.select_by_index(inliers, invert=True)

                        pcd.points = p.points
                        if not added:
                            vis.add_geometry(pcd)
                            added = True
                        else:
                            vis.update_geometry(pcd)
                        vis.poll_events()
                        vis.update_renderer()

            if time.monotonic() - last_print > 1.0:
                print(f"[ok={ok}] latency={last_latency_ms:.1f}ms pending={len(pending)}")
                last_print = time.monotonic()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
