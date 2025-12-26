#!/usr/bin/env python3
import argparse, socket, struct, time, zlib, threading
from collections import deque, OrderedDict
import numpy as np
import cv2

# ============================================================
# QUICK DEBUG CONFIG (edit these, then run with no CLI args)
# ============================================================
DEFAULT = dict(
    BIND="0.0.0.0",
    PORT=9000,
    RCVBUF=8_388_608,
    STALE_MS=200,

    RTSP_URL="rtsp://192.168.1.182:8554/oak",
    RTSP_TRANSPORT="udp",

    SYNC_MS=60,          # max allowed |rgb_ts - depth_ts|
    MAX_BUF=200,         # buffer frames (timestamps increase)

    SHOW=True,
    O3D=True,
    O3D_EVERY=1,

    MIN_MM=120,
    MAX_MM=2500,
    PCD_STRIDE=2,
    VOXEL=0.005,
    REMOVE_PLANE=True,
)

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

# ---- VID0 (RGB metadata over UDP) ----
MAGIC_VID = b"VID0"
HDR_VID = struct.Struct("<4sBIQHH")     # magic, ver, frame_id, ts_ns, w, h

def add_bool_arg(ap: argparse.ArgumentParser, name: str, default: bool, help_txt: str):
    dest = name.replace("-", "_")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help_txt + f" (default={default})")
    g.add_argument(f"--no-{name}", dest=dest, action="store_false", help="Disable: " + help_txt)
    ap.set_defaults(**{dest: default})

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

def find_nearest_ts(buf: OrderedDict, ts: int):
    if not buf:
        return None, None
    keys = list(buf.keys())
    import bisect
    i = bisect.bisect_left(keys, ts)
    candidates = []
    if i < len(keys): candidates.append(keys[i])
    if i > 0: candidates.append(keys[i-1])
    best = min(candidates, key=lambda k: abs(int(k) - int(ts)))
    return best, buf[best]

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

def transform_pts(pts: np.ndarray, T_3x4: np.ndarray, units: int):
    R = T_3x4[:, :3].astype(np.float32)
    t = T_3x4[:, 3].astype(np.float32)
    if units == UNITS_CM:
        t = t * 0.01
    elif units == UNITS_M:
        t = t
    else:
        # unknown, assume cm like DepthAI boards
        t = t * 0.01
    return (pts @ R.T) + t

def rtsp_decode_thread(rtsp_url: str, rtsp_transport: str,
                       meta_q: deque, rgb_buf: OrderedDict,
                       lock: threading.Lock, stop_evt: threading.Event,
                       max_buf: int):
    try:
        import av
    except Exception:
        print("[server][ERR] PyAV not installed. Install with: pip install av")
        stop_evt.set()
        return

    av_opts = {
        "rtsp_transport": rtsp_transport,
        "fflags": "nobuffer",
        "flags": "low_delay",
        "probesize": "32",
        "analyzeduration": "0",
        "max_delay": "0",
    }

    try:
        container = av.open(rtsp_url, options=av_opts)
    except Exception as e:
        print(f"[server][ERR] RTSP open failed for {rtsp_url}: {e}")
        stop_evt.set()
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    last_rgb = None
    while not stop_evt.is_set():
        try:
            for frame in container.decode(stream):
                if stop_evt.is_set():
                    break

                img = frame.to_ndarray(format="bgr24")
                last_rgb = img

                # Assign timestamp using next VID0 meta (same order; B-frames disabled on sender)
                meta = None
                t0 = time.monotonic()
                while (time.monotonic() - t0) < 0.25:
                    with lock:
                        if meta_q:
                            meta = meta_q.popleft()
                            break
                    time.sleep(0.001)

                if meta is None:
                    # still store "latest" under a fake key so UI shows something
                    with lock:
                        rgb_buf[time.time_ns()] = img
                        while len(rgb_buf) > max_buf:
                            rgb_buf.popitem(last=False)
                    continue

                (frame_id, ts_ns, w, h) = meta
                with lock:
                    rgb_buf[ts_ns] = img
                    while len(rgb_buf) > max_buf:
                        rgb_buf.popitem(last=False)

        except Exception:
            time.sleep(0.05)

def main():
    ap = argparse.ArgumentParser(description="Server: receive UDP depth+VID0, decode RTSP RGB, pair by device timestamps.")

    ap.add_argument("--bind", default=DEFAULT["BIND"])
    ap.add_argument("--port", type=int, default=DEFAULT["PORT"])
    ap.add_argument("--rcvbuf", type=int, default=DEFAULT["RCVBUF"])
    ap.add_argument("--stale-ms", type=int, default=DEFAULT["STALE_MS"])

    ap.add_argument("--rtsp-url", default=DEFAULT["RTSP_URL"])
    ap.add_argument("--rtsp-transport", choices=["udp","tcp"], default=DEFAULT["RTSP_TRANSPORT"])

    ap.add_argument("--sync-ms", type=int, default=DEFAULT["SYNC_MS"])
    ap.add_argument("--max-buf", type=int, default=DEFAULT["MAX_BUF"])

    add_bool_arg(ap, "show", DEFAULT["SHOW"], "Show OpenCV windows")
    add_bool_arg(ap, "o3d", DEFAULT["O3D"], "Show Open3D point cloud")
    ap.add_argument("--o3d-every", type=int, default=DEFAULT["O3D_EVERY"])

    ap.add_argument("--min-mm", type=int, default=DEFAULT["MIN_MM"])
    ap.add_argument("--max-mm", type=int, default=DEFAULT["MAX_MM"])
    ap.add_argument("--pcd-stride", type=int, default=DEFAULT["PCD_STRIDE"])
    ap.add_argument("--voxel", type=float, default=DEFAULT["VOXEL"])
    add_bool_arg(ap, "remove-plane", DEFAULT["REMOVE_PLANE"], "RANSAC remove dominant plane")

    args = ap.parse_args()

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    zstd_d, lz4mod = make_decompressor()

    pending = {}  # depth seq -> chunk state

    # calibration
    K_rgb = None
    K_right = None
    T_right_to_rgb = None
    units = UNITS_CM

    # VID0 queue and buffers
    meta_q = deque()
    rgb_buf = OrderedDict()
    depth_buf = OrderedDict()
    paired_depth_ts = set()

    lock = threading.Lock()
    stop_evt = threading.Event()

    # Open3D setup
    HAS_O3D = False
    if args.o3d:
        try:
            import open3d as o3d
            HAS_O3D = True
            vis = o3d.visualization.Visualizer()
            vis.create_window("OAK-D Synced PointCloud", 1280, 720)
            pcd = o3d.geometry.PointCloud()
            added = False
        except Exception as e:
            print("[server][WARN] Open3D not available:", e)
            args.o3d = False

    # Start RTSP decode thread
    t = threading.Thread(
        target=rtsp_decode_thread,
        args=(args.rtsp_url, args.rtsp_transport, meta_q, rgb_buf, lock, stop_evt, args.max_buf),
        daemon=True
    )
    t.start()

    last_print = time.monotonic()
    last_rgb = None
    last_depth = None
    ok_depth = 0
    ok_pairs = 0

    print(f"[server] UDP {args.bind}:{args.port}  RTSP={args.rtsp_url} ({args.rtsp_transport})  sync_ms={args.sync_ms}")

    def try_pair_some():
        nonlocal ok_pairs, last_rgb, last_depth
        with lock:
            # iterate a snapshot to avoid mutation issues
            for dts, depth in list(depth_buf.items()):
                if dts in paired_depth_ts:
                    continue
                rts, rgb = find_nearest_ts(rgb_buf, dts)
                if rgb is None:
                    continue
                dt_ms = abs(int(rts) - int(dts)) / 1e6
                if dt_ms <= args.sync_ms:
                    paired_depth_ts.add(dts)
                    ok_pairs += 1
                    last_rgb = rgb
                    last_depth = (depth, dt_ms)

    while True:
        # cleanup stale partial depth frames
        now = time.monotonic()
        for seq in list(pending.keys()):
            if (now - pending[seq]["t0"]) * 1000.0 > args.stale_ms:
                pending.pop(seq, None)

        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            # still try pairing when only RGB is flowing
            try_pair_some()
            if args.show and last_rgb is not None:
                cv2.imshow("RGB (latest)", last_rgb)
                if last_depth is not None:
                    depth, dt_ms = last_depth
                    d = depth.astype(np.float32)
                    valid = (d > 0) & (d >= args.min_mm) & (d <= args.max_mm)
                    vis8 = np.zeros(depth.shape, np.uint8)
                    if valid.any():
                        lo = np.percentile(d[valid], 3)
                        hi = np.percentile(d[valid], 97)
                        vis8 = np.clip((d - lo) * 255.0 / max(1.0, (hi - lo)), 0, 255).astype(np.uint8)
                    dc = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
                    cv2.putText(dc, f"Depth latest  pair_dt={dt_ms:.1f}ms", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.imshow("Depth (latest)", dc)
                if cv2.waitKey(1) == ord("q"):
                    break
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
            units = int(floats_units[-1])
            K_rgb = np.array(floats[0:9], dtype=np.float32).reshape(3,3)
            K_right = np.array(floats[9:18], dtype=np.float32).reshape(3,3)
            T_right_to_rgb = np.array(floats[18:30], dtype=np.float32).reshape(3,4)
            print(f"[server] CAL0 ver={ver} size={w}x{h} units={units} ts={ts_ns}")
            continue

        # ---- VID0 ----
        if magic == MAGIC_VID:
            if len(data) < HDR_VID.size:
                continue
            _, ver, frame_id, ts_ns, w, h = HDR_VID.unpack_from(data, 0)
            with lock:
                meta_q.append((int(frame_id), int(ts_ns), int(w), int(h)))
                while len(meta_q) > args.max_buf * 3:
                    meta_q.popleft()
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
                "ts_ns": int(ts_ns),
            }
            pending[seq] = st

        if st["parts"][idx] is None:
            st["parts"][idx] = payload
            st["got"] += 1

        if st["got"] != st["count"]:
            continue

        blob = b"".join(st["parts"])
        pending.pop(seq, None)

        expected = st["w"] * st["h"] * 2
        try:
            raw = decompress(blob, st["comp"], expected, zstd_d, lz4mod)
        except Exception:
            continue
        if len(raw) < expected:
            continue

        depth = np.frombuffer(raw[:expected], np.uint16).reshape((st["h"], st["w"]))
        ok_depth += 1

        with lock:
            depth_buf[st["ts_ns"]] = depth
            while len(depth_buf) > args.max_buf:
                depth_buf.popitem(last=False)

        # pairing attempt
        try_pair_some()

        # if we have a new pair, update visuals + pointcloud
        if last_rgb is not None and last_depth is not None:
            depth_latest, dt_ms = last_depth

            if args.show:
                # depth colormap
                d = depth_latest.astype(np.float32)
                valid = (d > 0) & (d >= args.min_mm) & (d <= args.max_mm)
                vis8 = np.zeros(depth_latest.shape, np.uint8)
                if valid.any():
                    lo = np.percentile(d[valid], 3)
                    hi = np.percentile(d[valid], 97)
                    vis8 = np.clip((d - lo) * 255.0 / max(1.0, (hi - lo)), 0, 255).astype(np.uint8)

                dc = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
                dc[~valid] = 0
                cv2.putText(dc, f"paired dt={dt_ms:.1f}ms  depth_ok={ok_depth} pairs={ok_pairs}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.imshow("RGB (paired)", last_rgb)
                cv2.imshow("Depth (paired)", dc)
                if cv2.waitKey(1) == ord("q"):
                    break

            # Open3D point cloud (in RGB frame, using K_right + T_right_to_rgb)
            if args.o3d and HAS_O3D and (ok_pairs % max(1, args.o3d_every) == 0) and (K_right is not None) and (T_right_to_rgb is not None):
                try:
                    import open3d as o3d
                    pts = depth_to_points(depth_latest, K_right,
                                          stride=max(1, args.pcd_stride),
                                          min_mm=args.min_mm, max_mm=args.max_mm)
                    if pts.shape[0] > 0:
                        pts_rgb = transform_pts(pts, T_right_to_rgb, units)

                        p = o3d.geometry.PointCloud()
                        p.points = o3d.utility.Vector3dVector(pts_rgb)

                        if args.voxel and args.voxel > 0:
                            p = p.voxel_down_sample(args.voxel)

                        if args.remove_plane and len(p.points) > 2000:
                            _plane_model, inliers = p.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=200)
                            p = p.select_by_index(inliers, invert=True)

                        if len(p.points) > 0:
                            pcd.points = p.points
                            if not added:
                                vis.add_geometry(pcd)
                                added = True
                            else:
                                vis.update_geometry(pcd)
                            vis.poll_events()
                            vis.update_renderer()
                except Exception:
                    pass

        if time.monotonic() - last_print > 1.0:
            with lock:
                rb = len(rgb_buf)
                db = len(depth_buf)
                mq = len(meta_q)
            print(f"[server] depth_ok={ok_depth} pairs={ok_pairs} rgb_buf={rb} depth_buf={db} meta_q={mq}")
            last_print = time.monotonic()

    stop_evt.set()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
