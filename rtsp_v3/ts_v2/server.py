#!/usr/bin/env python3
import argparse, socket, struct, time, zlib, threading
from collections import OrderedDict
import numpy as np
import cv2

DEFAULT = dict(
    BIND="0.0.0.0",
    PORT=9000,
    RCVBUF=8_388_608,
    STALE_MS=200,

    RTSP_URL="rtsp://192.168.1.182:8554/oak",
    RTSP_TRANSPORT="udp",
    RGB_FPS=30,

    SYNC_MS=25,      # tighter: 1 frame at 30fps is 33ms
    HOLD_MS=80,      # hold unmatched depth briefly for RGB to arrive
    MAX_BUF=200,

    SHOW=True,
    MIN_MM=120,
    MAX_MM=2500,
)

MAGIC_DPT = b"DPT0"
HDR_DPT = struct.Struct("<4sIHHQHHBH")

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4  = 2
COMP_ZLIB = 3

MAGIC_CAL = b"CAL0"
HDR_CAL = struct.Struct("<4sBQHH")
CAL_FLOATS = struct.Struct("<30fB")
UNITS_CM = 1
UNITS_M  = 2

MAGIC_VID = b"VID0"
HDR_VID = struct.Struct("<4sBIQHH")  # magic, ver, frame_id, ts_ns, w, h

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

def find_nearest_key(od: OrderedDict, ts: int):
    if not od:
        return None
    keys = list(od.keys())
    import bisect
    i = bisect.bisect_left(keys, ts)
    cands = []
    if i < len(keys): cands.append(keys[i])
    if i > 0: cands.append(keys[i-1])
    return min(cands, key=lambda k: abs(int(k) - int(ts)))

def rtsp_decode_thread(rtsp_url, rtsp_transport, rgb_fps,
                       meta_by_id: OrderedDict,
                       rgb_buf: OrderedDict,
                       lock: threading.Lock,
                       stop_evt: threading.Event,
                       max_buf: int):
    try:
        import av
    except Exception:
        print("[server][ERR] PyAV not installed: pip install av")
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
        print("[server][ERR] RTSP open failed:", e)
        stop_evt.set()
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # Map RTSP frame index (derived from PTS) -> sender frame_id
    id_offset = None
    last_n = -1

    while not stop_evt.is_set():
        try:
            for frame in container.decode(stream):
                if stop_evt.is_set():
                    break

                img = frame.to_ndarray(format="bgr24")

                # Compute RTSP frame index n from PTS (setts makes PTS ~= n/fps)
                if frame.pts is not None and stream.time_base is not None:
                    sec = float(frame.pts * stream.time_base)
                    n = int(round(sec * float(rgb_fps)))
                else:
                    n = last_n + 1
                last_n = n

                # Estimate sender frame_id using offset calibration
                with lock:
                    if not meta_by_id:
                        # no meta yet; store under "now" so UI isn't empty
                        rgb_buf[int(time.monotonic_ns())] = (img, time.monotonic_ns())
                        while len(rgb_buf) > max_buf:
                            rgb_buf.popitem(last=False)
                        continue

                    # Initialize / refine id_offset by nearest meta id
                    if id_offset is None:
                        # pick the closest available meta id to n
                        meta_ids = list(meta_by_id.keys())
                        best_mid = min(meta_ids, key=lambda mid: abs(int(mid) - int(n)))
                        id_offset = int(best_mid) - int(n)

                    want_id = int(n + id_offset)

                    # Exact match preferred; else nearest within +-2 frames
                    if want_id in meta_by_id:
                        dev_ts = meta_by_id.pop(want_id)
                    else:
                        # find nearest id
                        meta_ids = list(meta_by_id.keys())
                        best_mid = min(meta_ids, key=lambda mid: abs(int(mid) - int(want_id)))
                        if abs(int(best_mid) - int(want_id)) <= 2:
                            dev_ts = meta_by_id.pop(best_mid)
                            # adjust offset slowly to recover after drops
                            id_offset = int(best_mid) - int(n)
                        else:
                            # too far: likely burst loss; skip storing this frame with bad meta
                            continue

                    # Store RGB by *device timestamp* (key) for pairing with depth
                    rgb_buf[int(dev_ts)] = (img, time.monotonic_ns())
                    while len(rgb_buf) > max_buf:
                        rgb_buf.popitem(last=False)

                    # Prune meta_by_id to avoid growth
                    while len(meta_by_id) > max_buf * 3:
                        meta_by_id.popitem(last=False)

        except Exception:
            time.sleep(0.02)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default=DEFAULT["BIND"])
    ap.add_argument("--port", type=int, default=DEFAULT["PORT"])
    ap.add_argument("--rcvbuf", type=int, default=DEFAULT["RCVBUF"])
    ap.add_argument("--stale-ms", type=int, default=DEFAULT["STALE_MS"])

    ap.add_argument("--rtsp-url", default=DEFAULT["RTSP_URL"])
    ap.add_argument("--rtsp-transport", choices=["udp","tcp"], default=DEFAULT["RTSP_TRANSPORT"])
    ap.add_argument("--rgb-fps", type=int, default=DEFAULT["RGB_FPS"])

    ap.add_argument("--sync-ms", type=int, default=DEFAULT["SYNC_MS"])
    ap.add_argument("--hold-ms", type=int, default=DEFAULT["HOLD_MS"])
    ap.add_argument("--max-buf", type=int, default=DEFAULT["MAX_BUF"])

    ap.add_argument("--show", action="store_true", default=DEFAULT["SHOW"])
    ap.add_argument("--min-mm", type=int, default=DEFAULT["MIN_MM"])
    ap.add_argument("--max-mm", type=int, default=DEFAULT["MAX_MM"])
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)

    zstd_d, lz4mod = make_decompressor()

    pending = {}  # depth seq -> chunk state

    # Buffers keyed by DEVICE timestamp (ns)
    meta_by_id = OrderedDict()      # frame_id -> device_ts_ns
    rgb_buf = OrderedDict()         # device_ts_ns -> (bgr_img, recv_mono_ns)
    depth_buf = OrderedDict()       # device_ts_ns -> (depth_u16, recv_mono_ns)

    lock = threading.Lock()
    stop_evt = threading.Event()

    t = threading.Thread(
        target=rtsp_decode_thread,
        args=(args.rtsp_url, args.rtsp_transport, args.rgb_fps,
              meta_by_id, rgb_buf, lock, stop_evt, args.max_buf),
        daemon=True
    )
    t.start()

    last_pair = None
    last_print = time.monotonic()
    ok_depth = 0
    ok_pairs = 0

    def prune_by_recv_time(buf: OrderedDict, max_age_ms: int):
        now_ns = time.monotonic_ns()
        # buf values are (payload, recv_mono_ns)
        while buf:
            k0, v0 = next(iter(buf.items()))
            recv_ns = v0[1]
            if (now_ns - recv_ns) / 1e6 > max_age_ms:
                buf.popitem(last=False)
            else:
                break

    def try_pair_latest():
        nonlocal last_pair, ok_pairs
        with lock:
            if not depth_buf or not rgb_buf:
                return
            # take newest depth
            dts, (depth, drecv) = next(reversed(depth_buf.items()))
            # wait briefly if depth is very fresh and rgb may still arrive
            age_ms = (time.monotonic_ns() - drecv) / 1e6
            if age_ms < 2:  # tiny wait window, keeps latency low
                return

            rts = find_nearest_key(rgb_buf, dts)
            if rts is None:
                return
            rgb, _rrecv = rgb_buf[rts]
            dt_ms = abs(int(rts) - int(dts)) / 1e6

            if dt_ms <= args.sync_ms:
                ok_pairs += 1
                last_pair = (rgb, depth, dt_ms)
                # remove paired depth (prevents “behind” feeling)
                depth_buf.pop(dts, None)

    print(f"[server] UDP {args.bind}:{args.port}  RTSP={args.rtsp_url} ({args.rtsp_transport})  sync_ms={args.sync_ms} hold_ms={args.hold_ms}")

    while True:
        # prune old frames by receive-time (host monotonic)
        with lock:
            prune_by_recv_time(rgb_buf, max_age_ms=args.hold_ms)
            prune_by_recv_time(depth_buf, max_age_ms=args.hold_ms)

        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            try_pair_latest()
            if args.show and last_pair is not None:
                rgb, depth, dt_ms = last_pair
                cv2.imshow("RGB (paired)", rgb)

                d = depth.astype(np.float32)
                valid = (d > 0) & (d >= args.min_mm) & (d <= args.max_mm)
                vis8 = np.zeros(depth.shape, np.uint8)
                if valid.any():
                    lo = np.percentile(d[valid], 3)
                    hi = np.percentile(d[valid], 97)
                    vis8 = np.clip((d - lo) * 255.0 / max(1.0, (hi - lo)), 0, 255).astype(np.uint8)

                dc = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
                dc[~valid] = 0
                cv2.putText(dc, f"paired dt={dt_ms:.1f}ms  depth_ok={ok_depth} pairs={ok_pairs}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Depth (paired)", dc)

                if cv2.waitKey(1) == ord("q"):
                    break
            continue

        if len(data) < 4:
            continue
        magic = data[:4]

        # CAL0 ignored here (you can keep your calibration parsing if needed)

        # VID0: store meta by frame_id
        if magic == MAGIC_VID and len(data) >= HDR_VID.size:
            _, ver, frame_id, ts_ns, w, h = HDR_VID.unpack_from(data, 0)
            with lock:
                meta_by_id[int(frame_id)] = int(ts_ns)
                while len(meta_by_id) > args.max_buf * 3:
                    meta_by_id.popitem(last=False)
            continue

        # DPT0: reassemble + store by device timestamp
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
            # stale cleanup
            if (time.monotonic() - st["t0"]) * 1000.0 > args.stale_ms:
                pending.pop(seq, None)
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
            depth_buf[int(st["ts_ns"])] = (depth, time.monotonic_ns())
            while len(depth_buf) > args.max_buf:
                depth_buf.popitem(last=False)

        try_pair_latest()

        if time.monotonic() - last_print > 1.0:
            with lock:
                print(f"[server] depth_ok={ok_depth} pairs={ok_pairs} rgb_buf={len(rgb_buf)} depth_buf={len(depth_buf)} meta_by_id={len(meta_by_id)}")
            last_print = time.monotonic()

    stop_evt.set()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
