#!/usr/bin/env python3
import asyncio
import signal
import threading
import time
import struct

import depthai as dai
import websockets

WIDTH, HEIGHT = 640, 400
FPS = 30

# WS_PUB_URL = "ws://192.168.1.182:8765/pub"  # server ingest endpoint
WS_PUB_URL = "ws://192.168.1.100:8765/pub"  # server ingest endpoint

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())


def oak_reader_thread(loop: asyncio.AbstractEventLoop, latest_q: "asyncio.Queue[bytes]"):
    # Uses YOUR known-good style: Pipeline.start() + OutputQueue from node output
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        cam_out = cam.requestOutput(
            (WIDTH, HEIGHT),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            FPS
        )

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_out,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )

        # Best-effort low-latency tuning (depends on DepthAI version)
        try: enc.setNumBFrames(0)
        except Exception: pass
        try: enc.setKeyframeFrequency(10)
        except Exception: pass
        try: enc.setBitrateKbps(1500)
        except Exception: pass

        q = enc.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        try:
            while pipeline.isRunning() and not quit_event.is_set():
                pkt = q.tryGet()
                if pkt is None:
                    time.sleep(0.001)
                    continue

                # Drain queue -> keep newest packet only (min latency)
                while True:
                    newer = q.tryGet()
                    if newer is None:
                        break
                    pkt = newer

                data = pkt.getData().tobytes()

                # Push newest frame into asyncio queue (replace older)
                def put_latest():
                    try:
                        while True:
                            latest_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        latest_q.put_nowait(data)
                    except Exception:
                        pass

                loop.call_soon_threadsafe(put_latest)

        finally:
            try:
                pipeline.stop()
                pipeline.wait()
            except Exception:
                pass


async def ws_sender(latest_q: "asyncio.Queue[bytes]"):
    while not quit_event.is_set():
        try:
            async with websockets.connect(
                WS_PUB_URL,
                max_size=None,
                open_timeout=5,     # fail fast if server unreachable
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                # Optional header to help viewers
                await ws.send(f"H264 {WIDTH}x{HEIGHT} {FPS}\n".encode("ascii"))

                while not quit_event.is_set():
                    data = await latest_q.get()
                    # Length-prefix for safe parsing downstream
                    await ws.send(struct.pack(">I", len(data)) + data)

        except Exception as e:
            print(f"[publisher] WS disconnected: {e}")
            # short backoff
            for _ in range(20):
                if quit_event.is_set():
                    break
                await asyncio.sleep(0.1)


async def main():
    latest_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    loop = asyncio.get_running_loop()

    t = threading.Thread(target=oak_reader_thread, args=(loop, latest_q), daemon=True)
    t.start()

    await ws_sender(latest_q)

if __name__ == "__main__":
    asyncio.run(main())
