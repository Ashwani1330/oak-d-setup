#!/usr/bin/env python3
import subprocess, signal, threading
import depthai as dai

W, H = 640, 400
FPS = 30
RTSP_URL = "rtsp://192.168.1.100:8554/oakraw"

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

ffmpeg_cmd = [
    "ffmpeg", "-loglevel", "warning",
    "-fflags", "+genpts",
    "-use_wallclock_as_timestamps", "1",

    # raw frames coming from python
    "-f", "rawvideo",
    "-pix_fmt", "nv12",
    "-video_size", f"{W}x{H}",
    "-framerate", str(FPS),
    "-i", "pipe:0",

    # keep it raw but in a more standard RTP-friendly format
    "-vf", "format=yuv420p",
    "-c:v", "rawvideo",

    # publish via RTSP (media will be RTP/UDP if you choose udp)
    "-f", "rtsp",
    "-rtsp_transport", "udp",
    "-muxdelay", "0",
    "-muxpreload", "0",
    RTSP_URL,
]

def main():
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        out = cam.requestOutput(
            (W, H),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            FPS
        )

        q = out.createOutputQueue(maxSize=1, blocking=False)
        pipeline.start()

        try:
            while pipeline.isRunning() and not quit_event.is_set():
                frm = q.tryGet()
                if frm is None:
                    continue

                # freshest-frame-wins (drop backlog)
                while True:
                    newer = q.tryGet()
                    if newer is None:
                        break
                    frm = newer

                if proc.poll() is not None:
                    print("ffmpeg exited (publish failed).")
                    break

                try:
                    proc.stdin.write(frm.getData().tobytes())
                except BrokenPipeError:
                    print("ffmpeg pipe closed.")
                    break
        finally:
            try: proc.stdin.close()
            except Exception: pass
            proc.terminate()
            pipeline.stop()
            pipeline.wait()

if __name__ == "__main__":
    main()
