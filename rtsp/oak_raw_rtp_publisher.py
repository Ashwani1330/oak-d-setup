#!/usr/bin/env python3
import subprocess, signal, threading, time
import depthai as dai

W, H = 640, 400
FPS = 30
SERVER_IP = "192.168.1.100"
RTP_PORT = 5004

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

ffmpeg_cmd = [
    "ffmpeg", "-loglevel", "warning",
    "-f", "rawvideo",
    "-pix_fmt", "nv12",
    "-video_size", f"{W}x{H}",
    "-framerate", str(FPS),
    "-i", "pipe:0",
    "-an",
    "-f", "rtp",
    "-sdp_file", "oak_raw.sdp",
    f"rtp://{SERVER_IP}:{RTP_PORT}?pkt_size=1200",
]

def main():
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        out = cam.requestOutput((W, H), dai.ImgFrame.Type.NV12, dai.ImgResizeMode.CROP, FPS)
        q = out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        try:
            while pipeline.isRunning() and not quit_event.is_set():
                frm = q.tryGet()
                if frm is None:
                    continue

                # drop backlog (freshest-frame-wins)
                while True:
                    newer = q.tryGet()
                    if newer is None:
                        break
                    frm = newer

                proc.stdin.write(frm.getData().tobytes())
        finally:
            try: proc.stdin.close()
            except Exception: pass
            proc.terminate()
            pipeline.stop()
            pipeline.wait()

if __name__ == "__main__":
    main()
