#!/usr/bin/env python3
import subprocess, signal, threading, time
import depthai as dai

WIDTH, HEIGHT = 640, 400
FPS = 30
RTSP_URL = "rtsp://192.168.1.185:8554/oak"
# RTSP_URL = "rtsp://10.241.166.98:8554/oak"

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

ffmpeg_cmd = [
    "ffmpeg", "-loglevel", "warning",
    "-r", str(FPS),
    "-f", "h264", "-i", "pipe:0",
    "-c:v", "copy",
    "-f", "rtsp",
    "-rtsp_transport", "udp",
    "-pkt_size", "1200",
    "-muxdelay", "0",
    "-muxpreload", "0",
    RTSP_URL,
]

def main():
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        cam_out = cam.requestOutput(
            (WIDTH, HEIGHT),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            FPS
        )

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_out, frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )

        try: enc.setNumBFrames(0)
        except Exception: pass
        try: enc.setKeyframeFrequency(10)   # ~0.33s
        except Exception: pass
        try: enc.setBitrateKbps(1500)       # start lower; raise if stable
        except Exception: pass

        q = enc.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        try:
            while pipeline.isRunning() and not quit_event.is_set():
                pkt = q.tryGet()
                if pkt is None:
                    time.sleep(0.001)
                    continue
                try:
                    proc.stdin.write(pkt.getData().tobytes())
                except BrokenPipeError:
                    print("FFmpeg pipe closed (publish failed).")
                    break
        finally:
            try: proc.stdin.close()
            except Exception: pass
            proc.terminate()
            pipeline.stop()
            pipeline.wait()

if __name__ == "__main__":
    main()
