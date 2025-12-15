#!/usr/bin/env python3
import subprocess
import signal
import threading
import depthai as dai

WIDTH, HEIGHT = 640, 400
FPS = 30
RTSP_URL = "rtsp://192.168.1.213:8554/oak"   # ns_srv IP

quit_event = threading.Event()
signal.signal(signal.SIGINT, lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "warning",

    # raw H.264 from stdin
    "-r", str(FPS),
    "-f", "h264",
    "-i", "pipe:0",

    "-c:v", "copy",

    # publish via RTSP, but RTP over UDP (lower latency than TCP interleaving)
    "-f", "rtsp",
    "-rtsp_transport", "udp",
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
            cam_out,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )

        # Robotics-oriented encoder knobs (if supported in your DepthAI build)
        try:
            enc.setNumBFrames(0)
        except Exception:
            pass
        try:
            enc.setKeyframeFrequency(FPS)   # keyframe every 1s
        except Exception:
            pass
        try:
            enc.setBitrateKbps(2000)        # tune: 1500–5000 depending on Wi-Fi
        except Exception:
            pass

        q = enc.out.createOutputQueue(maxSize=60, blocking=True)

        pipeline.start()
        try:
            while pipeline.isRunning() and not quit_event.is_set():
                pkt = q.get()
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


'''
ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "info",
    "-fflags", "+genpts",
    "-use_wallclock_as_timestamps", "1",
    "-r", str(FPS),
    "-f", "h264",
    "-i", "pipe:0",
    "-c:v", "copy",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "-muxdelay", "0",
    "-muxpreload", "0",
    RTSP_URL,
]

'''
