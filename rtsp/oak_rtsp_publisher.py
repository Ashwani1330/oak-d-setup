#!/usr/bin/env python3
import os, subprocess, signal, threading, time
import depthai as dai

W, H = 640, 400
FPS = 30
RTSP_URL = "rtsp://192.168.1.182:8554/oak"

quit_event = threading.Event()
signal.signal(signal.SIGINT,  lambda *_: quit_event.set())
signal.signal(signal.SIGTERM, lambda *_: quit_event.set())

RTP_TICKS = 90000 // FPS  # 3000 for 30fps

ffmpeg_cmd = [
    "ffmpeg", "-nostdin", "-loglevel", "warning",

    "-f", "h264", "-i", "pipe:0",
    "-c:v", "copy",

    # force monotonic timestamps for RTP/RTSP
    "-bsf:v", f"setts=ts=N*{RTP_TICKS}:duration={RTP_TICKS}:time_base=1/90000",

    "-flush_packets", "1",
    "-muxdelay", "0", "-muxpreload", "0",

    "-f", "rtsp",
    "-rtsp_transport", "udp",
    "-pkt_size", "1200",
    RTSP_URL,
]
'''
ffmpeg_cmd = [
    "ffmpeg", "-loglevel", "warning",
    "-nostdin",

    # force input timing (don't let ffmpeg guess)
    "-fflags", "+genpts",
    "-use_wallclock_as_timestamps", "1",
    "-r", str(FPS),              # tell ffmpeg the intended fps
    "-probesize", "32",
    "-analyzeduration", "0",

    # raw H264 from stdin
    "-f", "h264",
    "-i", "pipe:0",

    "-c:v", "copy",

    # push immediately
    "-flush_packets", "1",
    "-muxdelay", "0",
    "-muxpreload", "0",

    "-f", "rtsp",
    "-rtsp_transport", "udp",
    "-pkt_size", "1200",
    RTSP_URL,
]

'''

def main():
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=0)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        cam_out = cam.requestOutput(
            (W, H),
            dai.ImgFrame.Type.NV12,
            dai.ImgResizeMode.CROP,
            FPS
        )

        enc = pipeline.create(dai.node.VideoEncoder).build(
            cam_out,
            frameRate=FPS,
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        )

        # Encoder tuning (DepthAI supports these APIs; some builds may not)
        try:
            enc.setNumBFrames(0)
        except Exception:
            pass

        # Start with 0.5–1.0s IDR. Too-frequent IDRs => packet bursts => loss.
        try:
            enc.setKeyframeFrequency(FPS)      # 1.0s
            # enc.setKeyframeFrequency(FPS // 2) # 0.5s (try if stable)
        except Exception:
            pass

        # Use CBR-ish behavior if available to avoid bitrate spikes over Wi-Fi.
        try:
            enc.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.CBR)
        except Exception:
            pass

        # Tune bitrate to *zero packet loss* first, then increase slowly.
        try:
            enc.setBitrateKbps(1200)  # start 800–1500 for Wi-Fi, raise later
        except Exception:
            pass

        q = enc.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        try:
            while pipeline.isRunning() and not quit_event.is_set():
                pkt = q.tryGet()
                if pkt is None:
                    time.sleep(0.0005)
                    continue

                # freshest-packet-wins
                while True:
                    newer = q.tryGet()
                    if newer is None:
                        break
                    pkt = newer

                if proc.poll() is not None:
                    print("ffmpeg exited (publish failed).")
                    break

                try:
                    proc.stdin.write(pkt.getData().tobytes())
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
