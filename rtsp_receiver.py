#!/usr/bin/env python3
import os
import time
import cv2

RTSP_URL = "rtsp://127.0.0.1:8554/oak"

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;0"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream (OpenCV FFmpeg)")

last = time.time()
frames = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Read failed; retrying...")
        time.sleep(0.1)
        continue

    frames += 1
    now = time.time()
    if now - last >= 1.0:
        print(f"FPS: {frames/(now-last):.1f}")
        last = now
        frames = 0

    cv2.imshow("RTSP Receiver", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
