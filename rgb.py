#!/usr/bin/env python3

import cv2
import depthai as dai

# Create the pipeline
with dai.Pipeline() as pipeline:
    # Create the Camera node (new single unified node)
    cam = pipeline.create(dai.node.Camera).build()

    # Request an output from the camera
    videoQueue = cam.requestOutput((640, 400)).createOutputQueue()

    # Start the pipeline
    pipeline.start()

    # Loop while streaming frames
    while pipeline.isRunning():
        videoIn = videoQueue.get()  # Blocking until frame arrives
        assert isinstance(videoIn, dai.ImgFrame)

        # Display the frame using OpenCV
        cv2.imshow("video", videoIn.getCvFrame())

        if cv2.waitKey(1) == ord("q"):
            break
