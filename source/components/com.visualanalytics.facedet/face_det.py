# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import boto3
import cv2
import time
import datetime


iot = boto3.client(
    'iot-data',
    aws_access_key_id="AKIAXKJO247JFRMHRTUA",
    aws_secret_access_key="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",
    region_name="cn-north-1",
    endpoint_url='https://a99kufhiz4vsw.ats.iot.cn-north-1.amazonaws.com.cn')


topic = 'visualanalytics/facedet'


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=5,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    while True:
        ret_val, img = cap.read()

        # TODO: face detector inference

        payload = {
            "timestamp": str(datetime.datetime.now()),
            "height": img.shape[0],
            "width": img.shape[1],
            "channels": img.shape[2]
        }
        print("Publish {}".format(payload))
        iot.publish(
            topic=topic,
            qos=0,
            payload=json.dumps(payload, ensure_ascii=False)
        )
else:
    print("Unable to open camera.")









