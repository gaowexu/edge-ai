from dlr.counter.phone_home import PhoneHome
PhoneHome.disable_feature()
import dlr
import json
import boto3
import datetime
import numpy as np
import cv2


# 设置gstreamer管道参数
def gstreamer_pipeline(
        capture_width=1280,  # 摄像头预捕获的图像宽度
        capture_height=720,  # 摄像头预捕获的图像高度
        display_width=1280,  # 窗口显示的图像宽度
        display_height=720,  # 窗口显示的图像高度
        framerate=60,  # 捕获帧率
        flip_method=0,  # 是否旋转图像
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


iot = boto3.client(
    'iot-data',
    aws_access_key_id="AKIAXKJO247JFRMHRTUA",
    aws_secret_access_key="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",
    region_name="cn-north-1",
    endpoint_url='https://a99kufhiz4vsw.ats.iot.cn-north-1.amazonaws.com.cn')


# Load model, /path/to/model is a directory containing the compiled model artifacts (.so, .params, .json)
model = dlr.DLRModel('./objectcls', 'gpu', 0)

# Warm up
print("Model Warmup Start")
x = np.random.rand(1, 3, 224, 224)
for _ in range(5):
    y = model.run(x)
print("Model Warmup Start")


# Configuration
topic = 'visualanalytics/imageclassification'
cls_id_name_mapping = {0: "poli", 1: "calculator", 2: "telescope", 3: "macqueen", 4: "timecounter"}
capture_width = 1280
capture_height = 720
display_width = 1280
display_height = 720
framerate = 5
flip_method = 0

# create gstreamer pipeline and combine it with CSI video stream
print(gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if cap.isOpened():
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

    # capture frame data and run inference
    while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        ret_val, img = cap.read()

        # pre-process
        image = img[:, :, ::-1]
        image /= 255.0
        model_input = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # model inference
        y = model.run(model_input)
        prob_dist = dict()
        probs = y[1][0]
        for index, value in enumerate(probs):
            prob_dist[cls_id_name_mapping[int(index)]] = value

        # publish message to IoT core
        payload = {
            "time": str(datetime.datetime.now()),
            "probs": json.dumps(prob_dist)
        }
        print("Publish {}".format(payload))
        iot.publish(
            topic=topic,
            qos=0,
            payload=json.dumps(payload, ensure_ascii=False)
        )

        cv2.imshow("CSI Camera", img)

        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:  # ESC键退出
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Failed to Open Camera.")
