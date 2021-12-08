# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import boto3
import time
import datetime
import numpy as np


iot = boto3.client(
    'iot-data',
    aws_access_key_id="XXXXX",
    aws_secret_access_key="XXXXX",
    region_name="cn-north-1",
    endpoint_url='https://a99kufhiz4vsw.ats.iot.cn-north-1.amazonaws.com.cn')


class DummySensor(object):
    def __init__(self, mean=25, variance=1):
        self.mu = mean
        self.sigma = variance

    def read_value(self):
        return np.random.normal(self.mu, self.sigma, 1)[0]


sensor = DummySensor()
topic = 'visualanalytics/helloworld'
publish_rate = 5.0

while True:
    payload = {
        "time": str(datetime.datetime.now()),
        "value": sensor.read_value()
    }
    print("Publish {}".format(payload))
    iot.publish(
            topic=topic,
            qos=0,
            payload=json.dumps(payload, ensure_ascii=False)
        )
    time.sleep(1.0 / publish_rate)



