#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from Detectron2Detector import Detectron2Detector

def detect_image():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    detectron2_detector = Detectron2Detector()
    detectron2_detector.loadModel(model_path, config_file)

    image_path = "/home/chli/baidu/car_dataset/images/1.jpg"
    image = cv2.imread(image_path)
    result_dict = detectron2_detector.detect_image(image)
    print(result_dict)

    for box in result_dict["pred_boxes"].astype(int):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
    cv2.imshow("result", image)
    cv2.waitKey(5000)
    return True

def detect_video():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    detectron2_detector = Detectron2Detector()
    detectron2_detector.loadModel(model_path, config_file)

    video_path = "/home/chli/videos/robot-1.mp4"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_dict = detectron2_detector.detect_image(frame)
        print(result_dict)

        for box in result_dict["pred_boxes"].astype(int):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
        cv2.imshow("result", frame)
        cv2.waitKey(1)
    return True

if __name__ == "__main__":
    detect_video()

