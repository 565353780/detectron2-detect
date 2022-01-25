# Copyright (c) Facebook, Inc. and its affiliates.

import multiprocessing as mp
import numpy as np
import os
import tempfile
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from predictor import VisualizationDemo

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def get_auto_cfg(model_path, config_file, confidence_threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def demo_detect_image():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    mp.set_start_method("spawn", force=True)
    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    path = "/home/chli/baidu/car_dataset/images/1.jpg"
    out_filename = "/home/chli/baidu/test.jpg"
    img = read_image(path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)
    visualized_output.save(out_filename)
    basename = os.path.basename(path)
    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    cv2.imshow(basename, visualized_output.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        return True # esc to quit
    return True

def demo_detect_video():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    mp.set_start_method("spawn", force=True)
    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    video_input = "/home/chli/videos/robot-1.mp4"
    output = "/home/chli/videos/robot-1_semantic.mp4"
    video = cv2.VideoCapture(video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_input)
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v")
    output_fname = output
    assert not os.path.isfile(output_fname), output_fname
    output_file = cv2.VideoWriter(
        filename=output_fname,
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*codec),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    assert os.path.isfile(video_input)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        output_file.write(vis_frame)
        cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
        cv2.imshow(basename, vis_frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    video.release()
    output_file.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    demo_detect_video()

