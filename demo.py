# Copyright (c) Facebook, Inc. and its affiliates.

import os
import cv2
import tqdm

from detectron2.config import get_cfg

from predictor import VisualizationDemo

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

    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    path = "/home/chli/baidu/car_dataset/images/1.jpg"
    img = cv2.imread(path)
    _, visualized_output = demo.run_on_image(img)
    basename = os.path.basename(path)
    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    cv2.imshow(basename, visualized_output.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        return True # esc to quit
    return True

def demo_save_image():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    path = "/home/chli/baidu/car_dataset/images/1.jpg"
    out_filename = "/home/chli/baidu/test.jpg"
    img = cv2.imread(path)
    _, visualized_output = demo.run_on_image(img)
    visualized_output.save(out_filename)
    return True

def demo_detect_video():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    video_input = "/home/chli/videos/robot-2.mp4"
    video = cv2.VideoCapture(video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_input)
    assert os.path.isfile(video_input)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
        cv2.imshow(basename, vis_frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    video.release()
    cv2.destroyAllWindows()
    return True

def demo_save_video():
    model_path = "/home/chli/.ros/model_final_a3ec72.pkl"
    config_file = "/home/chli/.ros/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg = get_auto_cfg(model_path, config_file)
    demo = VisualizationDemo(cfg)

    video_input = "/home/chli/videos/robot-3.mp4"
    output = "/home/chli/videos/robot-3_semantic.mp4"
    video = cv2.VideoCapture(video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fname = output
    assert not os.path.isfile(output_fname), output_fname
    output_file = cv2.VideoWriter(
        filename=output_fname,
        fourcc=cv2.VideoWriter_fourcc(*"MP4V"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    assert os.path.isfile(video_input)
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
        output_file.write(vis_frame)
    video.release()
    output_file.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    demo_save_video()

