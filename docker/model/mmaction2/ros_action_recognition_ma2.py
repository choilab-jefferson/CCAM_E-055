#!/usr/bin/env python3
# Human Action Recognition Pipeline on ROS for mmaction2
#
# Wookjin Choi <wchoi@vsu.edu>
# 08/06/2021

# Python libs
from action_recognition_ma2 import ActionRecognitionPipeline
import argparse
from mmcv import Config

# numpy and scipy
import numpy as np

# OpenCV
import cv2

# Ros libraries
import rospy

# Ros Messages
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, PointCloud2

from threading import Thread

label_map = "demo/label_map_k400.txt"

# config = "configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py"
# config = "configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth"

# config = "configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"

# publishing topics
action_pub = rospy.Publisher(
    "/output/action", String, queue_size=10)
pose_pub = rospy.Publisher("/output/pose", String, queue_size=10)
human_pub = rospy.Publisher(
    "/output/human", String, queue_size=10)


def callback_imshow(event):
    #print('Timer called at ' + str(event.current_real))
    try:
        cv2.imshow(f"multiview", np.hstack(images))
    except KeyboardInterrupt:
        return
    except:
        pass
    finally:
        cv2.waitKey(1)


def process_rgbd(pipeline, color_, depth_, cam_id, threshold):
    color = cv2.imdecode(color_, cv2.IMREAD_COLOR)
    depth = cv2.imdecode(depth_, cv2.IMREAD_UNCHANGED)

    if cam_id == 0:
        pipeline.put_frame((np.array(color[:, :, ::-1]), depth)) # bgr to rgb
        res = pipeline.get_result()
        if res is not None:
            msg = String()
            for i, result in enumerate(res):
                selected_label, score = result
                if score < threshold:
                    break
                text = selected_label + ': ' + str(round(score, 2))
                msg.data += text + ", "
            action_pub.publish(msg)

    return color, depth, cam_id


class ActionClassification(Thread):
    def __init__(self, cam_id, cfg):
        Thread.__init__(self)
        self.cam_id = cam_id
        self.show_images = True
        self.color = None
        self.depth = None
        self.pointcloud = None
        self.n_colorframe = 0
        self.n_depthframe = 0
        self.n_pointcloud = 0
        self.threshold = cfg.threshold

        if cam_id == 0:
            self.pipeline = ActionRecognitionPipeline(cfg)
        else:
            self.pipeline = None

    def callback_color(self, ros_data):
        #print("color", self.cam_id, ros_data.format.split(";"))
        self.color = np.frombuffer(ros_data.data, np.uint8)
        self.n_colorframe += 1

    def callback_depth(self, ros_data):
        #print("depth", self.cam_id, ros_data.format.split(";"))
        self.depth = np.frombuffer(ros_data.data, np.uint8)
        self.n_depthframe += 1

    def callback_pointcloud(self, ros_data):
        #print("pointcloud", self.cam_id)
        self.pointcloud = ros_data
        self.n_pointcloud += 1

    def run(self):
        cam_id = self.cam_id

        # subscribed topics
        self.subscribers = {
            "color": rospy.Subscriber(f"/camera{cam_id}/color/compressed", CompressedImage, self.callback_color, queue_size=2),
            "depth": rospy.Subscriber(f"/camera{cam_id}/depth/compressed", CompressedImage, self.callback_depth, queue_size=2),
            "pointcloud": rospy.Subscriber(f"/camera{cam_id}/depth/points", PointCloud2, self.callback_pointcloud, queue_size=2),
        }

        t1 = rospy.get_time()
        n_frame = 0
        try:
            while not rospy.is_shutdown():
                if self.color is None or self.depth is None:
                    pass
                else:
                    color_, depth_ = self.color, self.depth
                    self.color, self.depth = None, None
                    color, depth, _ = process_rgbd(
                        self.pipeline, color_, depth_, cam_id, self.threshold)
                    n_frame += 1

                    if self.show_images:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(cv2.resize(depth, (320, 240)), alpha=0.09), cv2.COLORMAP_JET)
                        images[cam_id] = np.vstack(
                            (cv2.resize(color, (320, 240)), depth_colormap))

                if n_frame > 0 and rospy.get_time() - t1 > 5:
                    print(
                        f"cam{cam_id} FPS: {n_frame/(rospy.get_time() - t1):0.2f}")
                    n_frame = 0
                    t1 = rospy.get_time()
        except KeyboardInterrupt:
            if self.pipeline is not None:
                print("Shutting down model pipeline")
                self.pipeline.stop()
        finally:
            print(
                f"Shutting down ROS Human Action Recogntion pipeline {cam_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Action Recogntion pipeline")
    args = parser.parse_args()

    with open(label_map, 'r') as f:
        label = [line.strip() for line in f]

    model_cfg = Config.fromfile(config)
    cfg = Config(dict(model_cfg=model_cfg, label_map=label_map, label=label,
                 threshold=0.01, inference_fps=4, drawing_fps=20,
                 checkpoint_file=checkpoint, device='cuda:0', average_size=1, gpus=1, worker_per_gpu=1))
    rospy.init_node('Human_Action_Recogntion', anonymous=True)
    acs = [ActionClassification(cam_id, cfg) for cam_id in range(4)]

    try:
        images = [None for _ in range(4)]
        for ac in acs:
            ac.start()
        rospy.Timer(rospy.Duration(1./cfg.drawing_fps), callback_imshow)
        rospy.spin()
    finally:
        for ac in acs:
            ac.join()
        print("Shutting down ROS Human Action Recogntion pipelines")
