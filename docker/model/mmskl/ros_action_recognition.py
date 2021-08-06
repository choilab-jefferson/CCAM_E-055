#!/usr/bin/env python3
# Human Action Recognition Pipeline on ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/26/2021

# Python libs
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

# publishing topics
action_pub = rospy.Publisher(
    "/output/action", String, queue_size=10)
pose_pub = rospy.Publisher("/output/pose", String, queue_size=10)
human_pub = rospy.Publisher(
    "/output/human", String, queue_size=10)
from action_recognition import ActionRecognitionPipeline

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
        pipeline.put_frame((color, depth))
        res = pipeline.get_result()
        msg = String()
        if res is not None:
            for i, result in enumerate(res):
                selected_label, score = result
                if score < threshold:
                    break
                text = selected_label + ': ' + str(round(score, 2))
                msg.data += text + ", "
                print(text)
            action_pub.publish(msg)


    return color, depth, cam_id

class ActionClassification(Thread):
    def __init__(self, cam_id, cfg):
        Thread.__init__(self)
        self.cam_id = cam_id
        self.show_images = False
        self.color = None
        self.depth = None
        self.pointcloud = None
        self.threshold = 0.01
        self.n_colorframe = 0
        self.n_depthframe = 0
        self.n_pointcloud = 0
        
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
                    color, depth, _ = process_rgbd(self.pipeline, color_, depth_, cam_id, self.threshold)
                    n_frame += 1

                    if self.show_images:
                        depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(cv2.resize(depth, (320,240)), alpha=0.09), cv2.COLORMAP_JET)
                        images[cam_id] = np.vstack((cv2.resize(color, (320,240)), depth_colormap))

                if rospy.get_time() - t1 > 5:
                    print(f"cam{cam_id} FPS: {n_frame/(rospy.get_time() - t1):0.2f}")
                    n_frame = 0
                    t1 = rospy.get_time()
        except KeyboardInterrupt:
            if self.pipeline is not None:
                print("Shutting down model pipeline")
                self.pipeline.stop()
        finally:
            print(f"Shutting down ROS Human Action Recogntion pipeline {cam_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Action Recogntion pipeline")
    args = parser.parse_args()

    cfg = Config.fromfile("action_recognition.yaml")
    cfg = cfg.cfg

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
