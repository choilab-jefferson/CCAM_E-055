#!/usr/bin/env python3
# Human Action Classification Pipeline on ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/26/2021

# Python libs
import argparse
import sys
import time
import random

from collections import deque
from multiprocessing.pool import Pool, ThreadPool

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
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


def human_detection(color, n_colorframe):
    #print("human", n_colorframe)
    #human_pub.publish(f"human, {n_colorframe}")
    return 0

def pose_estimation(color, depth, locs, n_colorframe, n_depthframe):
    #print("pose", n_colorframe, n_depthframe)
    #pose_pub.publish(f"pose, {n_colorframe}, {n_depthframe}")
    return 0

def action_classification(poses, n_colorframe):
    #print("action", n_colorframe)
    #action_pub.publish(f"action, {n_colorframe}")
    return 0

def process_rgbd(color_, depth_, n_colorframe, n_depthframe, cam_id):
    color = cv2.imdecode(color_, cv2.IMREAD_COLOR)
    depth = cv2.imdecode(depth_, cv2.IMREAD_UNCHANGED)
    #print(f"cam{cam_id} frame: color-{n_colorframe}, depth-{n_depthframe}, {rospy.get_time()-t2:0.3f} secs")
    #print(hex(id(color_)), color.shape, hex(id(depth_)), depth.shape)

    locs = human_detection(color, n_colorframe)
    poses = pose_estimation(color, depth, locs, n_colorframe, n_depthframe)
    action_classification(poses, n_colorframe)

    return color, depth, cam_id

class ActionClassification(Thread):
    def __init__(self, n):
        Thread.__init__(self)
        self.n_cam = n
        self.color = [None for _ in range(self.n_cam)]
        self.depth = [None for _ in range(self.n_cam)]
        self.point_cloud = [None for _ in range(self.n_cam)]
        self.n_colorframe = [0 for _ in range(self.n_cam)]
        self.n_depthframe = [0 for _ in range(self.n_cam)]
        self.n_point_cloud = [0 for _ in range(self.n_cam)]

        # subscribed topics
        self.subscribers = [{
            "color": rospy.Subscriber(f"/camera{i}/color/compressed", CompressedImage, self.callback_color, callback_args=i, queue_size=2),
            "depth": rospy.Subscriber(f"/camera{i}/depth/compressed", CompressedImage, self.callback_depth, callback_args=i, queue_size=2),
            "point_cloud": rospy.Subscriber(f"/camera{i}/depth/points", PointCloud2, self.callback_point_cloud, callback_args=i, queue_size=2),
        } for i in range(self.n_cam)]

        if False:
            rospy.Timer(rospy.Duration(0.033), self.callback_imshow)
            self.images = [None for _ in range(self.n_cam)]

    def callback_color(self, ros_data, cam_id):
        #print("color", ros_data.format.split(";"))
        self.color[cam_id] = np.frombuffer(ros_data.data, np.uint8)
        self.n_colorframe[cam_id] += 1

    def callback_depth(self, ros_data, cam_id):
        #print("depth", ros_data.format.split(";"))
        # remove header from raw data
        self.depth[cam_id] = np.frombuffer(ros_data.data, np.uint8)
        self.n_depthframe[cam_id] += 1
    
    def callback_point_cloud(self, ros_data, cam_id):
        self.point_cloud[cam_id] = ros_data
        self.n_point_cloud[cam_id] += 1

    def callback_imshow(self, event):
        #print('Timer called at ' + str(event.current_real))
        try:
            cv2.imshow(f"multiview", np.hstack(self.images))
        except:
            pass
        finally:
            cv2.waitKey(1)

    def run(self):
        t1 = rospy.get_time()
        n_no_frame = [0 for _ in range(self.n_cam)]
        n_frame = [0 for _ in range(self.n_cam)]
        while not rospy.is_shutdown():
            # # Consume the queue.
            # while len(pending_task) > 0 and pending_task[0].ready():
            #     color, depth, cam_id = pending_task.popleft().get()
            #     n_frame[cam_id] += 1

            for cam_id in range(self.n_cam):
                if self.color[cam_id] is None or self.depth[cam_id] is None:
                    pass
                else:
                    color_, depth_ = self.color[cam_id], self.depth[cam_id]
                    n_colorframe, n_depthframe = self.n_colorframe[cam_id], self.n_depthframe[cam_id]
                    self.color[cam_id], self.depth[cam_id] = None, None
                    if False: # multi thread
                        task = pool.apply_async(process_rgbd, (color_, depth_, n_colorframe, n_depthframe, cam_id))
                        pending_task.append(task)
                    else:
                        color, depth, _ = process_rgbd(color_, depth_, n_colorframe, n_depthframe, cam_id)
                        n_frame[cam_id] += 1

                    #depth_colormap = cv2.applyColorMap(
                    #    cv2.convertScaleAbs(cv2.resize(depth, (320,240)), alpha=0.09), cv2.COLORMAP_JET)
                    #self.images[cam_id] = np.vstack((cv2.resize(color, (320,240)), depth_colormap))

            if time.time() - t1 > 1:
                for cam_id in range(self.n_cam):
                    print(
                        f"cam{cam_id} FPS: {n_frame[cam_id]/(time.time() - t1):0.2f}")
                    n_frame[cam_id] = 0
                t1 = rospy.get_time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human Action Classification pipeline")
    args = parser.parse_args()

    # #pool = Pool(processes=4)
    # pool = ThreadPool(processes=4)
    # pending_task = deque()

    rospy.init_node('Human_Action_Classification', anonymous=True)
    ac = ActionClassification(4)

    ac.start()
    rospy.spin()
    print("Shutting down ROS Human Action Classification pipeline")