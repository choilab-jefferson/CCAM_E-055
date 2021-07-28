#!/usr/bin/env python3
# rs cameras to ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/26/2021

import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
import cv2

import roslib
import rospy
from sensor_msgs.msg import CompressedImage


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def camera_pipeline(cam_id, width, height, fps):
    # Configure depth and color streams from Camera
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(cam_id)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start streaming
    profile = pipeline.start(config)

    return {"cam_id": cam_id, "pipeline": pipeline, "config": config, "profile": profile}


def get_rgbd(pipeline, align):
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not depth_frame or not color_frame:
        return None, None

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


def publish_frames(pub, color_image, depth_image):
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', color_image)[1]).tobytes()
    # Publish new image
    pub["color"].publish(msg)

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "png"
    msg.data = np.array(cv2.imencode('.png', depth_image)[1]).tobytes()
    # Publish new image
    pub["depth"].publish(msg)    


if __name__ == "__main__":
    cam0 = camera_pipeline('036322250763', 640, 480, 30)
    cam1 = camera_pipeline('038122250356', 640, 480, 30)
    cams = (
        cam0, cam1, cam0, cam1,
        #camera_pipeline('036322250763', 640, 480, 30),
        #camera_pipeline('038122250356', 640, 480, 30),
        #camera_pipeline('f0245993', 0, 480, 30)
        #camera_pipeline('f0245826', 0, 480, 30)
    )

    publishers = [{
        "color": rospy.Publisher(f"/output/color{i}/compressed", CompressedImage, queue_size=2),
        "depth": rospy.Publisher(f"/output/depth{i}/compressed", CompressedImage, queue_size=2),
    } for i in range(len(cams))]

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    rospy.init_node('rs_cameras', anonymous=True)

    # Streaming loop
    frame_count = 0
    try:
        while not rospy.is_shutdown():
            if False:
                for cam, pub in zip(cams, publishers):
                    color_image, depth_image = get_rgbd(cam["pipeline"], align)
                    publish_frames(pub, color_image, depth_image)
            else:
                for i, cam in enumerate(cams[0:2]):
                    color_image, depth_image = get_rgbd(cam["pipeline"], align)
                    publish_frames(publishers[i], color_image, depth_image)
                    publish_frames(publishers[i+2], color_image, depth_image)
    finally:
        # Stop streaming
        for cam in cams:
            cam['pipeline'].stop()
