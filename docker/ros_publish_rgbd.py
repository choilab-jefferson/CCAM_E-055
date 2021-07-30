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
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField

import open3d as o3d

FIELDS_XYZBGR = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='b', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='r', offset=20, datatype=PointField.FLOAT32, count=1),
]

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def camera_pipeline(cam_sn, width, height, fps):
    # Configure depth and color streams from Camera
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(cam_sn)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    color_profile = rs.video_stream_profile(
        profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(
        profile.get_stream(rs.stream.depth))
    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()

    return {
        "cam_sn": cam_sn,
        "pipeline": pipeline,
        "config": config,
        "profile": profile,
        "depth_scale": depth_scale,
        "depth_intrinsics": depth_intrinsics,
        "color_intrinsics": color_intrinsics,
    }


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


def get_point_cloud(depth_scale, intrinsics_, color_image_, depth_image_):
    intrinsics = o3d.camera.PinholeCameraIntrinsic(intrinsics_.width, intrinsics_.height,
                                                   intrinsics_.fx, intrinsics_.fy,
                                                   intrinsics_.ppx, intrinsics_.ppy)

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 4  # 4 meter

    color_image = o3d.geometry.Image(color_image_)
    depth_image = o3d.geometry.Image(depth_image_)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)

    return pcd


def publish_camera_info(pub, color_info, depth_info):
    color_info_msg = CameraInfo()
    color_info_msg.width = color_info.width
    color_info_msg.height = color_info.height
    color_info_msg.K[2] = color_info.ppx
    color_info_msg.K[5] = color_info.ppy
    color_info_msg.K[0] = color_info.fx
    color_info_msg.K[4] = color_info.fy
    color_info_msg.D = color_info.coeffs
    color_info_msg.distortion_model = 'plumb_bob'

    depth_info_msg = CameraInfo()
    depth_info_msg.width = depth_info.width
    depth_info_msg.height = depth_info.height
    depth_info_msg.K[2] = depth_info.ppx
    depth_info_msg.K[5] = depth_info.ppy
    depth_info_msg.K[0] = depth_info.fx
    depth_info_msg.K[4] = depth_info.fy
    depth_info_msg.D = depth_info.coeffs
    depth_info_msg.distortion_model = 'plumb_bob'

    pub["cinfo"].publish(color_info_msg)
    pub["dinfo"].publish(depth_info_msg)


def publish_frames(pub, color_image, depth_image):
    color_msg = CompressedImage()
    color_msg.header.stamp = rospy.Time.now()
    color_msg.format = "jpeg"
    color_msg.data = np.array(cv2.imencode('.jpg', color_image)[1]).tobytes()

    depth_msg = CompressedImage()
    depth_msg.header.stamp = rospy.Time.now()
    depth_msg.format = "png"
    depth_msg.data = np.array(cv2.imencode('.png', depth_image)[1]).tobytes()

    # Publish new image
    pub["color"].publish(color_msg)
    pub["depth"].publish(depth_msg)


def publish_point_cloud(pub, point_cloud):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    arr = np.c_[points, colors].astype(np.float32)
    arr = np.atleast_2d(arr)

    point_cloud_msg = PointCloud2()
    point_cloud_msg.header.stamp = rospy.Time.now()
    point_cloud_msg.header.frame_id = "map"
    point_cloud_msg.height = 1
    point_cloud_msg.width = arr.shape[0]
    point_cloud_msg.fields = FIELDS_XYZBGR
    point_cloud_msg.is_bigendian = False 
    point_cloud_msg.point_step = arr.dtype.itemsize*arr.shape[1]
    point_cloud_msg.row_step = point_cloud_msg.point_step*arr.shape[0]
    point_cloud_msg.is_dense = False
    point_cloud_msg.data = arr.tobytes()

    pub["points"].publish(point_cloud_msg)


if __name__ == "__main__":
    ctx = rs.context()
    cams = []
    for i, device in enumerate(ctx.devices):
        cam_sn = device.get_info(rs.camera_info.serial_number)
        cams.append(camera_pipeline(cam_sn, 640, 480, 30))
    print(cams)

    cams = (
        cams[0], cams[1], cams[0], cams[1]
        #camera_pipeline('036322250763', 640, 480, 30),
        #camera_pipeline('038122250356', 640, 480, 30),
        #camera_pipeline('f0245993', 0, 480, 30)
        #camera_pipeline('f0245826', 0, 480, 30)
    )

    publishers = [{
        "color": rospy.Publisher(f"/camera{i}/color/compressed", CompressedImage, queue_size=2),
        "depth": rospy.Publisher(f"/camera{i}/depth/compressed", CompressedImage, queue_size=2),
        "cinfo": rospy.Publisher(f"/camera{i}/color/camera_info", CameraInfo, queue_size=2),
        "dinfo": rospy.Publisher(f"/camera{i}/depth/camera_info", CameraInfo, queue_size=2),
        "points": rospy.Publisher(f"/camera{i}/depth/points", PointCloud2, queue_size=2),
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
                    color_info, depth_info = cam["color_intrinsics"], cam["color_intrinsics"]
                    color_image, depth_image = get_rgbd(cam["pipeline"], align)
                    point_cloud = get_point_cloud(
                        cam["depth_scale"], cam["color_intrinsics"], color_image, depth_image)
                    publish_camera_info(pub, color_info, depth_info)
                    publish_frames(pub, color_image, depth_image)
                    publish_point_cloud(pub, point_cloud)
            else:
                for i, cam in enumerate(cams[0:2]):
                    color_info, depth_info = cam["color_intrinsics"], cam["color_intrinsics"]
                    color_image, depth_image = get_rgbd(cam["pipeline"], align)
                    point_cloud = get_point_cloud(
                        cam["depth_scale"], cam["color_intrinsics"], color_image, depth_image)
                    publish_camera_info(publishers[i], color_info, depth_info)
                    publish_camera_info(
                        publishers[i+2], color_info, depth_info)
                    publish_frames(publishers[i], color_image, depth_image)
                    publish_frames(publishers[i+2], color_image, depth_image)
                    publish_point_cloud(publishers[i], point_cloud)
                    publish_point_cloud(publishers[i+2], point_cloud)
    finally:
        # Stop streaming
        for cam in cams:
            cam['pipeline'].stop()