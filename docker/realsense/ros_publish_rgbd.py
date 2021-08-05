#!/usr/bin/env python3
# rs cameras to ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/26/2021

import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from threading import Thread
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


class D400_Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class L500_Preset(IntEnum):
    Custom = 0
    NoAmbientLight = 1
    LowAmbientLight = 2
    MaxRange = 3
    ShortRange = 4


def camera_pipeline(cam_sn, width, height, fps):
    # Configure depth and color streams from Camera
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(cam_sn)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start streaming
    try:
        profile = pipeline.start(config)
    except:
        config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, fps)
        profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, D400_Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

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
        "align": align,
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
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # Validate that both frames are valid
    if not depth_frame or not color_frame:
        return None, None

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


def get_pointcloud(depth_scale, intrinsics_, color_image_, depth_image_):
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics_.width, intrinsics_.height, intrinsics_.fx, intrinsics_.fy, intrinsics_.ppx, intrinsics_.ppy)

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
    uni_down_pcd = pcd.uniform_down_sample(3)

    return uni_down_pcd


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
    # maybe not because it is an inverse
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


def publish_pointcloud(pub, pointcloud):
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    arr = np.c_[points, colors].astype(np.float32)
    arr = np.atleast_2d(arr)

    pointcloud_msg = PointCloud2()
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header.frame_id = "map"
    pointcloud_msg.height = 1
    pointcloud_msg.width = arr.shape[0]
    pointcloud_msg.fields = FIELDS_XYZBGR
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = arr.dtype.itemsize*arr.shape[1]
    pointcloud_msg.row_step = pointcloud_msg.point_step*arr.shape[0]
    pointcloud_msg.is_dense = False
    pointcloud_msg.data = arr.tobytes()

    pub["points"].publish(pointcloud_msg)


def process_camera(cam_id):
    cam_sn = cams[cam_id]["cam_sn"]
    print(cam_sn)
    if cam_id >= len(cams_):
        while "pipeline" not in cams[cam_id-len(cams_)]:
            print(cam_sn, "no pipeline", id(cams))
            rospy.sleep(1)
        cam = cams[cam_id-len(cams_)].copy()
    else:
        try:
            cam = camera_pipeline(cam_sn, 640, 480, 30)
        except:
            cam = camera_pipeline(cam_sn, 960, 540, 30)
    cam["cam_id"] = cam_id
    cams[cam_id] = cam
    print(cam)
    
    pub = dict(
        color = rospy.Publisher(f"/camera{cam_id}/color/compressed", CompressedImage, queue_size=2),
        depth = rospy.Publisher(f"/camera{cam_id}/depth/compressed", CompressedImage, queue_size=2),
        cinfo = rospy.Publisher(f"/camera{cam_id}/color/camera_info", CameraInfo, queue_size=2),
        dinfo = rospy.Publisher(f"/camera{cam_id}/depth/camera_info", CameraInfo, queue_size=2),
        points = rospy.Publisher(f"/camera{cam_id}/depth/points", PointCloud2, queue_size=2),
    )

    n_frame = 0
    t1 = rospy.get_time()
    while not rospy.is_shutdown():
        color_info, depth_info = cam["color_intrinsics"], cam["depth_intrinsics"]
        publish_camera_info(pub, color_info, depth_info)
        color_image, depth_image = get_rgbd(cam["pipeline"], cam["align"])
        publish_frames(pub, color_image, depth_image)
        pointcloud = get_pointcloud(
            cam["depth_scale"], cam["color_intrinsics"], color_image, depth_image)
        publish_pointcloud(pub, pointcloud)
        n_frame += 1
        if rospy.get_time() - t1 > 1 and n_frame > 0:
            print(f"cam{cam_id} FPS: {n_frame/(rospy.get_time() - t1):0.2f}")
            n_frame = 0
            t1 = rospy.get_time()


if __name__ == "__main__":
    ctx = rs.context()
    cams_ = []
    for cam_id, device in enumerate(ctx.devices):
        cam_sn = device.get_info(rs.camera_info.serial_number)
        cams_.append(dict(cam_id=cam_id, cam_sn=cam_sn))

    # simulate 4 camera streams
    cams = cams_ * 4
    cams = cams[0:4]

    rospy.init_node('rs_cameras', anonymous=True, xmlrpc_port=45100, tcpros_port=45101)

    # Streaming loop
    try:
        procs = []
        for cam_id, cam in enumerate(cams):
            p = Thread(target=process_camera, args=(cam_id,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    finally:
        # Stop streaming
        for cam in cams_:
            cams[cam["cam_id"]]["pipeline"].stop()
