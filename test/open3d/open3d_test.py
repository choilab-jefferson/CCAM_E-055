# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import numpy as np
from enum import IntEnum

from datetime import datetime
import open3d as o3d
import cv2
import logging
import copy


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


def get_point_cloud(pipeline, pcd, depth_scale, flip_transform):
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        get_intrinsic_matrix(color_frame))

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        return -1

    depth_image = o3d.geometry.Image(
        np.array(aligned_depth_frame.get_data()))
    color_temp = np.asarray(color_frame.get_data())
    color_image = o3d.geometry.Image(color_temp)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    temp.transform(flip_transform)
    pcd.points = temp.points
    pcd.colors = temp.colors

    return 0


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


if __name__ == "__main__":
    # Configure depth and color streams...
    # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    #config_1.enable_device('f0245826')
    config_1.enable_device('036322250763')
    config_1.enable_stream(rs.stream.depth, 0, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 0, 480, rs.format.rgb8, 30)

    
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    #config_2.enable_device('f0245993')
    config_2.enable_device('038122250356')
    config_2.enable_stream(rs.stream.depth, 0, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 0, 480, rs.format.rgb8, 30)


    # Start streaming from both cameras
    profile_1 = pipeline_1.start(config_1)
    profile_2 = pipeline_2.start(config_2)

    depth_sensor_1 = profile_1.get_device().first_depth_sensor()
    depth_sensor_2 = profile_2.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor_1.set_option(rs.option.visual_preset, Preset.HighAccuracy)
    depth_sensor_2.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale_1 = depth_sensor_1.get_depth_scale()
    depth_scale_2 = depth_sensor_2.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2  # 3 meter
    clipping_distance_1 = clipping_distance_in_meters / depth_scale_1
    clipping_distance_2 = clipping_distance_in_meters / depth_scale_2
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = o3d.visualization.Visualizer()

    pcd_1 = o3d.geometry.PointCloud()
    pcd_2 = o3d.geometry.PointCloud()
    flip_transform_1 = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    flip_transform_2 = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # Streaming loop
    frame_count = 0
    try:
        while True:
            dt0 = datetime.now()
            # Camera 1
            if get_point_cloud(pipeline_1, pcd_1, depth_scale_1, flip_transform_1) < 0:
                continue

            # Camera 2
            if get_point_cloud(pipeline_2, pcd_2, depth_scale_2, flip_transform_2) < 0:
                continue
            
            if frame_count == 0:
                voxel_size = 0.05  # means 5cm for this dataset
                source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_2, pcd_1, voxel_size)

                result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
                print(result_ransac)
                draw_registration_result(source_down, target_down, result_ransac.transformation)

                result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
                print(result_icp)
                draw_registration_result(source, target, result_icp.transformation)
                
                flip_transform_2 = np.dot(result_icp.transformation, np.array(flip_transform_2))
                vis.create_window()
                vis.add_geometry(pcd_1)
                vis.add_geometry(pcd_2)

            vis.update_geometry(pcd_1)
            vis.update_geometry(pcd_2)
            vis.poll_events()
            vis.update_renderer()

            process_time = datetime.now() - dt0
            print("FPS: " + str(1 / process_time.total_seconds()))
            frame_count += 1

    finally:

        # Stop streaming
        pipeline_1.stop()
        pipeline_2.stop()

    vis.destroy_window()
