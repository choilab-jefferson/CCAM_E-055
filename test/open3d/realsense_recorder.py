# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join
import shutil
import json
from enum import IntEnum

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--output_folder",
                        default='dataset/realsense/',
                        help="set output folder")
    parser.add_argument("--record_rosbag",
                        action='store_true',
                        help="Recording rgbd stream into realsense.bag")
    parser.add_argument(
        "--record_imgs",
        action='store_true',
        help="Recording save color and depth images into realsense folder")
    parser.add_argument("--playback_rosbag",
                        action='store_true',
                        help="Play recorded realsense.bag file")
    parser.add_argument(
        "--view_imgs",
        action='store_true',
        help="Showing color and depth images")
    args = parser.parse_args()

    if sum(o is not False for o in vars(args).values()) != 2:
        parser.print_help()
        exit()

    path_output = args.output_folder
    path_depth = join(args.output_folder, "depth")
    path_color = join(args.output_folder, "color")
    if args.record_imgs:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    path_bag = join(args.output_folder, "realsense.bag")
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()

    # Configure depth and color streams...
    # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()

    if args.record_imgs or args.record_rosbag:
        #config_1.enable_device('f0245826')
        config_1.enable_device('036322250763')
        config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        #config_2.enable_device('f0245993')
        config_2.enable_device('038122250356')
        config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if args.record_rosbag:
            config_1.enable_record_to_file(path_bag)
            config_2.enable_record_to_file(path_bag)


    # Start streaming from both cameras
    profile_1 = pipeline_1.start(config_1)
    profile_2 = pipeline_2.start(config_2)

    depth_sensor_1 = profile_1.get_device().first_depth_sensor()
    depth_sensor_2 = profile_2.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor_1.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        depth_sensor_2.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale_1 = depth_sensor_1.get_depth_scale()
    depth_scale_2 = depth_sensor_2.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance_1 = clipping_distance_in_meters / depth_scale_1
    clipping_distance_2 = clipping_distance_in_meters / depth_scale_2

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    frame_count = 0
    try:
        while frame_count < 100:
            # Get frameset of color and depth
            frames_1 = pipeline_1.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames_1 = align.process(frames_1)

            # Get aligned frames
            aligned_depth_frame_1 = aligned_frames_1.get_depth_frame()
            color_frame_1 = aligned_frames_1.get_color_frame()

            # Get frameset of color and depth
            frames_2 = pipeline_2.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames_2 = align.process(frames_2)

            # Get aligned frames
            aligned_depth_frame_2 = aligned_frames_2.get_depth_frame()
            color_frame_2 = aligned_frames_2.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame_1 or not color_frame_1 or not aligned_depth_frame_2 or not color_frame_2:
                continue

            depth_image_1 = np.asanyarray(aligned_depth_frame_1.get_data())
            color_image_1 = np.asanyarray(color_frame_1.get_data())
            depth_image_2 = np.asanyarray(aligned_depth_frame_2.get_data())
            color_image_2 = np.asanyarray(color_frame_2.get_data())

            if args.record_imgs:
                if frame_count == 0:
                    save_intrinsic_as_json(
                        join(args.output_folder, "camera_intrinsic_1.json"),
                        color_frame_1)
                    save_intrinsic_as_json(
                        join(args.output_folder, "camera_intrinsic_2.json"),
                        color_frame_2)
                cv2.imwrite("%s/C1_%06d.png" % \
                        (path_depth, frame_count), depth_image_1)
                cv2.imwrite("%s/C1_%06d.jpg" % \
                        (path_color, frame_count), color_image_1)
                cv2.imwrite("%s/C2_%06d.png" % \
                        (path_depth, frame_count), depth_image_2)
                cv2.imwrite("%s/C2_%06d.jpg" % \
                        (path_color, frame_count), color_image_2)
                print("Saved color + depth image %06d" % frame_count)
                frame_count += 1

            if args.view_imgs:
                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                #depth image is 1 channel, color is 3 channels
                depth_image_3d_1 = np.dstack((depth_image_1, depth_image_1, depth_image_1))
                bg_removed_1 = np.where((depth_image_3d_1 > clipping_distance_1) | \
                        (depth_image_3d_1 <= 0), grey_color, color_image_1)
                
                depth_image_3d_2 = np.dstack((depth_image_2, depth_image_2, depth_image_2))
                bg_removed_2 = np.where((depth_image_3d_2 > clipping_distance_2) | \
                        (depth_image_3d_2 <= 0), grey_color, color_image_2)


                # Render images
                depth_colormap_1 = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image_1, alpha=0.09), cv2.COLORMAP_JET)
                depth_colormap_2 = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image_2, alpha=0.09), cv2.COLORMAP_JET)
                images = np.vstack((np.hstack((bg_removed_1, depth_colormap_1)),
                                   np.hstack((bg_removed_2, depth_colormap_2))))
                cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Recorder Realsense', images)
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break
    finally:
        pipeline_1.stop()
        pipeline_2.stop()
