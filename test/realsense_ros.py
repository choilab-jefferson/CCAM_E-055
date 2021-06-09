## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge

# Ros libraries
import roslib
import rospy

# Ros Messages
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


if device_product_line == 'L500':
    #config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

bridge = CvBridge()

# topic where we publish
image_pub = rospy.Publisher(
    "/output/color/compressed", CompressedImage, queue_size=2)
image_pub1 = rospy.Publisher(
    #"/output/depth/compressed", CompressedImage, queue_size=2)
    "/output/depth/image", Image, queue_size=2)

rospy.init_node('realsense', anonymous=True)
rate = rospy.Rate(30)

try:
    while not rospy.is_shutdown():
        # Wait for a coherent pair of frames: depth and color
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
        except RuntimeError as re:
            print(re)
            rate.sleep()
            cv2.waitKey(1)
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data()).astype(float)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #depth_colormap_dim = depth_colormap.shape
        #color_colormap_dim = color_image.shape

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "png"
        msg.data = np.array(cv2.imencode('.png', color_image)[1]).tobytes()
        # Publish new image
        image_pub.publish(msg)
        
        #### Create CompressedIamge ####
        #msg1 = CompressedImage()
        #msg1 = Image()
        #msg1.header.stamp = rospy.Time.now()
        #msg1.format = "png"
        #msg1.encoding = 'mono16'
        #msg1.data = np.array(cv2.imencode('.png', depth_image)[1]).tobytes()
        #msg1.data = depth_image.tobytes()
        #msg1 = bridge.cv2_to_compressed_imgmsg(depth_image, dst_format= "jp2")
        #msg1 = bridge.cv2_to_compressed_imgmsg(depth_image, dst_format= "png")
        #msg1 = bridge.cv2_to_compressed_imgmsg(depth_image, encoding= "passthrough")
        msg1 = bridge.cv2_to_imgmsg(depth_image, encoding= "passthrough")
        # Publish new image
        image_pub1.publish(msg1)

        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
