#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer, RosSubscriber
from sensor_msgs.msg import CompressedImage
#from sensor_msgs.msg import Image
#from sensor_msgs.msg import PointCloud2

def main():
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    buffer_size = rospy.get_param("/TCP_BUFFER_SIZE", 1024)
    connections = rospy.get_param("/TCP_CONNECTIONS", 10)
    tcp_server = TcpServer(ros_node_name, buffer_size, connections)
    rospy.init_node(ros_node_name, anonymous=True)

    tcp_server.start({
        '/camera1/color': RosSubscriber('/camera1/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera2/color': RosSubscriber('/camera2/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera3/color': RosSubscriber('/camera3/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera4/color': RosSubscriber('/camera4/color/image_raw/compressed', CompressedImage, tcp_server),
        # '/camera1/depth': RosSubscriber('/camera1/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage, tcp_server),
        # '/camera2/depth': RosSubscriber('/camera2/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage, tcp_server),
        # '/camera3/depth': RosSubscriber('/camera3/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage, tcp_server),
        # '/camera4/depth': RosSubscriber('/camera4/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage, tcp_server),
        '/camera1/depth1': RosSubscriber('/camera1/aligned_depth_to_color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera2/depth1': RosSubscriber('/camera2/aligned_depth_to_color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera3/depth1': RosSubscriber('/camera3/aligned_depth_to_color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera4/depth1': RosSubscriber('/camera4/aligned_depth_to_color/image_raw/compressed', CompressedImage, tcp_server),
        # '/camera1/color': RosSubscriber('/camera1/color/image_raw', Image, tcp_server),
        # '/camera2/color': RosSubscriber('/camera2/color/image_raw', Image, tcp_server),
        # '/camera3/color': RosSubscriber('/camera3/color/image_raw', Image, tcp_server),
        # '/camera4/color': RosSubscriber('/camera4/color/image_raw', Image, tcp_server),
        # '/camera1/depth': RosSubscriber('/camera1/aligned_depth_to_color/image_raw', Image, tcp_server),
        # '/camera2/depth': RosSubscriber('/camera2/aligned_depth_to_color/image_raw', Image, tcp_server),
        # '/camera3/depth': RosSubscriber('/camera3/aligned_depth_to_color/image_raw', Image, tcp_server),
        # '/camera4/depth': RosSubscriber('/camera4/aligned_depth_to_color/image_raw', Image, tcp_server),
        # '/camera1/depth/color/points': RosSubscriber('/camera1/depth/color/points', PointCloud2, tcp_server),
        # '/camera2/depth/color/points': RosSubscriber('/camera2/depth/color/points', PointCloud2, tcp_server),
        # '/camera3/depth/color/points': RosSubscriber('/camera3/depth/color/points', PointCloud2, tcp_server),
        # '/camera4/depth/color/points': RosSubscriber('/camera4/depth/color/points', PointCloud2, tcp_server),
    })

    rospy.spin()


if __name__ == "__main__":
    main()
