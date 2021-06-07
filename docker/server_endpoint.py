#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer, RosPublisher, RosSubscriber, RosService, UnityService
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2

def main():
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    buffer_size = rospy.get_param("/TCP_BUFFER_SIZE", 1024)
    connections = rospy.get_param("/TCP_CONNECTIONS", 10)
    tcp_server = TcpServer(ros_node_name, buffer_size, connections)
    rospy.init_node(ros_node_name, anonymous=True)

    tcp_server.start({
        '/camera1/color/image_raw/compressed': RosSubscriber('/camera1/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera2/color/image_raw/compressed': RosSubscriber('/camera2/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera3/color/image_raw/compressed': RosSubscriber('/camera3/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera4/color/image_raw/compressed': RosSubscriber('/camera4/color/image_raw/compressed', CompressedImage, tcp_server),
        '/camera1/depth/color/points': RosSubscriber('/camera1/depth/color/points', PointCloud2, tcp_server),
        '/camera2/depth/color/points': RosSubscriber('/camera2/depth/color/points', PointCloud2, tcp_server),
        '/camera3/depth/color/points': RosSubscriber('/camera3/depth/color/points', PointCloud2, tcp_server),
        '/camera4/depth/color/points': RosSubscriber('/camera4/depth/color/points', PointCloud2, tcp_server),
    })

    rospy.spin()


if __name__ == "__main__":
    main()
