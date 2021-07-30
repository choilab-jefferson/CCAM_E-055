#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer, RosSubscriber
from sensor_msgs.msg import CompressedImage, PointCloud2

def main():
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    buffer_size = rospy.get_param("/TCP_BUFFER_SIZE", 1024)
    connections = rospy.get_param("/TCP_CONNECTIONS", 12)
    tcp_server = TcpServer(ros_node_name, buffer_size, connections)
    rospy.init_node(ros_node_name, anonymous=True)

    tcp_server.start({
        '/camera0/color/compressed': RosSubscriber('/camera0/color/compressed', CompressedImage, tcp_server),
        '/camera1/color/compressed': RosSubscriber('/camera1/color/compressed', CompressedImage, tcp_server),
        '/camera2/color/compressed': RosSubscriber('/camera2/color/compressed', CompressedImage, tcp_server),
        '/camera3/color/compressed': RosSubscriber('/camera3/color/compressed', CompressedImage, tcp_server),
        '/camera0/depth/compressed': RosSubscriber('/camera0/depth/compressed', CompressedImage, tcp_server),
        '/camera1/depth/compressed': RosSubscriber('/camera1/depth/compressed', CompressedImage, tcp_server),
        '/camera2/depth/compressed': RosSubscriber('/camera2/depth/compressed', CompressedImage, tcp_server),
        '/camera3/depth/compressed': RosSubscriber('/camera3/depth/compressed', CompressedImage, tcp_server),
        '/camera0/depth/points': RosSubscriber('/camera0/depth/points', PointCloud2, tcp_server),
        '/camera1/depth/points': RosSubscriber('/camera1/depth/points', PointCloud2, tcp_server),
        '/camera2/depth/points': RosSubscriber('/camera2/depth/points', PointCloud2, tcp_server),
        '/camera3/depth/points': RosSubscriber('/camera3/depth/points', PointCloud2, tcp_server),
    })

    rospy.spin()


if __name__ == "__main__":
    main()
