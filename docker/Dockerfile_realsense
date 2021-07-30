FROM ros:melodic-ros-core
RUN apt-get update && \ 
    apt-get -qy install ros-melodic-compressed-image-transport ros-melodic-compressed-depth-image-transport python3-pip
RUN pip3 install --upgrade pip && pip3 install pyrealsense2 numpy opencv-python rospkg open3d
RUN rm -rf /var/lib/apt/lists/*
ADD ./ros_publish_rgbd.py ./
