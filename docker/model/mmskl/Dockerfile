FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

RUN apt-get update && \ 
    apt-get -qy install libgl1-mesa-glx libglib2.0-0 git ros-melodic-rospy ros-melodic-compressed-image-transport && \
    rm -rf /var/lib/apt/lists/*
RUN pip install rospkg

RUN pip install pycocotools lazy_import mmdet==2.3.0 mmpose && \
    pip install mmcv-full==1.0.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html && \
    git clone https://github.com/open-mmlab/mmpose.git -b v0.10.0 && \
    git clone https://github.com/open-mmlab/mmdetection.git -b v2.3.0

RUN git clone https://github.com/taznux/mmskeleton.git && \
    cd mmskeleton && \
    python setup.py develop && \
    cd mmskeleton/ops/nms/ && \
    python setup_linux.py develop && \
    cd ../../..
RUN mkdir ./config
ADD ./action_recognition.yaml ./action_recognition.py ./ros_action_recognition.py ./
