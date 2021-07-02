FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
RUN apt-get update && \ 
    apt-get -qy install libgl1-mesa-glx libglib2.0-0 libnvidia-gl-460 libusb-1.0-0
RUN rm -rf /var/lib/apt/lists/*

RUN pip install open3d pyrealsense2 opencv-python
