FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
RUN apt-get update && \ 
    apt-get -qy install libgl1-mesa-glx libglib2.0-0 git
RUN rm -rf /var/lib/apt/lists/*

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

