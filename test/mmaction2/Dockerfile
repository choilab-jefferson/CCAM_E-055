FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
RUN apt-get update && \ 
    apt-get -qy install libgl1-mesa-glx libglib2.0-0 git
RUN rm -rf /var/lib/apt/lists/*

RUN pip install scipy decord && \
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html && \
    git clone https://github.com/open-mmlab/mmaction2.git && \
    cd mmaction2 && pip install -r requirements/build.txt && pip install -v -e . && cd ..
ADD mmaction2.py ./
