# Multi-view Video Streaming

## Containers
1. **ros-master**
   - run roscore
2. **realsense**
   - Dockerfile
   - ros_publish_rgbd: initialize four realsense cameras
3. **tcp_endpoint**
   - Dockerfile
   - wait-for-cameras.sh: waiting for the four cameras coming up and run tcp-endpoint
4. **model**
    1. **mmskl**
        - Dockerfile
        - ros_action_recognition.py: a ROS pipeline for action recogntion
    2. **mmaction2**
        - Dockerfile
        - ros_action_recognition_ma2.py: a ROS pipeline for action recogntion
5. ~~rosbridge~~
    - ~~Dockerfile~~


Wookjin Choi <wchoi@vsu.edu>