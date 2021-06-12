# Multi-view Video Streaming

## Containers
1. **ros-master**
    - run roscore
2. **realsense**
    - Dockerfile_realsense
    - run-cameras.sh: initialize four realsense cameras
3. **tcp-endpoint**
    - Dockerfile_TCPEndpoint
    - wait-for-cameras: waiting for the four cameras coming up and run tcp-endpoint
4. ~~rosbridge~~
    - ~~Dockerfile_realsense~~

Wookjin Choi <wchoi@vsu.edu>