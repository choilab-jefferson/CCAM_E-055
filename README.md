# Multi-view Human Action Recognition

## Docker
docker

```bash
cd docker
docker-compose up --build 
```

## Unity
Unity


## Scripts
scripts

Data preparation for mmpose and mmskeletion
```bash
# split original video to action videolets
python scripts/split_video.py

# generates json for mmskeleton
python scripts/json_generation.py

```


## Test codes
- test/realsense_ros.py: cv_bridge interface 
- test/Unity_ROS: For ROS TCP connection test

Wookjin Choi <wchoi@vsu.edu>
