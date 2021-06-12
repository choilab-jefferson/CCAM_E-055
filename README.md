# Multi-view Human Action Recognition

## [Docker](docker)
**Docker Compose for multi-view video streaming in ROS**

```bash
cd docker
docker-compose up --build 
```

## [Unity](Unity)
**Unity project for a mixed reality environment**


## [Scripts](scripts)
**Data preparation for mmpose and mmskeletion**

1. split original video to action videolets
    ```bash
    python scripts/split_video.py
    ```

2. generates json for mmskeleton
    ```bash
    # this will generate model/florence_ccam.json
    python scripts/json_generation.py

    ```

## [Action Recognition Model](model)
**Deep learning models for action recognition**


## [Test codes](test)

Wookjin Choi <wchoi@vsu.edu>
