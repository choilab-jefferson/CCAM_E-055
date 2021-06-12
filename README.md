# Multi-view Human Action Recognition

## Docker
[docker](docker)

```bash
cd docker
docker-compose up --build 
```

## Unity
[Unity](Unity)
- Unity project for a mixed reality environment


## Scripts
[scripts](scripts)

- Data preparation for mmpose and mmskeletion
    1. split original video to action videolets
    ```bash
    python scripts/split_video.py
    ```

    2. generates json for mmskeleton
    ```bash
    # this will generate model/florence_ccam.json
    python scripts/json_generation.py

    ```

## Action Recognition Model
[model](model)
1. Build a docker image for mmskeleton
    ```bash
    cd model
    docker build -t mmskl .
    ```
2. Run the docker container
    ```bash
    docker run --gpus all -it -v $HOME:/root mmskl
    # TODO: add volume for the project repository and data
    ```
3. Generate skeleton dataset
    ```bash
    # in the docker container
    cd /root/data/CCAM_E-055/model # TODO: change the path
    mmskl build_dataset_florence.yaml
    mmskl build_dataset_ccam.yaml
    # './data/dataset_{dataset name}/'
    ```
4. Train models
    ```bash
    # the same path in the docker container above
    mmskl train.xml
    mmskl train_florence.xml
    mmskl train_ccam.xml
    # './workdir/dataset_{dataset name}/'
    ```
5. Test models
    ```bash
    # the same path in the docker container above
    mmskl test.xml
    mmskl test_florence.xml
    mmskl test_ccam.xml
    ```


## Test codes
[test](test)

- test/realsense_ros.py: cv_bridge interface 
- test/Unity_ROS: For ROS TCP connection test


Wookjin Choi <wchoi@vsu.edu>
