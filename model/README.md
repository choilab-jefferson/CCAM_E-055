# Human Action Recognition using mmskeleton
- Pose estimation: R-CNN + HRNet-32
- Action recognition: Spatio-Tempoal Graph Convolutional Network (ST-GCN)

## Docker
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

## Video to Skeleton
3. Generate skeleton dataset
    ```bash
    # in the docker container
    cd /root/data/CCAM_E-055/model # TODO: change the path
    mmskl build_dataset_florence.yaml
    mmskl build_dataset_ccam.yaml
    # './data/dataset_{dataset name}/'
    ```

## Train and Test Models
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

Wookjin Choi <wchoi@vsu.edu>