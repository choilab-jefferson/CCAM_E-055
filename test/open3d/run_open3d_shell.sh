#!/bin/bash
docker build -t open3d .
docker run -it -v $PWD:/workspace --privileged --volume /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY open3d
