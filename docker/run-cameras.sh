#!/bin/bash
# run cameras.sh

set -e

RS_SERIAL_NO=("036322250763" "038122250356" "f0245826" "f0245993")

# get length of an array
arraylength=${#RS_SERIAL_NO[@]}

# use for loop to read all values and indexes
for (( i=1; i<=${arraylength}; i++ ));
do
    while : ; do
        roslaunch realsense2_camera rs_camera.launch \
            camera:=camera$i serial_no:=${RS_SERIAL_NO[$i - 1]} \
            enable_depth:=True depth_width:=640 depth_height:=480 depth_fps:=15 \
            color_width:=640 color_height:=480 color_fps:=15 align_depth:=True > /dev/null &
        sleep 3
        rostopic info /camera$i/color/camera_info && break
    done
done

wait
