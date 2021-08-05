#!/bin/bash
# wait-for-cameras.sh

set -e

check_success() {
    [[ "${*}" =~ 0 ]]
    return
}

cam_on=(1 1 1 1)
until check_success ${cam_on[*]}
do
    for i in "${!cam_on[@]}"
    do
        if [ ${cam_on[i]} -ne 0 ]
        then
            >&2 echo "Check camera $i"
            rostopic info /camera$i/color/camera_info
            cam_on[$i]=$?
        else
            >&2 echo "Camera $i is up"
        fi
    done
done
  
>&2 echo "Cameras are up - executing command"
exec "$@"
