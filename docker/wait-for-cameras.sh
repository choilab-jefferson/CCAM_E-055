#!/bin/bash
# wait-for-cameras.sh

set -e
for i in {0..3}
do
    sleep 5
    until rostopic info /camera$i/color/camera_info; do
        >&2 echo "camera$i is unavailable - sleeping"
        sleep 3
    done
done
  
>&2 echo "Cameras are up - executing command"
exec "$@"
