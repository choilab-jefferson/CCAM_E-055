#!/bin/bash
docker build -t mmaction .
docker run -it -v $HOME:/root -v $HOME/data/dataset/action/ccam_actions:/ccam_actions mmaction2 bash -c "cd mmaction2 && python ../mmaction2.py"
