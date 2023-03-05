#! /bin/bash

set -x

sudo xhost +local:docker
sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 \
    -v $(pwd)/working_data:/pipeline/working_data \
    -v $(pwd)/source_data:/pipeline/source_data  \
    -v $(pwd)/logs:/pipeline/logs  \
    -v $(pwd)/classi/runs:/pipeline/classi/runs  \
    classi:latest

set +x
