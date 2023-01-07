#! /bin/bash

set -x

sudo DOCKER_BUILDKIT=1 docker build --progress=plain  -t classi .
sudo docker run -it --name classi --gpus device=0  --network=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --shm-size 2g   classi:latest
