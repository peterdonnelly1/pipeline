#! /bin/bash

set -x

sudo docker stop              classi
sudo docker container rm      classi
sudo docker           rmi -f  classi
#sudo DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain  -t classi .
sudo DOCKER_BUILDKIT=1 docker build            --progress=plain  -t classi .

[ $(basename $(pwd)) != "pipeline" ] && mkdir -p pipeline
cd pipeline 2>/dev/null || true
mkdir -p working_data
mkdir -p source_data
mkdir -p logs
mkdir -p classi/runs

sudo xhost +local:docker
sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 \
    -v $(pwd)/working_data:/pipeline/working_data \
    -v $(pwd)/source_data:/pipeline/source_data  \
    -v $(pwd)/logs:/pipeline/logs  \
    -v $(pwd)/classi/runs:/pipeline/classi/runs  \
    classi:latest

set +x
