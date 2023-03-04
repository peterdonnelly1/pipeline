#! /bin/bash

set -x

mkdir -p pipline
cd pipeline 2>/dev/null || true
mkdir -p working_data
mkdir -p source_data
mkdir -p logs
mkdir -p classi/runs


sudo docker stop              classi
sudo docker container rm      classi
sudo docker           rmi -f  classi
#sudo DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain  -t classi .
sudo DOCKER_BUILDKIT=1 docker build            --progress=plain  -t classi .
clear
sudo xhost +local:docker
#sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1  classi:latest
#~ sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data  classi:latest
sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 \
    -v $(pwd)/working_data:/pipeline/working_data \
    -v $(pwd)/source_data:/pipeline/source_data  \
    -v $(pwd)/logs:/pipeline/logs  \
    -v $(pwd)/classi/runs:/pipeline/classi/runs  \
    classi:latest
