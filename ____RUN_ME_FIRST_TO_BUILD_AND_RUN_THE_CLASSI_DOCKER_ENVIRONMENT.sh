#! /bin/bash

set -x

clear
sudo docker stop              classi
sudo docker container rm      classi
sudo docker           rmi -f  classi
sudo DOCKER_BUILDKIT=1 docker build --progress=plain  -t classi .
clear
#sudo docker run -it --name classi --gpus device=0  --network=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --shm-size 2g   classi:latest
sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data  classi:latest
