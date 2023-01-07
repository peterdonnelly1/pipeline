#! /bin/bash

set -x

clear
sudo docker container stop
sudo docker container rm      classi
sudo docker images    rmi -f  classi
sudo DOCKER_BUILDKIT=1 docker build --progress=plain  -t classi .
clear
#sudo docker run -it --name classi --gpus device=0  --network=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --shm-size 2g   classi:latest
sudo docker run -it --name classi --gpus device=0  --network=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --shm-size 2g -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data   classi:latest
