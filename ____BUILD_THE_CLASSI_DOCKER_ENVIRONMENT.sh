#! /bin/bash

set -x

[ $(basename $(pwd)) != "pipeline" ] && mkdir -p pipeline
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
