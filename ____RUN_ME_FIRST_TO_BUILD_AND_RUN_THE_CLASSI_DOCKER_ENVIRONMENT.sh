#! /bin/bash

set -x

COMMIT_HASH=$(git ls-remote https://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@github.com/peterdonnelly1/pipeline --hash HEAD | awk '{print $1}')
sudo DOCKER_BUILDKIT=1 docker build --build-arg HASH=$COMMIT_HASH --progress=plain  -t classi .
docker run -it --name classi --gpus device=0  --network=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --shm-size 2g   classi:latest
