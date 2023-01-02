######################################################################################################################################################################################################################################################################################################################################
#
# To build docker image:
#
#    sudo DOCKER_BUILDKIT=1 docker build --progress=plain  -t classi .
#
# To run experiment:
#
#    from a host console:
#        sudo docker run -it -p 6006:6006 --name classi --gpus device=0  --shm-size 2g   classi:latest
#    then in the classi container:
#       tensorboard --logdir=/home/peter/git/pipeline/classi/runs --samples_per_plugin images=0 --reload_interval=1 --bind_all &
#       cd pipeline
#       ./do_all_RUN_ME_TO_SEE_RNASEQ_PROCESSING.sh or ./do_all_RUN_ME_TO_SEE_IMAGE_PROCESSING.sh or ./do_all_RUN_ME_TO_SEE_CLUSTERING_1.sh
#
# To monitor experiment and see results:
#
#    during the experiment
#       monitor progress via container console output
#       observe learning curves with browser pointing to http://localhost:6006
#    after the experiment has comleted, run 'gimp' inside the container to view images produced by classi. eg. cd logs; gimp 230102_0247__01 ... bar_chart_AL.png
#
# To run with datasets external to the container, in the default location (don't use: this is for me during development):
#
#    sudo docker run -it -p 6006:6006 --name classi                  -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data  --shm-size 2g   classi:latest
#    sudo docker run -it -p 6006:6006 --name classi --gpus device=0  -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data  --shm-size 2g   classi:latest
#
# To enter the running classi container with a bash shell
#
#    sudo docker exec -it classi bash
#
# To stop/start the classi container:
#
#    sudo docker stop  classi
#    sudo docker start classi
#
# To delete the running container:
#  
#    sudo docker rm classi
#
#    this does not delete the classi image
#
# To delete the classi image:
#
#    sudo docker rmi -f classi
#
#    you will have to build it again if you do this. Building should be very fast since Docker will have cached everything that did not change since the last build.
#
######################################################################################################################################################################################################################################################################################################################################
#
# *** To install the NVIDIA Container Runtime on your host machine (highly recommended):
#
#   If you use standard docker, CLASSI will only make use of CPUs; GPUs will be ignored. All CLASSI capabilities except cuda_tsne (which is expicitly designed to use a GPU) will work, albeit a lot slower.
#   For GPU support, the NVIDIA Container Runtime is required on the system running Docker. The "FROM nvidia/cuda ... directive below creates an image that supports GPUs, but you _also_ require the NVIDIA Container Runtime. 
#   Installation instructions follow (these are from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Note: there is no  NVIDIA Container Runtime for Windows.
#
#   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
#            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# sudo apt-get update
# sudo apt-get install -y nvidia-docker2
# sudo systemctl restart docker
#
######################################################################################################################################################################################################################################################################################################################################


FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
#FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.0-base-ubuntu20.04

# CuPy, which is used in cuda_tsne, expects there to be a link called libcuda.so.1 to the cuda drivers. For some reason it is not put in place (Docker or not) so have to manually create one
RUN ln -s /usr/local/cuda-11.2/compat/libcuda.so.460.106.00  /usr/lib/x86_64-linux-gnu/libcuda.so.1 

ENV DEBIAN_FRONTEND=noninteractve

LABEL org.opencontainers.image.authors="pd@red.com.au"

####  ENV PYTHONPATH="/usr/local/lib/python3.7"

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN adduser --disabled-password --gecos 'default classi user' user_1

RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y git python3 python3-pip python3-numpy libvips openslide-tools wget git tree vim rsync libsm6 libxext6 mlocate gimp firefox net-tools

WORKDIR /home/peter/git

RUN mkdir pipeline

COPY requirements.txt   .
COPY requirements_1.txt .
COPY requirements_2.txt .

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade numpy
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements_2.txt
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements_1.txt

RUN   pip uninstall -y hdbscan
RUN   pip   install hdbscan==0.8.29
RUN   pip   install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html

RUN git clone --depth 1 --branch master https://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@github.com/peterdonnelly1/pipeline

#CMD ["/bin/bash", "./do_all.sh"]
