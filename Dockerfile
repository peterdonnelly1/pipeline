# To build:
#
#    sudo DOCKER_BUILDKIT=1 docker build -t classi .
#
# To run:                                                     
#
#    sudo docker run                 -it --shm-size 2g classi:latest bash 
#    sudo docker run --gpus device=0 -it --shm-size 2g classi:latest bash  << only use if you have installed the NVIDIA Container Runtime (see note below)
#
# To run with datasets that external to the container, but in the default location (don't use: this is for me):
#
#    sudo docker run                 -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data -it --shm-size 2g classi:latest bash
#    sudo docker run --gpus device=0 -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data -it --shm-size 2g classi:latest bash
#
# To install the NVIDIA Container Runtime (highly recommended):
#
#   If you use standard docker, CLASSI will only make use of CPUs; GPUs will be ignored. All CLASSI capabilities except cuda_tsne (which is expicitly designed to use a GPU) will work, albeit a lot slower.
#   For GPU support, the NVIDIA Container Runtime is required on the system running Docker. The base image (FROM nvidia/cuda ...) creates an image that supports GPUs, but you also require NVIDIA Container Runtime. 
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
#

FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
#FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.0-base-ubuntu20.04

# CuPy, which is used in cuda_tsne, expects there to be a link called libcuda.so.1 to the cuda drivers. For some reason it is not put in place (Docker or not) so have to manually create one
ln -s /usr/local/cuda-11.2/compat/libcuda.so.460.106.00  /usr/lib/x86_64-linux-gnu/libcuda.so.1 

ENV DEBIAN_FRONTEND=noninteractve

LABEL org.opencontainers.image.authors="pd@red.com.au"

####ENV PYTHONPATH="/usr/local/lib/python3.7"

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN adduser --disabled-password --gecos 'default classi user' user_1

RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y git python3 python3-pip python3-numpy libvips openslide-tools wget git tree vim rsync libsm6 libxext6


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

RUN   pip uninstall hdbscan
RUN   pip   install hdbscan==0.8.27


RUN git clone --depth 1 --branch master https://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@github.com/peterdonnelly1/pipeline

#RUN	git clone https://github.com/SBU-BMI/quip_classification && \
#	cd /root/quip_classification/u24_lymphocyte/prediction/NNFramework_TF_models && \
#	wget -v -O models.zip -L \
#	  https://stonybrookmedicine.box.com/shared/static/bl15zu4lwb9cc7ltul15aa8kyrn7kh2d.zip >/dev/null 2>&1 && \
#        unzip -o models.zip && rm -f models.zip && \
#	chmod 0755 /root/quip_classification/u24_lymphocyte/scripts/*

#ENV	BASE_DIR="/root/"
#ENV	PATH="./":$PATH
#ENV PYTHONPATH "${PYTHONPATH}:/your/custom/path"
#ENV PYTHONPATH "${PYTHONPATH}
#ENV PATH /usr/local/bin:$PATH

CMD ["/bin/bash", "./do_all.sh"]
