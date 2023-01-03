#!/bin/bash

######################################################################################################################################################################################################################################################################################################################################
#
#   Install the NVIDIA Container Runtime on your host machine (highly recommended):
#
#   If you use standard docker, CLASSI will only make use of CPUs; GPUs will be ignored. All CLASSI capabilities except cuda_tsne (which is expicitly designed to use a GPU) will work, albeit a lot slower.
#   For GPU support, the NVIDIA Container Runtime is required on the system running Docker. The "FROM nvidia/cuda ... directive below creates an image that supports GPUs, but you _also_ require the NVIDIA Container Runtime. 
#   Installation instructions follow (these are from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Note: there is no  NVIDIA Container Runtime for Windows.
#

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
           sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
           sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
