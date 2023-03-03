
# START HERE 


** First, install the NVIDIA Container Runtime on your host machine !!! (instructions below) **

---
 To build the CLASSI docker image:

    sudo DOCKER_BUILDKIT=1 docker build --progress=plain  -t classi .

 To run a CLASSI experiment from within the docker image just built:

    from host console:
        sudo docker container rm -f classi
        sudo docker run  -it --name classi --env TZ=$(cat /etc/timezone) --gpus device=0 \
         --network=host --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix \
         -e DISPLAY=unix$DISPLAY classi:latest

    then, from within the classi docker container:
       cd pipeline
       ./do_all_RUN_ME_TO_SEE_RNASEQ_PROCESSING.sh                     or
       ./do_all_RUN_ME_TO_SEE_IMAGE_PROCESSING.sh                      or
       ./do_all_RUN_ME_TO_SEE_CLUSTERING_USING_SCIKIT_SPECTRAL.sh      or

    'gimp' (image viewer) and 'geany' (text editor) will be started automatically
    when the container runs

 To monitor experiment and see results:

    _during_ the experiment:
       monitor progress via container console output
       observe learning curves with any browser pointing to http://localhost:6006
       
    _after_ the experiment has completed:
       run 'gimp' inside the container to view images produced by classi. 
        eg. cd logs; gimp 230102_0247__01 ... bar_chart_AL.png &

 To edit configuration files:

    geany                   > /dev/null 2>&1 &
    geany do_all.sh         > /dev/null 2>&1 &
    geany conf/variables.sh > /dev/null 2>&1 &

 To enter running classi container with a bash shell

    sudo docker exec -it classi bash

 To stop/start classi container:

    sudo docker stop  classi
    sudo docker start classi

 To delete classi container:
    sudo docker rm classi

    this does not delete the classi image

 To delete the classi image:

    sudo docker rmi -f classi

    you will have to build it again if you do this. 
    Building should be fast since Docker will have cached everything that did not change since the last build.

---

    To install the NVIDIA Container Runtime on your host machine:

    If you use standard docker, CLASSI will only make use of CPUs; GPUs will be ignored. All 
    CLASSI capabilities except cuda_tsne (which requires a GPU) will work, albeit much slower
    For GPU support, the NVIDIA Container Runtime is required on the system running Docker
    The "FROM nvidia/cuda ..." directive below creates an image that supports GPUs, but you also 
    require NVIDIA Container Runtime
    Note: there is no  NVIDIA Container Runtime for Windows
    
    Installation instructions (4 steps) follow 
    From the NVIDIA installation guide https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
          sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
          https://#g' | \
          sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

---

 To run with datasets external to the container, in the default location (
 don't use: this is for me during development):

    sudo docker run -it --name classi --env TZ=$(cat /etc/timezone)  --gpus device=0  --network=host \
    --shm-size 2g -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
    -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data \
    -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data  classi:latest

---


I gratefuly acknowledge the authors of the following software which is used by CLASSI.  

**dpcca**  
The core learning engine of ‘CLASSI’ buids on Gregory Gundersen’s ‘dpcca'. 
      An attractive feature of this software is that it permits neural network models to be dynamically selected at run-time  
Code: https://github.com/gwgundersen/dpcca  
Paper: "End-to-end Training of Deep Probabilistic CCA on Paired Biomedical Observations"  
Paper: http://auai.org/uai2019/proceedings/papers/340.pdf  

**SPCN**  
GPU version of 'Structure Preserving Color Normalisation' by D. Anand, G. Ramakrishnan and A. Sethi  
Code:  https://github.com/goutham7r/spcn  
Paper: "Fast GPU-enabled Color Normalization for Digital pathology, International Conference on Systems, 
        Signals and Image Processing, Osijek, Croatia (2019), pp. 219-224  
Paper: https://ieeexplore.ieee.org/document/8787328/  

**Reinhard Stain Colour Normalisation:**  
Takumi Ando, University of Tokyo, Tokyo, Japan  
Code: https://github.com/tand826  
  
  
  
Each portion of the code is governed by ... respective licenses - however our code is governed by the ...

