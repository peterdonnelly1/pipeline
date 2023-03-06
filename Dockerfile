
FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
  
# CuPy, which is used in cuda_tsne, expects there to be a link called libcuda.so.1 to the cuda drivers. For some reason it is not put in place (Docker or not) so have to manually create one
RUN ln -s /usr/local/cuda-11.2/compat/libcuda.so.460.106.00  /usr/lib/x86_64-linux-gnu/libcuda.so.1 

ENV DEBIAN_FRONTEND=noninteractve

LABEL org.opencontainers.image.authors="pd@red.com.au"

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN \
    --mount=type=cache,target=/var/cache/apt \
      apt-get update && apt-get install -y git python3 python3-pip python3-numpy libvips openslide-tools wget git tree vim rsync libsm6 libxext6 mlocate gimp firefox python3-tk geany

WORKDIR /

COPY Dockerfile_pip_requirements.txt   .                                               

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade numpy
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r Dockerfile_pip_requirements.txt

#RUN   pip uninstall -y hdbscan
#RUN   pip   install hdbscan==0.8.29
RUN   pip   install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html

RUN git clone --depth 13 https://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@github.com/peterdonnelly1/pipeline


# START UP SHENANIGANS
RUN echo '#!/bin/bash\ncd pipeline\ngimp > /dev/null 2>&1 &\ngeany > /dev/null 2>&1 &\nnohup tensorboard --logdir=classi/runs --samples_per_plugin images=0 --reload_interval=1 --bind_all &' > start.sh
RUN chmod +x start.sh
RUN echo "alias cls='printf \"\033c\"'" >> /root/.bashrc
CMD ["/bin/bash", "-c", "source start.sh && source /root/.bashrc && bash"]
