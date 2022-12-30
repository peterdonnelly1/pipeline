# To build:
#
#   sudo DOCKER_BUILDKIT=1 docker build -t classi .
#
# To run:                                                     
#
#    sudo docker run -it --shm-size 2g classi:latest bash
#
# To run using datasets outside the containter:
#
#    sudo docker run --gpus device=0 -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data -it --shm-size 2g classi:latest bash
#


#FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractve

LABEL org.opencontainers.image.authors="pd@red.com.au"

ENV PYTHONPATH="/usr/local/lib/python3.7"

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


RUN git clone --depth 1 https://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@github.com/peterdonnelly1/pipeline

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
