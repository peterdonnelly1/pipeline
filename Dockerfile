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
#    sudo docker run -v /home/peter/git/pipeline/working_data:/home/peter/git/pipeline/working_data -v /home/peter/git/pipeline/source_data:/home/peter/git/pipeline/source_data -it --shm-size 2g classi:latest bash
#


FROM python:3.7 as cache

#VERSION OF CUDA-TOOLKIT THAT I USED FOR DEVELOPING CLASSI
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb



FROM python:3.7

LABEL org.opencontainers.image.authors="pd@red.com.au"

#ENTRYPOINT []

#ENV PYTHONPATH="/usr/local/lib/python3.7"

RUN adduser --disabled-password --gecos 'default classi user' user_1

RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y git python3 python3-pip python3-numpy libvips openslide-tools wget git tree vim rsync


WORKDIR /home/peter/git

RUN mkdir pipeline

COPY requirements.txt   .
COPY requirements_1.txt .
COPY requirements_2.txt .
#COPY . .

RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install --upgrade numpy
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install -r requirements_2.txt
RUN --mount=type=cache,target=/root/.cache/pip python3.7 -m pip install -r requirements_1.txt


# VERSION OF THE CUDA-TOOLKIT THAT I USED FOR DEVELOPING CLASSI (11.2)
COPY --from=cache cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb  .
RUN dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
RUN apt-get update
#RUN apt-get install -y cuda


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
