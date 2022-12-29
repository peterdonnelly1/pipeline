
FROM python:3.7 as cache

#VERSION OF CUDA-TOOLKIT THAT I USED FOR DEVELOPING CLASSI
RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb   -O cuda.deb



FROM python:3.7

LABEL org.opencontainers.image.authors="pd@red.com.au"

#ENTRYPOINT []

#ENV PYTHONPATH="/usr/local/lib/python3.7"

RUN adduser --disabled-password --gecos 'default classi user' user_1

RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y git python3 python3-pip python3-numpy libvips openslide-tools wget git tree vim 


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

 


#VERSION OF CUDA-TOOLKIT THAT I USED FOR DEVELOPING CLASSI
COPY --from=cache cuda.deb   /home/peter/git
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
run dpkg -i cuda.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
RUN apt-get install -y /home/peter/git/cuda.deb


RUN git clone --depth 2 https://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@github.com/peterdonnelly1/pipeline

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
