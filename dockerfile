FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Minimal setup
RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
SHELL [ "/bin/bash" , "-c" ]

RUN apt update \
 && apt install -y --no-install-recommends curl \
 && apt install -y --no-install-recommends gnupg2 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y net-tools gedit

RUN apt-get update

RUN apt-get install -y python3-pip 
RUN pip install pytest
RUN pip install numpy && pip install matplotlib && pip install scipy
RUN pip install torch
RUN pip install torchrl
RUN pip install tensordict
RUN pip install pettingzoo
RUN pip install gradysim
RUN pip install flask

RUN mkdir -p /SushiGo

WORKDIR /SushiGo


    


