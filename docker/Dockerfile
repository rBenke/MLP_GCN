FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y vim
RUN apt-get install -y python3-pip
 
RUN pip3 install ogb==1.3.3
RUN pip3 install torch==1.12.0
RUN pip3 install torch-geometric==2.0.4
RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+11.6.html

WORKDIR /project

ENTRYPOINT /bin/bash
