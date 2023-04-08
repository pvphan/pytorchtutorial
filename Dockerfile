FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN pip install \
        matplotlib==3.5.1 \
        imageio==2.13.5

WORKDIR /pytorchtutorial
