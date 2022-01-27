FROM anibali/pytorch:1.8.1-cuda11.1

RUN pip install \
        matplotlib==3.5.1 \
        imageio==2.13.5

WORKDIR /pytorchtutorial
