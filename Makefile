
SHELL:=/bin/bash

BRANCH:=$(shell git branch --show-current)
WORKDIR_PATH=/pytorchtutorial
REPO_PATH:=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
IMAGE_TAG?=pvphan/pytorch-image:0.1
RUN_FLAGS = \
	--rm -it \
	--gpus=all \
	--ipc=host \
	--user="$(id -u):$(id -g)" \
	--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
	${IMAGE_TAG}

shell: image downloaddata
	docker run \
		${RUN_FLAGS} bash

image:
	docker build --tag ${IMAGE_TAG} .

downloaddata:
	./downloadmnist.sh

uploadimage: image
	docker push ${IMAGE_TAG}
