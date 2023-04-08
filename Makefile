
SHELL:=/bin/bash

WORKDIR_PATH=/pytorchtutorial
REPO_PATH:=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
IMAGE_TAG?=pvphan/pytorch-image:0.1

ifneq ($(shell lshw -C display 2> /dev/null | grep NVIDIA | wc -l), 0)
	GPU_FLAG:=--gpus=all
endif

RUN_FLAGS = \
	--rm -it \
	${GPU_FLAG} \
	--ipc=host \
	--user="$(id -u):$(id -g)" \
	--volume=${REPO_PATH}:${WORKDIR_PATH}:ro \
	--volume=/tmp/pytorchtutorial/output:/tmp/output \
	--volume=/tmp/pytorchtutorial/data:/tmp/data \
	${IMAGE_TAG}

runmnist: image downloaddata
	docker run ${RUN_FLAGS} python mnist.py

runmnistlenet5: image downloaddata
	docker run ${RUN_FLAGS} python mnistlenet5.py

shell: image downloaddata
	docker run ${RUN_FLAGS} bash

image:
	docker build --tag ${IMAGE_TAG} .

downloaddata:
	./downloadmnist.sh

uploadimage: image
	docker push ${IMAGE_TAG}
