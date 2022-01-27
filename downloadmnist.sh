#!/bin/bash

fileNames="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

mkdir -p ./data
dataDirSize=$(ls ./data | wc -l)
if [[ "$dataDirSize" -eq 0 ]]; then
    for fileName in $fileNames
    do
        echo $fileName
        wget -O ./data/$fileName http://yann.lecun.com/exdb/mnist/$fileName
        gunzip ./data/$fileName
    done
fi

