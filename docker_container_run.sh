#!/bin/bash

docker run \
    -it \
    --rm \
    -v /home:/home \
    --gpus all \
    --ipc="host" \
    pmorris2012/openpose