#!/bin/bash

docker run \
    -it \
    --rm \
    -p 7777:8888 \
    -v /home:/home \
    --gpus all \
    --ipc="host" \
    --name openpose \
    pmorris2012/openpose
#   /bin/bash