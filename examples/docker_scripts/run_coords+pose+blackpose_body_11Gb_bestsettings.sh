#!/bin/bash

docker run \
    -it \
    --rm \
    -v /media/mpcrpaul/data/HMDB51/dataset:/Input \
    -v /media/mpcrpaul/data/HMDB51/Videos_Pose:/Output \
    --gpus all \
    --ipc="host" \
    pmorris2012/openpose \
    python3 /openpose/examples/python/process_folder.py \
    --draw_pose --draw_black_pose --scale_number 4