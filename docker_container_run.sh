#!/bin/bash

docker run \
    -it \
    --rm \
    -v /media/mpcrpaul/data/HMDB51/dataset:/Videos \
    -v /media/mpcrpaul/data/HMDB51/Videos_Pose:/Videos_Pose \
    -v /home:/home \
    --gpus all \
    --ipc="host" \
    pmorris2012/openpose \
    python3 /home/mpcrpaul/Documents/school/openpose/examples/python/process_folder.py --face --hand --draw_pose --draw_black_pose