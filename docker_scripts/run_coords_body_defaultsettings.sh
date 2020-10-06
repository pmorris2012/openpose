#!/bin/bash

docker run \
    -it \
    --rm \
    -v /media/mpcrpaul/data/HMDB51/dataset:/Input \
    -v /media/mpcrpaul/data/HMDB51/Videos_Pose:/Output \
    -v /home/mpcrpaul/Documents/school/openpose/examples/python:/code \
    --gpus all \
    --ipc="host" \
    pmorris2012/openpose \
    python3 /code/process_folder.py