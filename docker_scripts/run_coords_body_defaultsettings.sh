#!/bin/bash

docker run \
    -it \
    --rm \
    -v /media/mpcrpaul/data/HMDB51/dataset:/Input \
    -v /media/mpcrpaul/data/HMDB51/Videos_Pose:/Output \
    -v $(dirname $(pwd))/examples/python:/code \
    --gpus all \
    --ipc="host" \
    pmorris2012/openpose \
    python3 /code/process_folder.py