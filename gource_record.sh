#!/bin/bash

gource \
    -c 3.0 \
    -1280x720 \
    --auto-skip-seconds .1 \
    --multi-sampling \
    --stop-at-end \
    --key \
    --highlight-users \
    --hide mouse,filenames \
    --file-idle-time 0 \
    --max-files 0  \
    --background-colour 000000 \
    --font-size 25 \
    --output-ppm-stream - \
    --output-framerate 30 \
    | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - -b 65536K movie.mp4

ffmpeg -i movie.mp4 -vcodec libx265 -crf 28 movie-out.mp4
rm movie.mp4