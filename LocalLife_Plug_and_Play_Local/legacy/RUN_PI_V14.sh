#!/usr/bin/env bash
set -e

cd /home/locallife/LocalLife

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Helps Raspberry Pi installations where pyrealsense2/OpenCV were installed system-wide.
export PYTHONPATH="/usr/local/lib/python3.13/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"

python3 locallife_v14.py
