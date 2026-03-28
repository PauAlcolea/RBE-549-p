#!/bin/bash

pip install -r requirements.txt
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip cache purge
python -m pip install --no-cache-dir opencv-python-headless