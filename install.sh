#!/bin/bash

uv venv --python 3.12 venv
source venv/bin/activate
uv pip install numpy==1.26.4
uv pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu128
deactivate
