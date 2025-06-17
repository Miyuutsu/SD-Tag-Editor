#!/bin/bash

git pull
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Installation not found. Installing..."
    chmod +x install.sh
    ./install.sh no_pause
    read -p "Press enter to continue..."
    deactivate
    exit 0
fi

uv pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu128
read -p "Press enter to continue..."
deactivate
exit 0
