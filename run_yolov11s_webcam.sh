#!/bin/bash

# YOLOV11s Webcam Detection Launcher Script
# This script activates the virtual environment and runs the YOLOV11s detection

echo "Starting YOLOV11s Webcam Detection..."

# Check if virtual environment exists
if [ ! -d "degirum_env" ]; then
    echo "Error: degirum_env virtual environment not found!"
    echo "Please make sure you have created the virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source degirum_env/bin/activate

# Check if the model exists
MODEL_PATH="models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please make sure your YOLOV11s model is in the models directory."
    exit 1
fi

# Check if webcam is available
if ! command -v v4l2-ctl &> /dev/null; then
    echo "Warning: v4l2-ctl not found. Cannot check webcam availability."
else
    echo "Checking webcam availability..."
    if ! v4l2-ctl --list-devices | grep -q "Camera\|Webcam\|USB"; then
        echo "Warning: No webcam detected. Make sure your webcam is connected."
    fi
fi

echo "Running YOLOV11s detection..."
echo "Press 'q' to quit, 's' to save screenshot, 'h' for help"
echo ""

# Run the detection script
python webcam_yolov11s_detection.py --model-path "$MODEL_PATH" "$@"

# Deactivate virtual environment
deactivate

echo "Detection finished." 