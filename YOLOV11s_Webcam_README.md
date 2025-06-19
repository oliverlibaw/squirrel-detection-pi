# YOLOV11s Webcam Object Detection on Raspberry Pi

This project provides a real-time object detection system using a custom YOLOV11s model on a Raspberry Pi with Hailo AI accelerator HAT and webcam input.

## Features

- **Real-time Detection**: Live object detection from webcam feed
- **Bounding Box Visualization**: Draws boxes around detected objects
- **Confidence Scores**: Displays confidence scores for each detection
- **FPS Monitoring**: Real-time FPS display
- **Screenshot Capture**: Save detection results as images
- **Configurable Parameters**: Adjustable confidence threshold, camera settings, etc.
- **Multiple Modes**: GUI mode with live display or headless mode with frame saving

## Prerequisites

- Raspberry Pi with Hailo AI accelerator HAT
- DeGirum SDK installed
- Virtual environment (`degirum_env`) with required packages
- USB webcam or Pi Camera
- YOLOV11s model converted to DeGirum format

## Installation

1. **Activate the virtual environment:**
   ```bash
   source degirum_env/bin/activate
   ```

2. **Install required packages (if not already installed):**
   ```bash
   pip install opencv-python numpy
   ```

3. **Make sure your model is in the correct location:**
   ```
   models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1/
   ├── labels_squirrel_yolov11s.json
   ├── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef
   └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json
   ```

## Usage

### Option 1: GUI Mode (with live display)

**Use when you have a desktop environment or X11 forwarding available.**

#### Quick Start
```bash
./run_yolov11s_webcam.sh
```

#### Direct Python Execution
```bash
# Activate virtual environment
source degirum_env/bin/activate

# Run with default settings
python webcam_yolov11s_detection.py

# Run with custom parameters
python webcam_yolov11s_detection.py \
    --model-path models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1 \
    --camera 0 \
    --confidence 0.6 \
    --width 640 \
    --height 480
```

#### Controls (GUI Mode)
- **`q`** - Quit the application
- **`s`** - Save a screenshot (saved as `yolov11s_screenshot_X.jpg`)
- **`h`** - Show help information

### Option 2: Headless Mode (save frames to disk)

**Use when running over SSH or in environments without display support.**

#### Quick Start
```bash
./run_yolov11s_headless.sh
```

#### Direct Python Execution
```bash
# Activate virtual environment
source degirum_env/bin/activate

# Run with default settings
python webcam_yolov11s_detection_headless.py

# Run with custom parameters
python webcam_yolov11s_detection_headless.py \
    --model-path models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1 \
    --camera 0 \
    --confidence 0.6 \
    --width 640 \
    --height 480 \
    --save-interval 30
```

#### Headless Mode Features
- Saves annotated frames to `detection_output/` directory
- Configurable save interval (default: every 30 frames ≈ 1 frame per second)
- Real-time console output with FPS and detection counts
- No display requirements - works over SSH

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1` | Path to model directory |
| `--camera` | `0` | Camera index (0 for default webcam) |
| `--confidence` | `0.5` | Confidence threshold (0.0-1.0) |
| `--width` | `640` | Camera width |
| `--height` | `480` | Camera height |
| `--device` | `HAILORT/HAILO8L` | Hailo device type |
| `--save-interval` | `30` | Save frame every N frames (headless mode only) |

## Model Information

The script is configured for a YOLOV11s model trained to detect:
- **Class**: Squirrel
- **Input Resolution**: 640x640
- **Device**: Hailo8L/Hailo8
- **Format**: Quantized HailoRT

## Performance Tips

1. **Camera Settings**: The script optimizes camera settings for low latency
2. **Resolution**: Lower resolutions may improve FPS
3. **Confidence Threshold**: Higher thresholds reduce false positives but may miss detections
4. **Device Selection**: Ensure the correct Hailo device type is specified
5. **Headless Mode**: Generally provides better performance as it doesn't need to render GUI windows

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Check if webcam is connected
   - Try different camera index (--camera 1, 2, etc.)
   - Ensure webcam permissions

2. **"Model not found"**
   - Verify model path is correct
   - Check that all model files are present

3. **Qt/X11 Display Errors (GUI Mode)**
   - Use headless mode instead: `./run_yolov11s_headless.sh`
   - Or enable X11 forwarding: `ssh -X pi@your-pi-ip`

4. **Low FPS**
   - Reduce camera resolution
   - Increase confidence threshold
   - Check Hailo device status
   - Use headless mode for better performance

5. **No detections**
   - Lower confidence threshold
   - Check if objects are in frame
   - Verify model is trained for your use case

### Debug Mode

For debugging, you can add verbose output:
```bash
python webcam_yolov11s_detection.py --confidence 0.3
```

## File Structure

```
hailo_examples/
├── webcam_yolov11s_detection.py           # GUI version
├── webcam_yolov11s_detection_headless.py  # Headless version
├── run_yolov11s_webcam.sh                 # GUI launcher
├── run_yolov11s_headless.sh               # Headless launcher
├── YOLOV11s_Webcam_README.md              # This file
├── detection_output/                       # Saved frames (headless mode)
├── models/
│   └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1/
│       ├── labels_squirrel_yolov11s.json
│       ├── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef
│       └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json
└── degirum_env/                            # Virtual environment
```

## Customization

### Using Different Models

To use a different model:

1. Place your model in the `models/` directory
2. Update the `--model-path` parameter
3. Ensure the model is compatible with Hailo devices

### Adding New Classes

To detect different objects:

1. Update the labels file in your model directory
2. Retrain or convert your model accordingly
3. Update the model path in the script

### Performance Optimization

For better performance:

1. Use lower resolution inputs
2. Optimize model quantization
3. Use appropriate Hailo device settings
4. Consider model pipelining for higher throughput
5. Use headless mode for maximum performance

## Support

For issues related to:
- **DeGirum SDK**: Check DeGirum documentation
- **Hailo Hardware**: Refer to Hailo documentation
- **Model Conversion**: Use DeGirum model zoo tools

## License

This script is provided as-is for educational and development purposes. 