# 🐿️ Squirrel Detection with YOLOV11s on Raspberry Pi

Real-time squirrel detection using a custom YOLOV11s model on Raspberry Pi with Hailo AI accelerator HAT.

## 🚀 Features

- **Real-time Detection**: Live object detection from webcam feed
- **Bounding Box Visualization**: Draws boxes around detected squirrels
- **Confidence Scores**: Displays confidence scores for each detection
- **FPS Monitoring**: Real-time FPS display
- **Multiple Modes**: GUI mode with live display or headless mode with frame saving
- **Optimized for Raspberry Pi**: Uses Hailo AI accelerator for efficient inference

## 📋 Prerequisites

- Raspberry Pi with Hailo AI accelerator HAT
- DeGirum SDK installed
- USB webcam or Pi Camera
- Custom YOLOV11s model converted to DeGirum format

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/squirrel-detection-pi.git
   cd squirrel-detection-pi
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv degirum_env
   source degirum_env/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your model files:**
   ```
   models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1/
   ├── labels_squirrel_yolov11s.json
   ├── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef
   └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json
   ```

## 🎯 Usage

### GUI Mode (with live display)

**Use when you have a desktop environment or X11 forwarding available.**

```bash
# Quick start
./run_yolov11s_webcam.sh

# Or run directly
source degirum_env/bin/activate
python webcam_yolov11s_detection.py --confidence 0.5
```

**Controls:**
- `q` - Quit the application
- `s` - Save a screenshot
- `h` - Show help information

### Headless Mode (save frames to disk)

**Use when running over SSH or in environments without display support.**

```bash
# Quick start
./run_yolov11s_headless.sh

# Or run directly
source degirum_env/bin/activate
python webcam_yolov11s_detection_headless.py --save-interval 30
```

**Features:**
- Saves annotated frames to `detection_output/` directory
- Configurable save interval (default: every 30 frames ≈ 1 frame per second)
- Real-time console output with FPS and detection counts
- No display requirements - works over SSH

## ⚙️ Configuration

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

### Example Usage

```bash
# High confidence detection
python webcam_yolov11s_detection.py --confidence 0.8

# Lower resolution for better performance
python webcam_yolov11s_detection.py --width 320 --height 240

# Save frames more frequently
python webcam_yolov11s_detection_headless.py --save-interval 15
```

## 📊 Model Information

- **Model**: Custom YOLOV11s
- **Class**: Squirrel
- **Input Resolution**: 640x640
- **Device**: Hailo8L/Hailo8
- **Format**: Quantized HailoRT
- **Performance**: ~22 FPS on Raspberry Pi with Hailo accelerator

## 🔧 Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Check if webcam is connected
   - Try different camera index: `--camera 1`
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
   - Lower confidence threshold: `--confidence 0.3`
   - Check if objects are in frame
   - Verify model is trained for your use case

### Testing Setup

Run the test script to verify your setup:

```bash
python test_webcam_setup.py
```

## 📁 Project Structure

```
squirrel-detection-pi/
├── webcam_yolov11s_detection.py           # GUI version
├── webcam_yolov11s_detection_headless.py  # Headless version
├── run_yolov11s_webcam.sh                 # GUI launcher
├── run_yolov11s_headless.sh               # Headless launcher
├── test_webcam_setup.py                   # Setup verification
├── requirements.txt                       # Python dependencies
├── config.yaml                           # Configuration file
├── detection_output/                      # Saved frames (headless mode)
├── models/                               # Model files (add your own)
│   └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1/
│       ├── labels_squirrel_yolov11s.json
│       ├── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef
│       └── squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json
└── README.md                             # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeGirum](https://degirum.com/) for the AI inference SDK
- [Hailo](https://hailo.ai/) for the AI accelerator hardware
- [Ultralytics](https://ultralytics.com/) for YOLO models

## 📞 Support

For issues and questions:
- Create an issue in this repository
- Check the [DeGirum documentation](https://docs.degirum.com/)
- Refer to [Hailo documentation](https://hailo.ai/developer-zone/)

---

**Happy squirrel hunting! 🐿️🔍**

