# Using DeGirum PySDK, DeGirum Tools, and DeGirum CLI with Hailo Hardware

This repository provides a comprehensive guide on using DeGirum PySDK, DeGirum Tools, and DeGirum CLI with Hailo hardware for efficient AI inference. These tools simplify edge AI development by enabling seamless integration, testing, and deployment of AI models on multiple hardware platforms, including Hailo-8 and Hailo-8L.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setting Up the Environment](#setting-up-the-environment)
   - [Linux/macOS](#linuxmacos)
   - [Windows](#windows)
4. [Installing DeGirum CLI](#installing-degirum-cli)
5. [Verifying Installation](#verifying-installation)
6. [Example Usage](#example-usage)
   - [Image Inference](#image-inference)
   - [Video Inference](#video-inference)
7. [Additional Resources](#additional-resources)

---

## Introduction

DeGirum provides a powerful suite of tools to simplify the development and deployment of edge AI applications:

- [DeGirum PySDK](https://github.com/DeGirum/PySDKExamples): The core library for integrating AI inference capabilities into applications.
- [DeGirum Tools](https://github.com/DeGirum/degirum_tools): Utilities for benchmarking, streaming, and interacting with DeGirum's model zoo.
- [DeGirum CLI](https://pypi.org/project/degirum-cli/): A command-line interface for testing and managing AI models.

These tools are designed to be hardware-agnostic, enabling developers to build scalable, flexible solutions without being locked into a specific platform.

---

## Prerequisites

- **Hailo Tools Installed**: Ensure that Hailo's tools and SDK are properly installed and configured. Refer to [Hailo's documentation](https://hailo.ai/) for detailed setup instructions.
- **Python 3.9 or Later**: Ensure Python is installed on your system. You can check your Python version using:
  ```bash
  python3 --version
  ```

---

## Setting Up the Environment

To keep your Python environment clean and avoid conflicts, it's recommended to use a virtual environment for installing the required packages.

### Linux/macOS

1. Navigate to the directory where you'd like to create the environment.
2. Run the following commands:
   ```bash
   python3 -m venv degirum_env
   source degirum_env/bin/activate
   ```

### Windows

1. Navigate to the directory where you'd like to create the environment.
2. Run the following commands:
   ```bash
   python3 -m venv degirum_env
   degirum_env\Scripts\activate
   ```

### Update `pip`

Ensure `pip` is up-to-date within your virtual environment:

```bash
pip install --upgrade pip
```

---

## Installing DeGirum CLI

Install the DeGirum CLI package from PyPI using `pip`. This package includes `degirum`, `degirum_tools`, and `degirum_cli` for easy testing and development:

```bash
pip install degirum_cli
```

This will automatically install:
- **`degirum`**: The core PySDK library for AI inference.
- **`degirum_tools`**: Additional tools for streaming, benchmarking, and other utilities.
- **`degirum_cli`**: A command-line interface for interacting with DeGirum PySDK.

---

## Verifying Installation

To verify the installation, run the following commands:

### Check CLI Installation

```bash
degirum_cli --help
```

You should see a list of available commands and their usage.

### Check Hailo Hardware Integration

Run the following command to verify that the Hailo hardware is recognized by the DeGirum package:

```bash
degirum sys-info
```

Look for `hailort` in the output to ensure the Hailo hardware is properly integrated. Below is an example output when Hailo hardware is detected:

```
Devices:
  HAILORT/HAILO8:
  - '@Index': 0
    Board Name: Hailo-8
    Device Architecture: HAILO8
    Firmware Version: 4.19.0
    ID: '0000:02:00.0'
    Part Number: HM218B1C2LA
    Product Name: HAILO-8 AI ACCELERATOR M.2 B+M KEY MODULE
    Serial Number: SomeSerialNumber
```

> **Note:** DeGirum PySDK supports Hailo Runtime version **4.19.0**. Ensure your Hailo environment is configured to use this version.

---

## Example Usage

### Image Inference

#### Linux/macOS

```bash
degirum_cli predict-image \
    --inference-host-address @local \
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 \
    --model-zoo-url degirum/models_hailort
```

#### Windows

```cmd
degirum_cli predict-image ^
    --inference-host-address @local ^
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 ^
    --model-zoo-url degirum/models_hailort
```

### Video Inference

#### Linux/macOS

```bash
degirum_cli predict-video \
    --inference-host-address @local \
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 \
    --model-zoo-url degirum/models_hailort
```

#### Windows

```cmd
degirum_cli predict-video ^
    --inference-host-address @local ^
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 ^
    --model-zoo-url degirum/models_hailort
```

---

## Additional Resources

- [Hailo Model Zoo](./hailo_model_zoo.md): Explore the full list of models optimized for Hailo hardware.
- [Hailo8 Tutorial Notebook](examples/quick_start_hailo8.ipynb): Jupyter notebook tutorial to get started with Hailo8.
- [Hailo8L Tutorial Notebook](examples/quick_start_hailo8l.ipynb): Jupyter notebook tutorial to get started with Hailo8L.
- [DeGirum Documentation](https://docs.degirum.com)
- [Hailo Documentation](https://hailo.ai/)

---

Feel free to clone this repository and contribute by submitting pull requests or raising issues.

