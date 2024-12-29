# Using DeGirum PySDK with Hailo Hardware

This repository provides a step-by-step guide on using the DeGirum PySDK with Hailo hardware for efficient AI inference. The guide assumes that Hailo's tools and SDK are already installed and configured on your machine.

---

## Prerequisites

- **Hailo Tools Installed**: Ensure that Hailo's tools and SDK are properly installed and configured. Refer to [Hailo's documentation](https://hailo.ai/) for detailed setup instructions.
- **Python 3.9 or Later**: Ensure Python is installed on your system. You can check your Python version using:
  ```bash
  python3 --version
  ```

---

## Step 1: Set Up a Virtual Environment

To keep your Python environment clean and avoid conflicts, it's recommended to use a virtual environment for installing the required packages.

### Create a Virtual Environment

#### Linux/macOS

1. Navigate to the directory where you'd like to create the environment.
2. Run the following commands:
   ```bash
   python3 -m venv degirum_env
   source degirum_env/bin/activate
   ```

#### Windows

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

## Step 2: Install DeGirum CLI

Install the DeGirum CLI package from PyPI using `pip`. This package includes `degirum`, `degirum_tools`, and `degirum_cli` for easy testing and development:

```bash
pip install degirum_cli
```

This will automatically install:

- **`degirum`**: The core PySDK library for AI inference.
- **`degirum_tools`**: Additional tools for streaming, benchmarking, and other utilities.
- **`degirum_cli`**: A command-line interface for interacting with DeGirum PySDK.

---

## Step 3: Verify Installation

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

Ensure that the `Devices` section lists the Hailo device details as shown above.

---

## Step 4: Example Usage with DeGirum CLI

DeGirum CLI simplifies testing and development. Below is an example of using the CLI with Hailo hardware:

### Run Inference on an Image

#### Linux/macOS

```bash
degirum_cli predict-image \
    --inference-host-address @local \
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 \
    --model-zoo-url degirum/models_hailort
```

#### Windows

For Windows, use the following format instead of the backslash continuation:

```cmd
degirum_cli predict-image ^
    --inference-host-address @local ^
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 ^
    --model-zoo-url degirum/models_hailort
```

- **`--inference-host-address`**: Set the inference host address (use `@local` for local inference).
- **`--model-name`**: Provide the name of the model you want to use.
- **`--model-zoo-url`**: Specify the URL for the Hailo model zoo.

### Run Inference on a Video

#### Linux/macOS

```bash
degirum_cli predict-video \
    --inference-host-address @local \
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 \
    --model-zoo-url degirum/models_hailort
```

#### Windows

For Windows, use the following format instead of the backslash continuation:

```cmd
degirum_cli predict-image ^
    --inference-host-address @local ^
    --model-name yolov8n_relu6_coco--640x640_quant_hailort_hailo8_1 ^
    --model-zoo-url degirum/models_hailort
```

- **`--inference-host-address`**: Set the inference host address (use `@local` for local inference).
- **`--model-name`**: Provide the name of the model you want to use.
- **`--model-zoo-url`**: Specify the URL for the Hailo model zoo.

---

## Additional Resources

- [DeGirum Documentation](https://docs.degirum.com)
- [Hailo Documentation](https://hailo.ai/)

---

Feel free to clone this repository and contribute by submitting pull requests or raising issues.

