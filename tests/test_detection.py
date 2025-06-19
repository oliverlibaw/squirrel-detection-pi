#!/usr/bin/env python3
"""
Tests for the squirrel detection functionality
"""

import pytest
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_path_exists():
    """Test that the model directory exists."""
    model_path = Path("models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1")
    assert model_path.exists(), f"Model path {model_path} does not exist"

def test_model_files_exist():
    """Test that all required model files exist."""
    model_path = Path("models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1")
    
    required_files = [
        "labels_squirrel_yolov11s.json",
        "squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef",
        "squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json"
    ]
    
    for file in required_files:
        file_path = model_path / file
        assert file_path.exists(), f"Required model file {file_path} does not exist"

def test_labels_file_content():
    """Test that the labels file contains valid JSON."""
    import json
    
    labels_file = Path("models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1/labels_squirrel_yolov11s.json")
    
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    assert isinstance(labels_data, dict), "Labels file should contain a dictionary"
    assert len(labels_data) > 0, "Labels file should not be empty"

def test_script_files_exist():
    """Test that all script files exist."""
    required_scripts = [
        "webcam_yolov11s_detection.py",
        "webcam_yolov11s_detection_headless.py",
        "test_webcam_setup.py"
    ]
    
    for script in required_scripts:
        script_path = Path(script)
        assert script_path.exists(), f"Required script {script_path} does not exist"

def test_launcher_scripts_exist():
    """Test that launcher scripts exist and are executable."""
    launcher_scripts = [
        "run_yolov11s_webcam.sh",
        "run_yolov11s_headless.sh"
    ]
    
    for script in launcher_scripts:
        script_path = Path(script)
        assert script_path.exists(), f"Launcher script {script_path} does not exist"
        assert os.access(script_path, os.X_OK), f"Launcher script {script_path} should be executable"

def test_requirements_file():
    """Test that requirements.txt exists and contains required packages."""
    requirements_file = Path("requirements.txt")
    assert requirements_file.exists(), "requirements.txt should exist"
    
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    required_packages = ["degirum", "opencv-python", "numpy"]
    for package in required_packages:
        assert package in content, f"requirements.txt should contain {package}"

def test_readme_exists():
    """Test that README.md exists."""
    readme_file = Path("README.md")
    assert readme_file.exists(), "README.md should exist"
    
    with open(readme_file, 'r') as f:
        content = f.read()
    
    # Check for key sections
    assert "Squirrel Detection" in content, "README should mention Squirrel Detection"
    assert "Installation" in content, "README should have installation section"
    assert "Usage" in content, "README should have usage section"

if __name__ == "__main__":
    pytest.main([__file__]) 