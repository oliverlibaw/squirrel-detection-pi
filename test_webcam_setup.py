#!/usr/bin/env python3
"""
Test script to verify webcam and model setup for YOLOV11s detection
"""

import cv2
import os
import sys
from pathlib import Path

def test_webcam(camera_index=0):
    """Test if webcam is working."""
    print(f"Testing webcam at index {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Failed to open webcam at index {camera_index}")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read frame from webcam at index {camera_index}")
        cap.release()
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Webcam working: {width}x{height} @ {fps:.1f}fps")
    cap.release()
    return True

def test_model_path(model_path):
    """Test if model files exist."""
    print(f"Testing model path: {model_path}")
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    # Check for required files
    required_files = [
        "labels_squirrel_yolov11s.json",
        "squirrel_yolov11s--640x640_quant_hailort_multidevice_1.hef",
        "squirrel_yolov11s--640x640_quant_hailort_multidevice_1.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    print("✅ Model files found")
    return True

def test_degirum_import():
    """Test if DeGirum can be imported."""
    print("Testing DeGirum import...")
    
    try:
        import degirum as dg
        print("✅ DeGirum imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import DeGirum: {e}")
        print("Make sure you're in the degirum_env virtual environment")
        return False

def test_opencv_import():
    """Test if OpenCV can be imported."""
    print("Testing OpenCV import...")
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
        return True
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("YOLOV11s Webcam Setup Test")
    print("=" * 50)
    
    # Test imports
    degirum_ok = test_degirum_import()
    opencv_ok = test_opencv_import()
    
    # Test model path
    model_path = "models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1"
    model_ok = test_model_path(model_path)
    
    # Test webcam
    webcam_ok = test_webcam(0)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    print(f"DeGirum Import: {'✅' if degirum_ok else '❌'}")
    print(f"OpenCV Import:  {'✅' if opencv_ok else '❌'}")
    print(f"Model Files:    {'✅' if model_ok else '❌'}")
    print(f"Webcam:         {'✅' if webcam_ok else '❌'}")
    
    if all([degirum_ok, opencv_ok, model_ok, webcam_ok]):
        print("\n🎉 All tests passed! You can run the detection script.")
        print("Run: ./run_yolov11s_webcam.sh")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running detection.")
        
        if not degirum_ok:
            print("- Activate the degirum_env virtual environment")
        if not opencv_ok:
            print("- Install OpenCV: pip install opencv-python")
        if not model_ok:
            print("- Check that your model files are in the correct location")
        if not webcam_ok:
            print("- Connect a webcam or check camera permissions")

if __name__ == "__main__":
    main() 