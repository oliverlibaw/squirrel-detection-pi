#!/usr/bin/env python3
"""
YOLOV11s Object Detection with Webcam on Raspberry Pi (Headless Version)
This script runs the custom YOLOV11s model on webcam input and saves annotated frames to disk.
"""

import cv2
import degirum as dg
import degirum_tools
import numpy as np
import time
import argparse
import os
from pathlib import Path

class YOLOV11sWebcamDetectorHeadless:
    def __init__(self, model_path, confidence_threshold=0.5, device_type=['HAILORT/HAILO8L']):
        """
        Initialize the YOLOV11s detector with webcam input (headless mode).
        
        Args:
            model_path: Path to the model directory
            confidence_threshold: Minimum confidence score for detections
            device_type: Hailo device type
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device_type = device_type
        
        # Load labels
        self.labels = self._load_labels()
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize webcam
        self.cap = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Output directory for saved frames
        self.output_dir = Path("detection_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_labels(self):
        """Load class labels from the model directory."""
        labels_file = self.model_path / "labels_squirrel_yolov11s.json"
        if labels_file.exists():
            import json
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
                return labels_data
        return {}
    
    def _load_model(self):
        """Load the YOLOV11s model."""
        print(f"Loading model from: {self.model_path}")
        
        # Get the model name from the directory
        model_name = self.model_path.name
        
        # Load the model using local models directory
        model = dg.load_model(
            model_name=model_name,
            inference_host_address="@local",
            zoo_url="models",  # Use local models directory instead of cloud zoo
            token="",
            device_type=self.device_type
        )
        
        # Set overlay properties after model is loaded
        model.overlay_show_probabilities = True  # Show confidence scores
        model.overlay_show_labels = True         # Show class labels
        model.overlay_line_thickness = 2         # Bounding box line thickness
        model.overlay_font_scale = 0.6           # Text size
        model.overlay_font_thickness = 2         # Text thickness
        
        print(f"Model loaded successfully: {model_name}")
        return model
    
    def _initialize_webcam(self, camera_index=0, resolution=(640, 480)):
        """Initialize webcam with specified settings."""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # For Raspberry Pi, try to optimize camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        print(f"Webcam initialized: {resolution[0]}x{resolution[1]} @ 30fps")
        return True
    
    def _calculate_fps(self):
        """Calculate and update FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _add_text_overlay(self, frame, inference_time, detection_count):
        """Add text overlay to the frame."""
        # Add FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Add model info
        model_info = f"Model: YOLOV11s"
        cv2.putText(frame, model_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Add confidence threshold info
        conf_info = f"Conf: {self.confidence_threshold}"
        cv2.putText(frame, conf_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Add inference time
        inference_text = f"Inference: {inference_time*1000:.1f}ms"
        cv2.putText(frame, inference_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection count
        count_text = f"Detections: {detection_count}"
        cv2.putText(frame, count_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _filter_detections(self, result):
        """Return a filtered list of detections based on confidence threshold."""
        if hasattr(result, 'results') and result.results:
            filtered_results = [
                detection for detection in result.results
                if hasattr(detection, 'score') and detection.score >= self.confidence_threshold
            ]
            return filtered_results
        return []
    
    def run_detection(self, camera_index=0, resolution=(640, 480), save_interval=30):
        """
        Run object detection on webcam stream and save frames to disk.
        
        Args:
            camera_index: Camera index
            resolution: Camera resolution (width, height)
            save_interval: Save a frame every N frames (default: 30 = ~1 frame per second at 30fps)
        """
        try:
            # Initialize webcam
            if not self._initialize_webcam(camera_index, resolution):
                return
            
            print("Starting YOLOV11s detection on webcam (headless mode)...")
            print(f"Frames will be saved to: {self.output_dir}")
            print("Press Ctrl+C to stop")
            
            frame_counter = 0
            saved_frame_counter = 0
            
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                # Run inference
                start_time = time.time()
                result = self.model(frame)
                inference_time = time.time() - start_time
                
                # Filter detections by confidence (for counting/display purposes)
                filtered_results = self._filter_detections(result)
                
                # Get the overlay image with bounding boxes
                if hasattr(result, 'image_overlay'):
                    output_frame = result.image_overlay.copy()
                else:
                    output_frame = frame.copy()
                
                # Add text overlay
                self._add_text_overlay(output_frame, inference_time, len(filtered_results))
                
                # Calculate FPS
                self._calculate_fps()
                
                # Save frame periodically
                if frame_counter % save_interval == 0:
                    saved_frame_counter += 1
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = self.output_dir / f"yolov11s_frame_{timestamp}_{saved_frame_counter:04d}.jpg"
                    cv2.imwrite(str(filename), output_frame)
                    print(f"Saved frame: {filename} (Detections: {len(filtered_results)}, FPS: {self.current_fps:.1f})")
                
                frame_counter += 1
                
                # Print status every 100 frames
                if frame_counter % 100 == 0:
                    print(f"Processed {frame_counter} frames, FPS: {self.current_fps:.1f}, Detections: {len(filtered_results)}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during detection: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        print("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="YOLOV11s Object Detection with Webcam (Headless)")
    parser.add_argument("--model-path", 
                       default="models/squirrel_yolov11s--640x640_quant_hailort_multidevice_1",
                       help="Path to the model directory")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--width", type=int, default=640,
                       help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Camera height (default: 480)")
    parser.add_argument("--device", default="HAILORT/HAILO8L",
                       help="Hailo device type (default: HAILORT/HAILO8L)")
    parser.add_argument("--save-interval", type=int, default=30,
                       help="Save a frame every N frames (default: 30)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        return
    
    # Create detector
    detector = YOLOV11sWebcamDetectorHeadless(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device_type=[args.device]
    )
    
    # Run detection
    detector.run_detection(
        camera_index=args.camera,
        resolution=(args.width, args.height),
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main() 