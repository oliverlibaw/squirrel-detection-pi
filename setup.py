#!/usr/bin/env python3
"""
Setup script for Squirrel Detection Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="squirrel-detection-pi",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time squirrel detection using YOLOV11s on Raspberry Pi with Hailo AI accelerator",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/squirrel-detection-pi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "squirrel-detection=webcam_yolov11s_detection:main",
            "squirrel-detection-headless=webcam_yolov11s_detection_headless:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="squirrel detection yolo computer vision raspberry pi hailo ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/squirrel-detection-pi/issues",
        "Source": "https://github.com/yourusername/squirrel-detection-pi",
        "Documentation": "https://github.com/yourusername/squirrel-detection-pi#readme",
    },
) 