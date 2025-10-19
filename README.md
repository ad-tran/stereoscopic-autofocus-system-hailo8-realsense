# Stereoscopic Autofocus System for Cinema lenses (Intel RealSense D455 + Hailo-8 AI accelerator + Raspberry Pi 5)

## About the project
This project was developed as part of the Camera Technology module in the Media Technology program at the University of Applied Sciences Cologne during the Summer Semester 2025.

Supervision:
- Prof. Gregor Fischer - IMP F07
- Prof. Dirk Poggemann - IMP F07
- Christian Loebich - IMP F07

Project Manager: Anh Duong Tran

Contributions:
- Project Idea, Concept & Equipment Selection: Anh Duong Tran
- Control Hardware Setup: Angelika Allgäuer, Mark Ulanowski
- Motor Control System: Angelika Allgäuer, Doron Kohler, Mark Ulanowski
- Stereo Camera System & GUI Development: Hannah Strippel, Anh Duong Tran
- Object Detection & Tracking: Anh Duong Tran

## Overview
![Finalset](images/finalResult.png)

This project implements a real-time stereoscopic autofocus system, which provides an interactive GUI to perform person detection, tracking, depth-based distance estimation, and stepper-motor driven lens focusing in real time. It uses:
- Intel RealSense D455 for RGB + depth frames
- YOLO models for person, face, and segmentation
- Hailo-8 AI accelerator for besser Object detection
- SORT for multi-object tracking
- An Adafruit Motor HAT to drive a stepper on the lens focus ring
- Raspberry Pi 5
- The whole system powered by a Vmount baterry
![Hardware](images/controlHardware.png)


## Key Features
- Launch flow: Loading screen → Calibration screen → Main screen
- Calibration checklist with lighting condition selection
- Live video view with adjustable ROI (drag corners)
- Person detection + SORT tracking; tap a tracked person to focus on them
- Face detection for precise distance; person mask sampling fallback if no face is visible
- Optical-flow point in ROI for tap-to-focus when no tracked person is selected
- Depth correction via lighting-dependent LUTs (inside/outside, good/bad light)
- Real-time focus distance computation and conversion to stepper steps
- Focus timing slider (smooth vs. fast moves) and hysteresis to reduce jitter
- Depth profile panel with:
  - Valid focus range guidance
  - Current target focus distance (white bar)
  - Motor focus plane indicator
  - Colored sample points (green = selected target, gray = others)
- FPS display, reset tracking button, and safe cleanup on exit

## Demo Video

Tracking-by-detection with YOLO und SORT:

<div align="center">
  <a href="https://www.youtube.com/shorts/ReqabGZ8ctM">
    <img src="images/trackingSortYT.png" alt="Object Tracking" width="30%">
  </a>
  <a href="https://www.youtube.com/shorts/-5ozBTq2l24?feature=share">
    <img src="images/oFYT.png" alt="Optical Flow" width="30%">
  </a>
  <a href="https://www.youtube.com/shorts/RXhNjNWLM1I?feature=share">
    <img src="images/GUIShortYT.png" alt="GUI" width="30%">
  </a>
</div>

## Distance Correction for Stereo Camera

We tested the Intel RealSense D455 stereo camera and discovered that the measured distances were consistently greater than the actual distances. To address this, we developed a correction function and lookup tables (LUTs) to calibrate the measured distances to true distances across different lighting conditions.

Inside in good lighting condition
<div align="center">
  <img src="images/InsideGoodLighting.png" alt="1.1" width="45%">
  <img src="images/InsideGoodLightingCorrected.png" alt="1.2" width="45%">
</div>

Inside in bad lighting condition
<div align="center">
  <img src="images/InsideBadLighting.png" alt="2.1" width="45%">
  <img src="images/InsideBadLightingCorrected.png" alt="2.2" width="45%">
</div>

Outsid in good lighting condition
<div align="center">
  <img src="images/OutsideGoodLighting.png" alt="1.1" width="45%">
  <img src="images/OutsideGoodLightingCorrected.png" alt="1.2" width="45%">
</div>

Link to google sheet of measurement
```bash
https://docs.google.com/spreadsheets/d/1Mt2iLZNg792dCo4I_J54miuZmEHCaRKvOUn7tRtqKD0/edit?usp=sharing
```


## Computer Vision Pipeline
- Person detection (YOLO model via DeGirum/Hailo)
- SORT tracker for consistent IDs and track selection
- Face detection within the selected person’s crop
- Person segmentation mask for robust depth sampling (selected vs. non-selected)
- Optical flow around a user-selected point inside the ROI (depth windows aggregated)

## Focusing Logic
- Depth readouts in meters from aligned depth frame
- Lighting-dependent correction:
  - Linear interpolation across a measured→true distance LUT
- Focus offset (camera-to-lens baseline) added to corrected distance
- Conversion to motor steps via a calibrated steps↔distance LUT
- Hysteresis threshold to trigger motor moves only on meaningful distance changes

## Hardware Control
- RealSense D455 streams:
  - Color: 1280×720 @ 30 FPS
  - Depth: 848×480 @ 30 FPS
  - Depth laser power set to 360
- Motor HAT:
  - Background process consumes move commands (steps + focus_time)
  - Interleaved stepping; homing back to zero on shutdown

## Requirements
- Python 3.10+
- Kivy, NumPy, OpenCV
- pyrealsense2 (Intel librealsense2 stack)
- degirum PySDK for Inference on Hailo device (or HailoRT runtime as required)
- Adafruit MotorKit and adafruit_motor
- SORT
