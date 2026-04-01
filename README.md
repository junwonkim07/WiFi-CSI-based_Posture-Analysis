WiFi CSI-based Contactless Pose and Heart Rate Tracking
Overview

This project presents a contactless human sensing system using WiFi Channel State Information (CSI).
The system estimates human body pose and heart rate without cameras or wearable devices.

By leveraging the penetration property of WiFi signals, the system remains functional even under occlusion, while preserving user privacy.

Objectives
Perform human pose estimation using WiFi CSI
Estimate heart rate without wearable sensors
Evaluate pose accuracy using PCK@50
Measure exercise counting accuracy
Analyze performance under occlusion
Key Idea (Summary)
Use WiFi signals to detect human motion
Estimate joint positions and evaluate with PCK@50
Build a real-time exercise analysis system
WiFi CSI

WiFi CSI (Channel State Information) describes how a wireless signal changes in amplitude and phase as it propagates from transmitter to receiver.

These variations capture environmental changes and enable motion sensing and physiological signal extraction.

WiFi Sensing

WiFi sensing utilizes RF signals to detect environmental changes without cameras or physical sensors.

Based on IEEE 802.11bf standardization efforts
Works in low-light or occluded environments
Preserves user privacy
Pose Estimation via WiFi

This project uses the RuView framework to infer human pose from CSI data.

Pipeline
Collect CSI amplitude and phase data
Preprocess signals
Extract motion-related features
Apply deep learning models (CNN / RNN)
Output 2D or 3D skeleton with 17 keypoints
Heart Rate Estimation via WiFi

Heart rate is estimated by analyzing subtle chest movements reflected in CSI phase data.

Method
Remove dominant breathing signals
Extract heartbeat-related frequency components
Apply Fast Fourier Transform (FFT)
Convert frequency to BPM
Experimental Setup
Environment
Indoor 3m × 3m controlled space
Metallic objects removed to reduce signal noise
Hardware
4 × ESP32-S3
Arranged in a 2×2 square
Height: 1.3 m
Oriented toward the center
Network
Local WiFi network (offline)
CSI data transmitted via UDP (port 5005)
Sampling rate: 30 Hz
Ground Truth (Pose)

Ground truth pose data is obtained using Google MediaPipe.

Camera distance: 2 m
Camera height: 1.2 m
Resolution: 1080p at 30 FPS

Each frame provides 17 keypoints for comparison.

Evaluation Metrics
Pose Accuracy
PCK@50 (Percentage of Correct Keypoints)
Normalized by torso length
Counting Accuracy
Accuracy of exercise repetition detection
Experiments
Exercise Tracking

Exercises:

Push-ups
Squats
Running in place

Metrics:

PCK@50
Counting accuracy
Latency
Occlusion Test

Performance is compared under two conditions:

Without obstacles
With obstacles (yoga mat, box, chair)

The experiment evaluates how occlusion affects pose estimation and counting accuracy.

Heart Rate Measurement

Heart rate estimation is validated against Apple Watch measurements.

Conditions:

Resting
During exercise
Post-exercise

Metrics:

MAE (Mean Absolute Error)
Correlation coefficient (R²)

Target:

MAE ≤ 10 BPM
R² ≥ 0.8
Tech Stack
Hardware: ESP32-S3
Backend: Python, Docker (RuView)
Pose Ground Truth: MediaPipe
Models: CNN / RNN
Signal Processing: FFT
Limitations
Increased noise during dynamic movement
Reduced heart rate accuracy during exercise
Requires controlled experimental setup
