# TRX City Smart Surveillance POC

**Project:** Point of Interest (POI) Detection & Alert Testing for Smart Surveillance


## 📌 Project Overview
This Proof of Concept (POC) project was developed for the TRX City Smart Surveillance system. The objective was to train a custom computer vision model to automatically detect specific urban infrastructure anomalies and security scenarios in real-time, specifically focusing on open/closed states of critical cabinets and overflowing rubbish bins.

## 📍 Points of Interest (POI) & Scenarios
The model was trained to monitor the following locations and scenarios:
1. **Menara Prudential Ramp**: Rubbish Bin (Overflow) & Electrical Cabinet (Open/Close)
2. **Raintree Plaza (Outdoor Area)**: 
   - Electrical Cabinet (Open/Close) near Super Matcha
   - Fire Hydrant Hose Cabinet (Open/Close) in front of Super Matcha
   - Rubbish Bin (Overflow) near public parking
3. **Event Area**: Fire Hydrant Hose Cabinet (Open/Close)

## 🔄 End-to-End Technical Pipeline

### 1. Data Collection (ROS 2 & Robotics)
- **Process:** We used a mobile robotic platform/camera setup (B2 camera feed) to navigate the POIs and record real-world footage.
- **Tech Stack:** Robot Operating System 2 (ROS 2).
- **Details:** The camera nodes published video frames to ROS 2 topics. We recorded these topics into large ROS 2 Bag files (`.db3` format) to capture high-fidelity, timestamped sensor data across all 6 camera feeds (360-degree views, front, and rear).

### 2. Data Engineering & Extraction
- **Process:** ROS 2 bags are not directly compatible with standard machine learning platforms, so the raw data had to be parsed and converted.
- **Tech Stack:** Python, `rosbags`, `numpy`, `imageio`, OpenCV.
- **Details:** I developed a custom Python pipeline to parse gigabytes of `.db3` bag files, deserialize the `sensor_msgs/Image` payloads, decode the raw byte arrays (handling different color encodings like RGB8), and compile them into standard `.mp4` video files. 

### 3. Data Ingestion & Annotation (Roboflow)
- **Process:** The generated `.mp4` files were uploaded into Roboflow for dataset management.
- **Details:** Roboflow automatically sampled frames from the videos. I performed manual bounding-box annotation to label the critical objects and their states. 
- **Classes Annotated:**
  - `electrical_box_closed` / `electrical_box_open`
  - `firehose_closed` / `firehose_open`
  - `rubbish_bin_normal` / `rubbish_bin_overflow`

### 4. Model Training (Computer Vision)
- **Process:** Trained an object detection model to recognize the annotated classes.
- **Tech Stack:** Roboflow Train (AutoML / YOLO).
- **Details:** We applied data augmentations (like varying brightness and orientation) to make the model robust to different outdoor lighting conditions at Raintree Plaza. We then initiated the automated training pipeline in Roboflow, which optimized a neural network to detect the visual differences between an open vs. closed cabinet.

### 5. Testing & Validation
- **Process:** Evaluated the model's performance on unseen test data to ensure it could reliably trigger alerts for the defined scenarios (e.g., detecting if an electrical box was left open by maintenance, or if a rubbish bin was overflowing).

## 🚀 Key Achievements
- Engineered a robust Python script to successfully salvage and convert massive corrupted ROS 2 bag files into usable `.mp4` formats.
- Managed the end-to-end lifecycle of a Computer Vision POC from raw robotic sensor data collection to a deployed detection model.
