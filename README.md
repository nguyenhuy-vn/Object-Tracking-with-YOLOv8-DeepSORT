# Object-Tracking-with-YOLOv8-DeepSORT
🚀Overview
This module combines YOLOv8 (for object detection) and DeepSORT (for object tracking), designed for:  
- **Production line tracking** to monitor products moving on a conveyor belt.  
- **Traffic monitoring** to count the number of cars passing through a path.  

The module supports two modes:  
- **Live Camera Mode** – Track objects in real time using a webcam or external camera.  
- **Input Video Mode** – Process and analyze pre-recorded videos.  

📦 **Project**  
│-- 📁 configs      # Configure parameters (model, tracking, input, output, etc.)  
│-- 📁 data         # Store input videos  
│-- 📁 output       # Save result videos after tracking  
│-- 📁 model        # Contain YOLOv8 and DeepSORT models  
│-- 📁 src          # Main source code of the module  
│-- 📁 utils        # Utility functions for data processing
│-- requirements.txt # Necessary packages
│-- README.md       # Project documentation

🛠️ **Installation**
Install the required dependencies:
```bash
pip install -r requirements.txt
```
🎯 **Usage**  

1️⃣ **Live Camera Mode**  
To use the live camera mode, modify the `configs.yaml` file to enable camera input. Once configured, run:
```bash
python test.py
```
2️⃣ **Input Video Mode**
Run the module with a video file:
1. Place your video file in the data/ folder
2. Update the video path in configs.yaml (located in the configs/ folder).
3. Run the module:
```bash
python test.py
```
🎯 **Target Selection**
You can specify which objects to track in the configs.yaml file:
![image](https://github.com/user-attachments/assets/5ec0e975-9b6a-4f98-bcf3-6c2a9210d603)

By default, the module tracks all objects that the model can detect.
