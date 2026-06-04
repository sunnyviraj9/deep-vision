# DeepVision Crowd Monitor

AI-powered crowd density estimation and overcrowding detection system using Deep Learning and Computer Vision.

## Overview

DeepVision Crowd Monitor is designed to estimate crowd density, predict crowd counts, and detect overcrowding in real time. The project utilizes density map generation, crowd counting neural networks, and an interactive dashboard for monitoring crowd situations.

## Features

- Crowd Density Estimation
- Crowd Count Prediction
- Real-Time Crowd Monitoring
- Density Heatmap Visualization
- Overcrowding Detection
- Email Alerts
- SMS Notifications
- Streamlit Dashboard
- Webcam & Video Feed Support

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Streamlit
- Twilio
- Ngrok

## Project Workflow

1. Load crowd images and annotations.
2. Generate density maps using Gaussian filtering.
3. Train crowd counting models (CSRNet & Tiny-MCNN).
4. Predict crowd density and count.
5. Detect overcrowding situations.
6. Send alerts via Email and SMS.
7. Display results on a Streamlit dashboard.

## Dataset

This project uses the ShanghaiTech Crowd Counting Dataset.

- Part A: Dense crowd scenes
- Part B: Sparse crowd scenes

## Installation

Clone the repository:

```bash
git clone https://github.com/sunnyviraj9/deep-vision.git
cd deep-vision
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install numpy scipy matplotlib pillow
pip install streamlit twilio pyngrok tqdm
```

## Running the Project

### Train the Model

```bash
python train.py
```

### Run Real-Time Monitoring

```bash
python realtime.py
```

### Launch Dashboard

```bash
streamlit run app_task4.py
```

## Applications

- Smart City Surveillance
- Public Event Monitoring
- Stadium Management
- Airport Crowd Analysis
- Railway Station Monitoring
- Disaster Management

## Future Improvements

- YOLO-based Person Detection
- Multi-Camera Monitoring
- Cloud Deployment (AWS/GCP)
- Mobile Application Integration
- Advanced Crowd Analytics

## Author

**Sunny Viraj**

B.Tech Data Science  
NMIMS Hyderabad

GitHub: https://github.com/sunnyviraj9

## License

This project is intended for educational and research purposes.
