# YOLO 11n Object Detection Application

Real-time object detection application using YOLO 11n with support for multiple video sources including webcams, network streams, and Raspberry Pi Camera.

## Features

- **Multiple Video Sources**
  - USB Webcams
  - Network Streams (RTSP, HTTP)
  - Raspberry Pi Camera (via GStreamer)
- **Real-time Object Detection** using YOLO 11n
- **Hardware Optimization**
  - NCNN export for Raspberry Pi
  - ROCm support detection for AMD GPUs (PC)
- **User-friendly Interface** built with Streamlit

## Requirements

- Python 3.12
- Webcam or network camera (optional)
- Raspberry Pi Camera Module (for Pi deployment)

### Hardware Acceleration

- **NVIDIA GPUs**: CUDA support via PyTorch
- **AMD GPUs**: ROCm support (automatically detected)
- **Raspberry Pi**: NCNN optimization available

## Installation

### Using uv (Recommended)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install streamlit opencv-python
uv pip install ultralytics
uv pip install .
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install .
```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

### Selecting a Video Source

1. **Webcam**: Select "Webcam" and choose the camera index (usually 0)
2. **Network Stream**: Select "Network Stream" and enter the URL (e.g., `rtsp://...`)
3. **Pi Camera**: Select "Pi Camera" (Raspberry Pi only)

### Running Detection

1. Select your video source from the sidebar
2. Click **"Start Detection"** to begin
3. Click **"Stop Detection"** to end the stream

### Optimizing for Raspberry Pi

For better performance on Raspberry Pi:

1. Click **"Optimize for Pi (Export to NCNN)"** in the sidebar
2. Wait for the export to complete
3. Check **"Use Optimized NCNN Model"**
4. Start detection

## Raspberry Pi Setup

### Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libcamera-dev gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
```

### Running on Pi

Follow the standard installation steps, then ensure:
- libcamera is properly configured
- GStreamer is installed
- OpenCV is built with GStreamer support

## Platform-Specific Notes

### PC (Linux/Windows/macOS)

- CUDA acceleration is automatic if NVIDIA GPU is detected
- ROCm support is checked automatically for AMD GPUs
- CPU-only mode is supported

### Raspberry Pi 4

- Use the Pi Camera option for best compatibility
- NCNN export significantly improves inference speed
- Recommended resolution: 640x480 for real-time performance

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── pyproject.toml         # Project dependencies and metadata
├── verify_install.py      # Installation verification script
└── README.md             # This file
```

## Troubleshooting

### Camera not detected
- Verify camera index (try 0, 1, 2)
- Check camera permissions
- For Pi Camera, ensure libcamera is working: `libcamera-hello`

### Slow inference
- Use NCNN optimization on Raspberry Pi
- Lower the video resolution
- Use a smaller YOLO model variant

### GStreamer errors (Pi)
- Ensure GStreamer plugins are installed
- Test pipeline manually: `gst-launch-1.0 libcamerasrc ! videoconvert ! autovideosink`

## License

This project uses the following libraries:
- **Ultralytics YOLO**: AGPL-3.0
- **Streamlit**: Apache-2.0
- **OpenCV**: Apache-2.0

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection model
- [Streamlit](https://streamlit.io/) for the web interface framework
