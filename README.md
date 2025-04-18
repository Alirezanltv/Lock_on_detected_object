# Bird Detection and Tracking with YOLOv8

This application uses YOLOv8 and ByteTrack to detect and track birds in video footage, with smooth bounding box tracking to minimize jitter.

## Features

- Real-time bird detection using YOLOv8
- Stable tracking using ByteTrack algorithm
- Smooth bounding box movement to reduce jitter
- Support for video file input
- Real-time visualization and video output

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input video file (e.g., `bird.mp4` or `two_birds.mp4`) in the project directory.

2. Run the application:
```bash
python bird_detection.py
```

3. The application will:
   - Load the YOLOv8 model (will be downloaded automatically on first run)
   - Process the input video
   - Display the tracking in real-time
   - Save the output to `output.mp4`

4. Press 'q' to quit the application.

## Customization

You can modify the following parameters in `bird_tracker.py`:

- `model_name`: Change the YOLOv8 model (default: "yolov8n.pt")
- `conf_threshold`: Adjust detection confidence threshold (default: 0.3)
- `smooth_factor`: Modify smoothing intensity (default: 0.3)

## Notes

- The application uses the COCO dataset's bird class (class_id: 14)
- For best performance, use a GPU with CUDA support
- The output video will be saved in MP4 format 
