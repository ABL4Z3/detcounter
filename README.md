# Finger and object Counter Project

This project uses YOLOv8 object detection and Mediapipe hand tracking to count objects and fingers in real-time from a webcam feed. It displays the count of detected objects, fingers, and their total on the video stream along with the FPS (frames per second).

## Features

- Real-time object detection using YOLOv8
- Real-time finger counting using Mediapipe
- Combined display of object and finger counts
- FPS display for performance monitoring

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- ultralytics (YOLOv8)
- OpenCV
- cvzone
- Mediapipe
- PyTorch and torchvision
- Other scientific and utility libraries

## Setup

1. Download or place the YOLOv8 model file `yolo11m.pt` in the project directory.  
   You can use the official YOLOv8 models or a fine-tuned/custom model.

2. Connect a webcam to your computer. The script uses webcam device index `1` by default.  
   If you want to use the default webcam, you may need to change the device index in `yolo-running.py`:
   ```python
   cap = cv2.VideoCapture(0)  # Change to 0 for default webcam
   ```

## Usage

Run the main script:

```bash
python yolo-running.py
```

The video window will open showing the webcam feed with detected objects and finger counts.  
Press the `q` key to quit the program.

## Sample Videos

The project directory contains sample videos (`cars.mp4`, `bikes.mp4`) which you can use for testing or development purposes, though the current main script uses the webcam feed.

## Notes

- Ensure your system has a compatible GPU and CUDA installed for faster YOLOv8 inference.  
- Adjust the confidence threshold in the script if needed for better detection accuracy.

## License

This project is open source and available under the MIT License.
