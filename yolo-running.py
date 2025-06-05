from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import mediapipe as mp

# Load YOLOv8 model
model = YOLO("yolo11m.pt").to('cuda')  # You can use yolov8s.pt or a fine-tuned model

# Load class names
class_names = model.names

# Mediapipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(1)  # Change to `1` if using an external webcam
cap.set(3, 1280)
cap.set(4, 720)

prev_frame_time = 0

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_count = 0

    # Thumb Detection
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        finger_count += 1

    # Other Finger Detection
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1
    
    return finger_count

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    finger_count = 0  # Ensure it's initialized

    # Object Detection
    results = model(img, stream=True)
    object_count = 0  # Initialize object count
    if not success:
        print("Failed to capture image from webcam.")
        break
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimal places
            
            if conf > 0.4:  # Only process objects with confidence > 0.4
                cls = int(box.cls[0])
                label = class_names[cls]  # Get class name directly
                object_count += 1  # Count only filtered objects
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cv2.putText(img, f'{label} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hand Detection
    hand_results = hands.process(img_rgb)

    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            finger_count += count_fingers(handLms)

            # Instead of hands.min_detection_confidence, try checking if hand detection was successful
            hand_confidence = 0.7  # Using the threshold you defined initially

            cv2.putText(img, f'Hand Confidence: {hand_confidence:.2f}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Combine counts and show
    total_count = object_count + finger_count
    cv2.putText(img, f'Objects: {object_count} | Fingers: {finger_count} | Total: {total_count}',
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # FPS Calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display
    cv2.imshow("Object + Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()