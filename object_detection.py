import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can use 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
model = YOLO('yolov8n.pt')  # 'n' is for nano — smallest & fastest, replace with 'yolov8s.pt' for better accuracy

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
