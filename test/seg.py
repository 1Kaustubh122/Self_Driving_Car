import cv2
import numpy as np
import zmq
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n-seg.pt")


# Set up ZMQ socket to receive images
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
print("Segmentation Client: Listening on tcp://localhost:5555")

while True:
    try:
        # Receive image
        buffer = socket.recv()
        img_array = np.frombuffer(buffer, dtype=np.uint8)
        img_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Run YOLO segmentation
        results = model(img_rgb)[0]

        # Create a mask (initialize empty)
        mask = np.zeros_like(img_rgb)

        if results.masks is not None:
            for box, seg in zip(results.boxes, results.masks.xy):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0])  # Confidence
                cls = int(box.cls[0])  # Class ID
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw segmentation mask
                polygon = np.array(seg, np.int32)
                cv2.fillPoly(mask, [polygon], (0, 255, 0))

        # Use cv2.WINDOW_NORMAL to avoid input blocking
        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Segmentation Mask", cv2.WINDOW_NORMAL)

        # Show images (non-blocking)
        cv2.imshow("RGB Image", img_rgb)
        cv2.imshow("Segmentation Mask", mask)
        cv2.waitKey(1)  # Prevents blocking

    except KeyboardInterrupt:
        print("Stopping...")
        break

cv2.destroyAllWindows()