import carla
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the latest YOLO segmentation model
model = YOLO("yolov8s-seg.pt")

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get all actors
actors = world.get_actors()

# Find a vehicle
vehicle = None
for actor in actors:
    if "vehicle" in actor.type_id:
        vehicle = actor
        break

if vehicle is None:
    print("No vehicle found.")
    exit()

print(f"Found vehicle: {vehicle.type_id}")

# Remove existing cameras to avoid duplicates
for actor in actors:
    if "sensor.camera.rgb" in actor.type_id:
        actor.destroy()

# Define camera position (front of vehicle)
camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("fov", "110")  # Wide FOV

camera_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

print(f"Spawned front-facing camera: {camera.id}")

# Image processing function
def process_image(image):
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))  # BGRA format
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)  # Convert to BGR

    # Run YOLO segmentation
    results = model(img_rgb)[0]

    # Create a black mask
    mask = np.zeros_like(img_rgb)

    # Draw segmentation masks
    for box, seg in zip(results.boxes, results.masks.xy):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        conf = float(box.conf[0])  # Confidence score
        cls = int(box.cls[0])  # Class ID
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw bounding box on original image
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw segmentation mask on black background
        polygon = np.array(seg, np.int32)
        cv2.fillPoly(mask, [polygon], (0, 255, 0))

    # Show images
    cv2.imshow("RGB Image", img_rgb)
    cv2.imshow("Segmentation Mask", mask)

    # Stop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.stop()
        cv2.destroyAllWindows()
        exit()

# Attach callback function to the camera
camera.listen(lambda image: process_image(image))

print("Running YOLO segmentation live. Press 'q' to exit.")

try:
    while True:
        world.wait_for_tick()
except KeyboardInterrupt:
    print("Stopping...")
    camera.stop()
    cv2.destroyAllWindows()
