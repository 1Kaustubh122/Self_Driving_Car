import carla
import cv2
import numpy as np
import zmq
import time
import threading

# ZMQ Publisher Setup
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

# Connect to CARLA
client = carla.Client("localhost", 2000)
world = client.get_world()

# Find or spawn vehicle
vehicle = None
for actor in world.get_actors():
    if "vehicle" in actor.type_id:
        vehicle = actor
        break

if not vehicle:
    print("No vehicle found, spawning one...")
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.volkswagen.t2_2021")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Camera setup
camera_bp = world.get_blueprint_library().filter("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("fov", "110")

# Attach camera to the vehicle
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))  # Move it to the front
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Function to send images
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    array = array[:, :, :3]  # Remove alpha channel

    _, buffer = cv2.imencode(".jpg", array)
    socket.send(buffer.tobytes())  # Send image via ZMQ

# Run the camera stream in a separate thread
camera.listen(lambda image: threading.Thread(target=camera_callback, args=(image,)).start())

print("Camera streaming to Segmentation Model...")
try:
    while True:
        time.sleep(0.1)  # Keep script running
except KeyboardInterrupt:
    print("Stopping...")
    camera.destroy()
    vehicle.destroy()
