import sys
import cv2
import zmq
import time
import math
import carla
import pickle
import random
import numpy as np
from threading import Lock
sys.path.append(r'C:\Users\WORKSTATION2\Downloads\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CarlaServer():
    def __init__(self, port):
        
        # Carla Connection
        self.client_ = carla.Client("localhost", 2000)
        self.client_.set_timeout(10.0)
        self.world_ = self.client_.get_world()
        self.map_ = self.world_.get_map()
        
        # Spawn Point and start point
        self.spawn_points_ = self.map_.get_spawn_points()
        self.start_point_ = self.spawn_points_[0]
    
        self.blueprint_library_ = self.world_.get_blueprint_library()
        
        # Vehicle Setup
        self.vehicle_bp = self.blueprint_library_.find('vehicle.volkswagen.t2_2021')
        self.vehicle_ = None

        # Camera Setup
        self.camera_bp_ = self.blueprint_library_.find('sensor.camera.rgb')
        # self.camera_bp_.set_attribute('image_size_x', '800')
        # self.camera_bp_.set_attribute('image_size_y', '288')
        # self.camera_bp_.set_attribute('fov', '90')
        self.camera_bp_.set_attribute('image_size_x', '640')
        self.camera_bp_.set_attribute('image_size_y', '360')
        # self.camera_bp_.set_attribute('fov', '90')
        
        self.camera_sensor = None
        self.camera_transform = carla.Transform(carla.Location(z=2.5, x=0.65))
        self.image = None
        self.image_lock = Lock()
        
        # GNSS Setup
        self.gnss_lock = Lock()
        self.gnss_bp = self.blueprint_library_.find("sensor.other.gnss")
        self.gnss_transform = carla.Transform(carla.Location(z=2.0))
        self.gnss_sensor = None
        self.gnss_data = {"latitude":0.0, "longitude":0.0}
        
        # Collision Setup
        self.collision_lock = Lock()
        self.collision_bp = self.blueprint_library_.find("sensor.other.collision")
        self.collision_transform = carla.Transform()
        self.collision_sensor = None
        self.collision_event = None
        
        # IMU Setup
        self.imu_lock = Lock()
        self.imu_bp = self.blueprint_library_.find("sensor.other.imu")
        self.imu_transform = carla.Transform(carla.Location(z=2.0))
        self.imu_sensor = None
        self.imu_data = None
        
        # Lane Invasion Setup
        self.lane_lock = Lock()
        self.lane_inv_bp = self.blueprint_library_.find("sensor.other.lane_invasion")
        self.lane_inv_transform = carla.Transform()
        self.lane_inv_sensor = None
        self.lane_inv_data = {"violated": False, "last_event": None}
        
        # XMQ Setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
    def image_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        with self.image_lock:
            self.image = array
    
    def gnss_callback(self, event):
        with self.gnss_lock:
            self.gnss_data = {
                "latitude": event.latitude,
                "longitude": event.longitude
            }
    
    def on_lane_invasion(self, event):
        with self.lane_lock:
            self.lane_inv_data["violated"] = True
            self.lane_inv_data["last_event"] = [str(x) for x in event.crossed_lane_markings]
            
    def euler_to_quaternion(self):
        pitch, roll = 0, 0
        yaw=self.vehicle_.get_transform().rotation.yaw
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return {
            "w": cr * cp * cy + sr * sp * sy,
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy
        }

    def imu_callback(self, event):
        with self.imu_lock:
            
            self.imu_data = {
                "orientation": self.euler_to_quaternion(),
                "orientation_covariance": [0.0]*9,  
                "angular_velocity": {
                    "x": event.gyroscope.x,
                    "y": event.gyroscope.y,
                    "z": event.gyroscope.z
                },
                "angular_velocity_covariance": [0.0]*9,  
                "linear_acceleration": {
                    "x": event.accelerometer.x,
                    "y": event.accelerometer.y,
                    "z": event.accelerometer.z
                },
                "linear_acceleration_covariance": [0.0]*9 
            }

    def get_imu_vector(self):
        with self.imu_lock:
            ori = self.imu_data["orientation"]
            av = self.imu_data["angular_velocity"]
            la = self.imu_data["linear_acceleration"]
            
            return np.array([
                ori["x"], ori["y"], ori["z"], ori["w"],
                av["x"], av["y"], av["z"],
                la["x"], la["y"], la["z"]
            ], dtype=np.float32)

    def collision_callback(self, event):
        with self.collision_lock:
            self.collision_event = {
                "frame": event.frame,
                "intensity": event.normal_impulse.length(),
                "other_actor": event.other_actor.type_id
            }
                        
    def spawn(self):
        self.destroy_actors()
        
        self.vehicle_ = self.world_.try_spawn_actor(self.vehicle_bp, self.start_point_)
        
        self.camera_sensor = self.world_.spawn_actor(self.camera_bp_, self.camera_transform, attach_to=self.vehicle_)
        self.camera_sensor.listen(lambda data: self.image_callback(data))
        
        self.gnss_sensor = self.world_.spawn_actor(self.gnss_bp, self.gnss_transform, attach_to=self.vehicle_)
        self.gnss_sensor.listen(lambda data: self.gnss_callback(data))
        
        self.imu_sensor = self.world_.spawn_actor(self.imu_bp, self.imu_transform, attach_to=self.vehicle_)
        self.imu_sensor.listen(lambda data: self.imu_callback(data))

        self.collision_sensor = self.world_.spawn_actor(self.collision_bp, self.collision_transform, attach_to=self.vehicle_)
        self.collision_sensor.listen(lambda data: self.collision_callback(data))
        
        self.lane_inv_sensor = self.world_.spawn_actor(self.lane_inv_bp, self.lane_inv_transform, attach_to=self.vehicle_)
        self.lane_inv_sensor.listen(lambda event: self.on_lane_invasion(event))
        
        print("Vehicle, camera, collision and gnss spawned")
        
    def destroy_actors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

        if self.gnss_sensor is not None:
            self.gnss_sensor.stop()
            self.gnss_sensor.destroy()
            self.gnss_sensor = None
            
        if self.imu_sensor is not None:
            self.imu_sensor.stop()
            self.imu_sensor.destroy()
            self.imu_sensor = None
            self.imu_data = None

        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_event = None
            self.collision_sensor = None
        
        if self.lane_inv_sensor is not None:
            self.lane_inv_sensor.stop()
            self.lane_inv_sensor.destroy()
            self.lane_inv_sensor = None
            self.lane_inv_data = None
        
        if self.vehicle_ is not None:
            self.vehicle_.destroy()
            self.vehicle_ = None
            

    def get_observation(self):
        with self.image_lock:
            if self.image is None:
                raise RuntimeError("Image missing")
            image_copy = self.image.copy()
            
        with self.gnss_lock:
            if self.gnss_data is None:
                raise RuntimeError("GNSS missing")
            gnss_copy = self.gnss_data.copy()
            
        with self.imu_lock:
            imu_copy = self.imu_data.copy() if self.imu_data else None
        
        with self.collision_lock:
            collision_copy = self.collision_event.copy() if self.collision_event else None
            
        with self.lane_lock:
            lane_copy = self.lane_inv_data.copy() if self.lane_inv_data else None
            self.lane_inv_data = {"violated": False, "last_event": None}
            
        _, encoded_image = cv2.imencode(".jpg", image_copy)
        return {
            "image": encoded_image.tobytes(),
            "gnss": gnss_copy,
            "imu": imu_copy,
            "collision": collision_copy,
            "lane_invaded": lane_copy,
            "timestamp": time.time()
        }
    
    def choose_random_coordinate(self, filename="goal_points.txt"):
        with open(filename, "r") as file:
            lines = file.readlines()
            random_coordinate = random.choice(lines).strip()
            lat, lon = map(float, random_coordinate.split(","))
            return lat, lon
        
    def gnss_to_location(self, lat, lon):
        x = -14418.6285 * lat + 111279.5690 * lon - 3.19252014
        y = -109660.6210 * lat +    4.33686914 * lon + 0.367254638
        return carla.Location(x=x, y=y, z=0.0)

    def run(self):
        self.spawn()
        while True:
            message = self.socket.recv()
            try:
                data = pickle.loads(message)
                if data is None or not isinstance(data, dict) or "image" not in data:
                    # print(data)
                    if data.get("command") == "reset":
                        # print("Command Reset")
                        self.spawn()
                        initial_obs = self.get_observation()
                        start_lat = initial_obs["gnss"]["latitude"]
                        start_lon = initial_obs["gnss"]["longitude"]
                        start_loc = self.gnss_to_location(start_lat, start_lon)
                        goal_lat, goal_lon = self.choose_random_coordinate()
                        goal_loc = self.gnss_to_location(goal_lat, goal_lon)
                        grp = GlobalRoutePlanner(self.map_, sampling_resolution=2.0)
                        route = grp.trace_route(start_loc, goal_loc)
                        route_coords = [[wp.transform.location.x, wp.transform.location.y] for wp, _ in route]
                        for wp, _ in route:
                            self.world_.debug.draw_point(
                                wp.transform.location + carla.Location(z=0.5),
                                size=0.2,
                                color=carla.Color(0, 255, 0),
                                life_time=4.0
                            )
                        payload = {
                            "status": "reset done",
                            "observation": initial_obs,
                            "route": route_coords
                        }
                        self.socket.send(pickle.dumps(payload))
                        continue

                    action = data.get("action")
                    if not isinstance(action, (list, tuple)) or len(action) != 2:
                        raise ValueError("Invalid action received")
                    
                    steer = float(action[0])
                    throttle = 0
                    brake = 0
                    if action[1] > 0:
                        throttle = action[1]
                    else:
                        # brake = abs(action[1])
                        brake = 1

                    self.vehicle_.apply_control(
                        carla.VehicleControl(
                            throttle=throttle, 
                            steer=steer, 
                            brake=brake
                        )
                    )
                    self.world_.tick() 

                    payload = self.get_observation()
                    self.socket.send(pickle.dumps(payload))
                
            except Exception as e:
                print(f"[Error h] {e}")
                self.socket.send(pickle.dumps({"error": str(e)}))

if __name__ == "__main__":
    port=6501
    server = CarlaServer(port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        server.destroy_actors()
        try:
            server.socket.unbind(f"tcp://*:{port}")
            print("Socket unbound.")
        except zmq.ZMQError as e:
            print(f"ZMQ unbind error: {e}")
        server.socket.close()
        server.context.term()
        print("Actors destroyed. Goodbye.")