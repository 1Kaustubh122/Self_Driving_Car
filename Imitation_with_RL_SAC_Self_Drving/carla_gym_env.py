import os
import cv2
import time
import torch
import carla
import logging
import numpy as np
import gymnasium as gym
from threading import Lock
import torch.nn.functional as F
import torchvision.transforms as T
from gymnasium.spaces import Dict, Box 
import torchvision.transforms as transforms
from Imitation_Learning_RL.process_frame import process_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pre_process(rgb_image, seg_image, transform=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Resize
    rgb_image = cv2.resize(rgb_image, (128, 128))
    seg_image = cv2.resize(seg_image, (128, 128))

    if transform:
        rgb_tensor = transform(rgb_image)
    else:
        rgb_tensor = transforms.ToTensor()(rgb_image).to(device)  # (3, 128, 128)

    # Masks
    lane_mask = np.all(seg_image == [0, 255, 0], axis=2).astype(np.uint8)
    obs_mask  = np.all(seg_image == [255, 0, 0], axis=2).astype(np.uint8)
    seg_tensor = torch.tensor(np.stack([lane_mask, obs_mask], axis=0), dtype=torch.float32).to(device)  # (2, 128, 128)

    # If you interpolate, do it on a 4D tensor, then squeeze:
    if seg_tensor.shape[1:] != rgb_tensor.shape[1:]:
        seg_tensor = F.interpolate(
            seg_tensor.unsqueeze(0),  # (1, 2, H, W)
            size=rgb_tensor.shape[1:],
            mode='nearest'
        ).squeeze(0)  # (2, H, W)

    # Now both are (C, H, W)
    input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)  # (5, 128, 128)
    return input_tensor

# def pre_process(rgb_image, seg_image, transform, device="cuda" if torch.cuda.is_available() else "cpu"):

#     # Resizing image to save GPU memory, it sucks
#     rgb_image = cv2.resize(rgb_image, (128, 128))
#     seg_image = cv2.resize(seg_image, (128, 128))
    
#     if transform:
#         rgb_tensor = transform(rgb_image)
#     else:
#         rgb_tensor = transforms.ToTensor()(rgb_image).to(device)

#     seg_image = np.array(seg_image)
    
#     lane_mask = np.all(seg_image == [0, 255, 0], axis=2).astype(np.uint8)
#     obs_mask = np.all(seg_image == [255, 0, 0], axis=2).astype(np.uint8)
    
#     seg_tensor = torch.tensor(np.stack([lane_mask, obs_mask], axis=0), dtype=torch.float32).to(device)
    
#     if seg_tensor.shape[1:] != rgb_tensor.shape[1:]:
#         seg_tensor = F.interpolate(
#             seg_tensor.unsqueeze(0),
#             size=rgb_tensor.shape[1:],
#             mode='nearest'
#         ).squeeze(0)
    
#     input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)
    
#     return input_tensor


class CarlaEnv(gym.Env):
    # def __init__(self, host='localhost', port=2000):
    def __init__(self, host, port):

        self.rgb_count = 0
        self.seg_count = 0
        super(CarlaEnv, self).__init__()
        
        self.client_ = carla.Client(host, port)
        self.client_.set_timeout(10.0)
        self.world_ = self.client_.get_world()
        self.blueprint_library_ = self.world_.get_blueprint_library()
        self.map_ = self.world_.get_map()
        self.spawn_points_ = self.map_.get_spawn_points()
        self.start_point_ = self.spawn_points_[0]


        self.vehicle_bp = self.blueprint_library_.find('vehicle.volkswagen.t2_2021')
        # self.vehicle_bp = self.blueprint_library_.find('vehicle.mini.cooper')
        self.vehicle_ = None

        camera_location = carla.Location(
            x=4.442184 / 2.9 , 
            y=0,
            z=2.2
        )
        
        self.camera_bp_ = self.blueprint_library_.find('sensor.camera.rgb')
        self.camera_bp_.set_attribute('image_size_x', '128')
        self.camera_bp_.set_attribute('image_size_y', '128')
        self.camera_sensor = None
        self.camera_transform = carla.Transform(camera_location)
        self.rgb_image = None
        self.seg_image = None
        self.image_lock = Lock()


        self.collision_lock = Lock()
        self.collision_bp = self.blueprint_library_.find("sensor.other.collision")
        self.collision_transform = carla.Transform()
        self.collision_sensor = None
        self.collision = None

        self.lane_lock = Lock()
        self.lane_inv_bp = self.blueprint_library_.find("sensor.other.lane_invasion")
        self.lane_inv_transform = carla.Transform()
        self.lane_invasion_sensor = None
        self.lane_invasion = False

        self.wp_draw_lock = Lock()


                        # Steer, Throttle, Brake
        self.action_space = gym.spaces.Box(
            low= np.array([-1.0, 0.0, 0.0]),
            high= np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
                                # 5 Channel image (244 x 244)
                                # [speed, steer, throttle, brake] Max and min values (high and low)
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(5, 128, 128), dtype=np.uint8),
            "state": Box(
                low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -np.pi]),
                high=np.array([100.0, 1.0, 1.0, 1.0, 100.0, np.pi]), 
                shape=(6,), 
                dtype=np.float32
            )
        })
        
       
        self.stuck_start_time = None
        self.is_stuck = False
        
        self.max_steps = 6000
        self.episode_step = 0
        self.previous_location = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        
        
        self.steer_lock_start_time = None
        self.is_steer_locked = False
        self.prev_steer = 0.0

        
    # Sensors Callbacks
    def image_callback(self, data):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
        with self.image_lock:
            self.rgb_image = array
            self.rgb_count += 1
            self.seg_image = process_frame(self.rgb_image)
            self.seg_count += 1
            

    def collision_callback(self, data):
        with self.collision_lock:
            self.collision = True

    def lane_invasion_callback(self, data):
        with self.lane_lock:
            self.lane_invasion = True

    def draw_waypoints(self, waypoints, life_time=100.0):
        with self.wp_draw_lock:
            for wp in waypoints:
                self.world_.debug.draw_point(
                    wp.transform.location + carla.Location(z=0.5),
                    size=0.1,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=life_time
                )
                self.world_.debug.draw_arrow(
                    wp.transform.location,
                    wp.transform.location + wp.transform.get_forward_vector() * 2,
                    thickness=0.1,
                    arrow_size=0.2,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=life_time
                )
    
    def destroy_actors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None


        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision = None
            self.collision_sensor = None
        
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.stop()
            self.lane_invasion_sensor.destroy()
            self.lane_invasion_sensor = None
            self.lane_invasion = None
        
        if self.vehicle_ is not None:
            self.vehicle_.destroy()
            self.vehicle_ = None

        try:
            for actor in self.world_.get_actors().filter('*vehicle*'):
                actor.destroy()
            for sensor in self.world_.get_actors().filter('*sensor*'):
                sensor.destroy()
        except:
            pass
        
    def reset(self, seed = None, options = None):

        self.episode_step = 0
        self.stuck_start_time = None


        if seed is not None:
            np.random.seed(seed)

        # destroy all actors
        self.destroy_actors()

        self.destroy_actors()
        
        self.vehicle_ = self.world_.try_spawn_actor(self.vehicle_bp, self.start_point_)
        
        self.camera_sensor = self.world_.spawn_actor(self.camera_bp_, self.camera_transform, attach_to=self.vehicle_)
        self.camera_sensor.listen(lambda data: self.image_callback(data))
        
        self.collision_sensor = self.world_.spawn_actor(self.collision_bp, self.collision_transform, attach_to=self.vehicle_)
        self.collision_sensor.listen(lambda data: self.collision_callback(data))
        
        self.lane_invasion_sensor = self.world_.spawn_actor(self.lane_inv_bp, self.lane_inv_transform, attach_to=self.vehicle_)
        self.lane_invasion_sensor.listen(lambda event: self.lane_invasion_callback(event))
        

        
        self.stuck_start_time = None
        self.is_stuck = False
    
        # try:
        #     cv2.destroyAllWindows()
        # except:
        #     pass
                
        spawn_point = self.world_.get_map().get_spawn_points()[0]
        
    
        
        self.episode_step = 0
        self.collision = False
        self.lane_invasion = False
        # print(self.vehicle_)
        self.previous_location = self.vehicle_.get_location()
        self.prev_steer = 0.0
        
        map_ = self.world_.get_map()
        self.waypoints = [map_.get_waypoint(spawn_point.location)]
        for _ in range(50):
            next_wp = self.waypoints[-1].next(5.0)
            self.waypoints.append(next_wp[0] if next_wp else self.waypoints[-1])
        self.current_waypoint_idx = 0
        self.draw_waypoints(self.waypoints)

        
        return self.get_observation(), {}
    
    

    # def get_observation(self):
    #     while self.rgb_image is None or self.seg_image is None:
    #         time.sleep(0.001)
    #         # self.world_.tick()
        
    #     with self.image_lock:
    #         if self.rgb_image is None or self.seg_image is None:
    #             raise RuntimeError("Image not found! Either seg or rgb!")
            
    #         image_obs = pre_process(self.rgb_image.copy(), self.seg_image.copy(), transform = None, device="cuda" if torch.cuda.is_available() else "cpu").cpu().numpy()
            
    #     velocity = self.vehicle_.get_velocity()
    #     speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    #     control = self.vehicle_.get_control()
    #     steering = control.steer
    #     throttle = control.throttle
    #     brake = control.brake

    #     current_location = self.vehicle_.get_location()
    #     if self.current_waypoint_idx < len(self.waypoints):
    #         wp = self.waypoints[self.current_waypoint_idx]
    #         distance = current_location.distance(wp.transform.location)
    #         forward_vec = wp.transform.get_forward_vector()
    #         vehicle_vec = self.vehicle_.get_transform().get_forward_vector()
    #         angle = np.arccos(np.clip(np.dot([forward_vec.x, forward_vec.y], [vehicle_vec.x, vehicle_vec.y]), -1.0, 1.0))
    #     else:
    #         distance, angle = 0.0, 0.0

    #     state_obs = np.array([speed, steering, throttle, brake, distance, angle], dtype=np.float32)
    #     state_obs = np.clip(
    #         state_obs,
    #         a_min=[0.0, -1.0, 0.0, 0.0, 0.0, -np.pi],
    #         a_max=[100.0, 1.0, 1.0, 1.0, 100.0, np.pi]
    #     )

    #     return {"image": image_obs.astype(np.uint8), "state": state_obs}
    

    def get_observation(self, timeout=2.0):
        start_time = time.time()
        while True:
            with self.image_lock:
                rgb_ready = self.rgb_image is not None
                seg_ready = self.seg_image is not None
            if rgb_ready and seg_ready:
                break
            if time.time() - start_time > timeout:
                logging.warning("Sensor data timeout. Resetting environment.")
                self.reset()  # Or raise an exception to trigger a full reset in your RL loop
                # Optionally, return a blank observation or last valid one
                return self.get_observation(timeout)
            time.sleep(0.005)

        with self.image_lock:
            rgb = self.rgb_image.copy()
            seg = self.seg_image.copy()

        image_obs = pre_process(rgb, seg, transform=None, device="cuda" if torch.cuda.is_available() else "cpu").cpu().numpy()

        velocity = self.vehicle_.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        control = self.vehicle_.get_control()
        steering = control.steer
        throttle = control.throttle
        brake = control.brake

        current_location = self.vehicle_.get_location()
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            distance = current_location.distance(wp.transform.location)
            forward_vec = wp.transform.get_forward_vector()
            vehicle_vec = self.vehicle_.get_transform().get_forward_vector()
            angle = np.arccos(np.clip(np.dot([forward_vec.x, forward_vec.y], [vehicle_vec.x, vehicle_vec.y]), -1.0, 1.0))
        else:
            distance, angle = 0.0, 0.0

        state_obs = np.array([speed, steering, throttle, brake, distance, angle], dtype=np.float32)
        state_obs = np.clip(
            state_obs,
            a_min=[0.0, -1.0, 0.0, 0.0, 0.0, -np.pi],
            a_max=[100.0, 1.0, 1.0, 1.0, 100.0, np.pi]
        )

        return {"image": image_obs.astype(np.uint8), "state": state_obs}

 

    # def step(self, action):

    #     ##DEBUG
    #     start_time = time.time()

    #     # print(action)
        
    #     # if self.lane_invasion:
    #     #     print("Lane Invaded")
    #     # if self.collision:
    #     #     print("collided")

    #     steer_, throttle_, brake_  = action

    #     # brake_ = 1.0 if brake_ > 0.65 else 0.0
    #     # brake = max(min(brake_, 1.0), 0.0)
        
    #     self.vehicle_.apply_control(carla.VehicleControl(
    #         throttle = float(throttle_),
    #         # throttle = 1.0,
    #         steer = float(steer_),
    #         # brake = 0, 
    #         brake = 1.0 if brake_ > 0.8 else 0, 
    #         hand_brake=False
    #     ))
        
    #     # self.world_.tick()
     
        
    #     obs = self.get_observation()
    #     reward = self.compute_reward()
    #     self.episode_step += 1
        
    #     done = (
    #         self.collision or
    #         self.lane_invasion or    
    #         self.episode_step >= self.max_steps or
    #         self.current_waypoint_idx >= len(self.waypoints) - 1
    #     )
        
    #     truncated = self.episode_step >= self.max_steps
        
          
    #     # if self.client_.get_world().get_settings().synchronous_mode:
    #     #     combined_image = np.hstack((self.seg_image, self.rgb_image))
    #     #     cv2.imshow(f"Camera_Port_{self.client_.get_world().get_settings().port}", combined_image)
    #     #     cv2.waitKey(1)
        
    #     # step_time = self.episode_step >= self.max_steps
    #     # step_time = time.time() - start_time
    #     # logger.info(f"Step time: {step_time:.3f}s")
    #     logging.info(f"RGB callbacks: {self.rgb_count}, Seg callbacks: {self.seg_count}")
    #     return obs, reward, done or truncated, truncated, {}
    


    def step(self, action):
        start_time = time.time()

        steer_, throttle_, brake_ = action

        self.vehicle_.apply_control(carla.VehicleControl(
            throttle = float(throttle_),
            steer = float(steer_),
            brake = 1.0 if brake_ > 0.8 else 0.0,
            hand_brake=False
        ))

        obs = self.get_observation()
        reward = self.compute_reward()
        self.episode_step += 1

        # Check if stuck for too long
        current_time = self.world_.get_snapshot().timestamp.elapsed_seconds
        velocity = self.vehicle_.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        stuck_duration_threshold = 2.0  # seconds
        if not hasattr(self, 'stuck_start_time'):
            self.stuck_start_time = current_time

        is_stuck_too_long = False
        if speed < 0.2:
            if (current_time - self.stuck_start_time) > stuck_duration_threshold:
                is_stuck_too_long = True
        else:
            self.stuck_start_time = current_time

        done = (
            self.collision or
            # self.lane_invasion or    
            self.episode_step >= self.max_steps or
            self.current_waypoint_idx >= len(self.waypoints) - 1 or
            is_stuck_too_long
        )

        truncated = self.episode_step >= self.max_steps

        # logging.info(f"RGB callbacks: {self.rgb_count}, Seg callbacks: {self.seg_count}")
        
        return obs, reward, done, truncated, {}




    # def compute_reward(self):
    #     reward = 0
        
    #     current_location = self.vehicle_.get_location()
    #     control = self.vehicle_.get_control()
    #     current_time = self.world_.get_snapshot().timestamp.elapsed_seconds
        
    #     stuck_duration_threshold = 2.0  # 2 seconds
            
    #     while self.current_waypoint_idx < len(self.waypoints) - 1:
    #         wp = self.waypoints[self.current_waypoint_idx]
    #         if current_location.distance(wp.transform.location) < 2.0:
    #             self.current_waypoint_idx += 1
    #             reward += 10.0
    #         else:
    #             break
                
    #     if self.collision:
    #         reward -= 100.0
    #     if self.lane_invasion:
    #         reward -= 100.0
        
    #     map_ = self.world_.get_map()
    #     wp = map_.get_waypoint(current_location)
        
    #     if self.current_waypoint_idx < len(self.waypoints):
    #         current_wp = self.waypoints[self.current_waypoint_idx]
    #         prev_dist = self.previous_location.distance(current_wp.transform.location)
    #         curr_dist = current_location.distance(current_wp.transform.location)
    #         reward += max(prev_dist - curr_dist, 0) * 0.1
    #     self.previous_location = current_location
        
    #     velocity = self.vehicle_.get_velocity()
    #     speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    #     reward += speed * 0.1

    #     steer_change = abs(control.steer - self.prev_steer)
    #     reward -= steer_change * 0.5
    #     self.prev_steer = control.steer

    #     reward -= 0.1
        
    #     if not hasattr(self, 'stuck_start_time'):
    #         self.stuck_start_time = current_time
    #         self.is_stuck = False
        
    #     if speed < 0.2 and control.throttle > 0.3 and control.brake < 0.1:
    #         if not self.is_stuck:
    #             self.is_stuck = True
    #             self.stuck_start_time = current_time
    #         elif current_time - self.stuck_start_time > stuck_duration_threshold:
    #             reward -= 100.0  # Penalty for being stuck
    #             self.is_stuck = False
    #             self.stuck_start_time = current_time
    #             # print(f"Stuck detected! Duration: {current_time - self.stuck_start_time:.2f}s, Speed: {speed:.2f}, Throttle: {control.throttle:.2f}")
    #     else:
    #         self.is_stuck = False
    #         self.stuck_start_time = current_time
        
    #     steer_lock_threshold = 0.8
    #     steer_lock_duration_threshold = 4.0  # seconds

    #     if abs(control.steer) > steer_lock_threshold:
    #         reward -= 1.0
    #         if not self.is_steer_locked:
    #             self.is_steer_locked = True
    #             self.steer_lock_start_time = current_time
    #         elif current_time - self.steer_lock_start_time > steer_lock_duration_threshold:
    #             reward -= 50.0 
    #             self.is_steer_locked = False
    #             self.steer_lock_start_time = current_time 
    #     else:
    #         self.is_steer_locked = False
    #         self.steer_lock_start_time = current_time


    #     if speed < 0.2:
    #         reward -= 100.0


    #     # reward = np.clip(reward, -100, 50.0)
        
    #     return reward

    def compute_reward(self):
        reward = 0

        current_location = self.vehicle_.get_location()
        control = self.vehicle_.get_control()
        current_time = self.world_.get_snapshot().timestamp.elapsed_seconds

        # --- Parameters ---
        stuck_duration_threshold = 2.0  # seconds
        steer_lock_threshold = 0.8
        steer_lock_duration_threshold = 4.0  # seconds
        min_speed_threshold = 1.0  #

        # --- Waypoint Progress Reward ---
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_waypoint_idx]
            if current_location.distance(wp.transform.location) < 2.0:
                self.current_waypoint_idx += 1
                reward += 10.0
            else:
                break

        # --- Collision and Lane Invasion Penalties ---
        if self.collision:
            reward -= 100.0
        if self.lane_invasion:
            reward -= 100.0

        # --- Progress Towards Current Waypoint ---
        if self.current_waypoint_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_idx]
            prev_dist = self.previous_location.distance(current_wp.transform.location)
            curr_dist = current_location.distance(current_wp.transform.location)
            delta_progress = prev_dist - curr_dist
            if delta_progress > 0:
                reward += delta_progress * 0.1
        self.previous_location = current_location

        # --- Speed and Direction ---
        velocity = self.vehicle_.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        # Penalize slow speed (< 1.0 m/s)
        if speed < min_speed_threshold:
            reward -= 1.0
        else:
            reward += speed * 0.1  

        # --- Steering Stability ---
        steer_change = abs(control.steer - self.prev_steer)
        reward -= steer_change * 0.5
        self.prev_steer = control.steer

        # Penalize constant high steering (tight loop tricking)
        if abs(control.steer) > steer_lock_threshold:
            reward -= 1.0
            if not self.is_steer_locked:
                self.is_steer_locked = True
                self.steer_lock_start_time = current_time
            elif current_time - self.steer_lock_start_time > steer_lock_duration_threshold:
                reward -= 50.0
                self.is_steer_locked = False
                self.steer_lock_start_time = current_time
        else:
            self.is_steer_locked = False
            self.steer_lock_start_time = current_time

        # --- Stuck Detection ---
        if not hasattr(self, 'stuck_start_time'):
            self.stuck_start_time = current_time
            self.is_stuck = False

        if speed < 0.2 and control.throttle > 0.3 and control.brake < 0.1:
            if not self.is_stuck:
                self.is_stuck = True
                self.stuck_start_time = current_time
            elif current_time - self.stuck_start_time > stuck_duration_threshold:
                reward -= 100.0  # Heavy penalty for stuck
                self.is_stuck = False
                self.stuck_start_time = current_time
        else:
            self.is_stuck = False
            self.stuck_start_time = current_time

        # --- Constant Small Penalty ---
        reward -= 0.1

        # --- Clip final reward ---
        reward = np.clip(reward, -100.0, 50.0)

        # --- Rolling log of last 100 rewards ---
        log_path = "rewards_log.txt"

        # Read existing entries
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        # Prepend new reward (as a string)
        lines.insert(0, f"{reward}\n")

        # Keep only the most recent 100 entries
        lines = lines[:100]

        # Write back
        with open(log_path, "w") as f:
            f.writelines(lines)

        # logging.info(f"Reward: {reward}")

        return reward
