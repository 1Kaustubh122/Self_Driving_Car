import cv2
import torch
import carla
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
import torchvision.transforms as transforms
from Imitation_Learning_RL.process_frame import process_frame


def pre_process(rgb_image, seg_image, transform):
    
    if transform:
        rgb_tensor = transform(rgb_image)
    else:
        rgb_tensor = transforms.ToTensor()(rgb_image)
        
    seg_image = np.array(seg_image)
    
    lane_mask = np.all(seg_image == [0, 255, 0], axis=2).astype(np.uint8)
    obs_mask = np.all(seg_image == [255, 0, 0], axis=2).astype(np.uint8)
    
    seg_tensor = torch.tensor(np.stack([lane_mask, obs_mask], axis=0), dtype=torch.float32)
    
    if seg_tensor.shape[1:] != rgb_tensor.shape[1:]:
        seg_tensor = F.interpolate(
            seg_tensor.unsqueeze(0),
            size=rgb_tensor.shape[1:],
            mode='nearest'
        ).squeeze(0)
    
    input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)
    
    return input_tensor

class CarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000):
        super(CarlaEnv, self).__init__()
        
        self.client_ = carla.Client(host, port)
        self.client_.set_timeout(10.0)
        self.world_ = self.client_.get_world()
        self.blueprint_library_ = self.world_.get_blueprint_library()

                        # Steer, Throttle, Brake
        self.action_space = gym.spaces.Box(
            low= np.array([-1.0, 0.0, 0.0]),
            high= np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
                                # 5 Channel image (244 x 244)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255,
            shape=(5, 360, 640), 
            dtype=np.uint8
        )
        
        
        self.vehicle_ = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.stuck_start_time = None
        self.is_stuck = False
        
        self.max_steps = 1000
        self.episode_step = 0
        self.previous_location = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        
        self.rgb_image = None
        self.seg_image = None
        self.collision = False
        self.lane_invasion = False
        self.steer_lock_start_time = None
        self.is_steer_locked = False

        
    # Sensors Callbacks
    def image_callback(self, data):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
        self.rgb_image = array
        self.seg_image = process_frame(self.rgb_image)

    def draw_waypoints(self, waypoints, life_time=100.0):
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


    def collision_callback(self, data):
        self.collision = True

    def lane_invasion_callback(self, data):
        self.lane_invasion = True
    
    def reset(self, seed = None, options = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.stuck_start_time = None
        self.is_stuck = False
    
        # print(self.vehicle_)
        # if self.vehicle_ is not None:
        # for actor in self.world_.get_actors().find("vehicle.volkswagen.t2_2021"):
        #     print(actor)
        #     actor.destroy()
        
        for actor in self.world_.get_actors().filter('vehicle.volkswagen.t2_2021'):
            actor.destroy()
        for sensor in self.world_.get_actors().filter('*sensor*'):
            sensor.destroy()
                
        # for sensor in [self.camera_sensor, self.collision_sensor, self.lane_invasion_sensor]:
        #     if sensor is not None:
        #         sensor.destroy()
        try:
            cv2.destroyAllWindows()
        except:
            pass
                
        spawn_point = self.world_.get_map().get_spawn_points()[0]
                
        vehicle_bp_ = self.blueprint_library_.find('vehicle.volkswagen.t2_2021')
        camera_bp_ = self.blueprint_library_.find('sensor.camera.rgb')
        collision_bp = self.blueprint_library_.find("sensor.other.collision")
        lane_inv_bp = self.blueprint_library_.find("sensor.other.lane_invasion")
        
        camera_bp_.set_attribute('image_size_x', '640')
        camera_bp_.set_attribute('image_size_y', '360')
        
        self.vehicle_ = self.world_.try_spawn_actor(vehicle_bp_, spawn_point)
        
        if self.vehicle_ is None:
            raise RuntimeError("Vehicle failed to spawn. Likely due to occupied spawn point or invalid config.")

        camera_location = carla.Location(x=4.442184 / 2.9, y=0, z=2.2)
        
        camera_transform = carla.Transform(camera_location)
        collision_transform = carla.Transform()
        lane_inv_transform = carla.Transform()
        
        self.camera_sensor = self.world_.spawn_actor(camera_bp_, camera_transform, attach_to=self.vehicle_)
        self.collision_sensor = self.world_.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle_)
        self.lane_invasion_sensor = self.world_.spawn_actor(lane_inv_bp, lane_inv_transform, attach_to=self.vehicle_)
        
        self.camera_sensor.listen(lambda data: self.image_callback(data))
        self.collision_sensor.listen(lambda data: self.collision_callback(data))
        self.lane_invasion_sensor.listen(lambda data: self.lane_invasion_callback(data))
        
        self.episode_step = 0
        self.collision = False
        self.lane_invasion = False
        self.previous_location = self.vehicle_.get_location()
        
        map_ = self.world_.get_map()
        self.waypoints = [map_.get_waypoint(spawn_point.location)]
        for _ in range(50):
            next_wp = self.waypoints[-1].next(5.0)
            self.waypoints.append(next_wp[0] if next_wp else self.waypoints[-1])
        self.current_waypoint_idx = 0
        self.draw_waypoints(self.waypoints)

        
        return self.get_observation(), {}
    
    def get_observation(self):
        while self.rgb_image is None or self.seg_image is None:
            self.world_.tick()
        
        obs = pre_process(self.rgb_image, self.seg_image, transform = None).numpy()
        return obs.astype(np.uint8)

    
    def step(self, action):
   
        # print(action)
        
        # if self.lane_invasion:
        #     print("Lane Invaded")
        # if self.collision:
        #     print("collided")
        steer_, throttle_, brake_  = action
        # brake_ = 1.0 if brake_ > 0.65 else 0.0
        # brake = max(min(brake_, 1.0), 0.0)
        
        self.vehicle_.apply_control(carla.VehicleControl(
            throttle = float(throttle_),
            # throttle = 1.0,
            steer = float(steer_),
            # brake = 0, 
            brake = 1.0 if brake_ > 0.8 else 0, 
            hand_brake=False
        ))
        
        self.world_.tick()
     
        
        obs = self.get_observation()
        reward = self.compute_reward()
        self.episode_step += 1
        
        done = (
            self.collision or
            self.lane_invasion or    
            self.episode_step >= self.max_steps or
            self.current_waypoint_idx >= len(self.waypoints) - 1
        )
        
        truncated = self.episode_step >= self.max_steps
        
          
        combined_image = np.hstack((self.seg_image, self.rgb_image))
        cv2.imshow("Camera", combined_image)
        cv2.waitKey(1)
        
        
        return obs, reward, done or truncated, truncated, {}
        
    def compute_reward(self):
        reward = 0
        
        current_location = self.vehicle_.get_location()
        control = self.vehicle_.get_control()
        current_time = self.world_.get_snapshot().timestamp.elapsed_seconds
        
        stuck_duration_threshold = 2.0  # 2 seconds
            
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_waypoint_idx]
            if current_location.distance(wp.transform.location) < 2.0:
                self.current_waypoint_idx += 1
                reward += 10.0
            else:
                break
                
        if self.collision:
            reward -= 100.0
        if self.lane_invasion:
            reward -= 100.0
        
        map_ = self.world_.get_map()
        wp = map_.get_waypoint(current_location)
        
        if self.current_waypoint_idx < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_idx]
            prev_dist = self.previous_location.distance(current_wp.transform.location)
            curr_dist = current_location.distance(current_wp.transform.location)
            reward += max(prev_dist - curr_dist, 0) * 0.1
        self.previous_location = current_location
        
        velocity = self.vehicle_.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        reward += speed * 0.1
        
        if not hasattr(self, 'stuck_start_time'):
            self.stuck_start_time = current_time
            self.is_stuck = False
        
        if speed < 0.2 and control.throttle > 0.3 and control.brake < 0.1:
            if not self.is_stuck:
                self.is_stuck = True
                self.stuck_start_time = current_time
            elif current_time - self.stuck_start_time > stuck_duration_threshold:
                reward -= 100.0  # Penalty for being stuck
                self.is_stuck = False
                self.stuck_start_time = current_time
                # print(f"Stuck detected! Duration: {current_time - self.stuck_start_time:.2f}s, Speed: {speed:.2f}, Throttle: {control.throttle:.2f}")
        else:
            self.is_stuck = False
            self.stuck_start_time = current_time
        
        steer_lock_threshold = 0.9
        steer_lock_duration_threshold = 4.0  # seconds

        if abs(control.steer) > steer_lock_threshold:
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


        if speed < 0.2:
            reward -= 100.0


        # reward = np.clip(reward, -100, 50.0)
        
        return reward