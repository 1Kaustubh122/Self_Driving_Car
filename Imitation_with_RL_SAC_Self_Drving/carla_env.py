import carla
import numpy as np
import gymnasium as gym

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        
        self.client_ = carla.Client('localhost', 2000)
        self.client_.set_timeout(10.0)
        self.world_ = self.client_.get_world()
        self.blueprint_library_ = self.world_.get_blueprint_library()

                        # Steer, Throttle, Brake
        self.action_space_ = gym.spaces.Box(
            low= np.array([-1.0, 0.0, 0.0]),
            high= np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=0, high=244, shape=(5, 244, 244), dtype=np.float32
        )
        
        self.vehicle_ = None
        self.rgb_camera_ = None
        self.seg_camera_ = None
        self.rgb_images_ = []
        self.seg_images_ = []
        
    def step(self, action):
        
        steer_, throttle_, brake_  = action
        brake_ = 1.0 if brake_ > 0.65 else 0.0
        
        self.vehicle_.apply_control(carla.VehicleControl(
            throttle = float(throttle_),
            steer = float(steer_),
            brake = float(brake_)
        ))
        
        reward = 0.0
        terminated = False
        truncated = False
        
        velocity_ = self.vehicle_.get_velocity()
        speed = np.sqrt(velocity_.x**2 + velocity_.y**2)
        reward += speed * 0.05
        
        waypoint_ = self.world_.get_map().get_waypoint(self.vehicle_.get_location())
        lane_deviation = np.abs(waypoint_.transform.location.x - self.vehicle_.get_location().x)
        reward -= lane_deviation * 0.5
        
        cp
        return 
    
    
    
    
    
    
    
    
    
    
import gymnasium as gym
import carla
import numpy as np

# Custom CARLA Environment for RL
class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Action space: [steer, throttle, brake]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: 5-channel image (224x224)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(5, 224, 224), dtype=np.float32
        )
        
        self.vehicle = None
        self.rgb_camera = None
        self.seg_camera = None
        self.rgb_images = []
        self.seg_images = []
    
    def step(self, action):
        steer, throttle, brake = action
        brake = 1.0 if brake > 0.5 else 0.0  # Binary brake
        
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        ))
        
        # Reward function
        reward = 0.0
        terminated = False
        truncated = False
        
        # Encourage progress
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        reward += speed * 0.05  # Reward per m/s
        
        # Lane keeping
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        lane_deviation = np.abs(waypoint.transform.location.x - self.vehicle.get_location().x)
        reward -= lane_deviation * 0.5  # Penalize deviation from lane center
        
        # Collision penalty
        collision_sensor = self.vehicle.get_world().get_spectator()  # Simplified, assumes collision detection
        if hasattr(self.vehicle, 'collision_detected') and self.vehicle.collision_detected:
            reward -= 100.0
            terminated = True
        
        # Speed regulation (target 20-40 km/h, ~5.5-11 m/s)
        if 5.5 < speed < 11.0:
            reward += 0.2
        elif speed > 11.0:
            reward -= 0.5  # Discourage overspeeding
        
        # Smoothness penalty
        prev_action = getattr(self, '_prev_action', action)
        action_diff = np.abs(action - prev_action)
        reward -= action_diff.sum() * 0.1  # Penalize abrupt changes
        self._prev_action = action
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}