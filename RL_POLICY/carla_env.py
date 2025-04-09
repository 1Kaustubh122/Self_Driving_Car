import gym
from gym import spaces
import numpy as np
import pickle
import zmq
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from stable_baselines3 import PPO
import random

# Define a simple Location class for distance calculations
class Location:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        """Compute Euclidean distance to another Location."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

# Convert GNSS (latitude, longitude) to Carla coordinates
def gnss_to_location(lat, lon):
    """Convert GNSS coordinates to Carla x, y coordinates."""
    x = -14418.6285 * lat + 111279.5690 * lon - 3.19252014
    y = -109660.6210 * lat + 4.33686914 * lon + 0.367254638
    return Location(x, y)

# Convert quaternion to yaw angle
def quaternion_to_yaw(q):
    """Extract yaw angle from quaternion orientation."""
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

# Normalize angle to [-pi, pi]
def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# Choose a random goal coordinate from a file
def choose_random_coordinate(filename="goal_points.txt"):
    """Select a random (latitude, longitude) pair from a text file."""
    with open(filename, "r") as f:
        lines = f.readlines()
        lat, lon = random.choice(lines).strip().split(",")
        return float(lat), float(lon)

# Custom Gym environment for Carla
class CarlaEnv(gym.Env):
    def __init__(self, port=12345):
        super(CarlaEnv, self).__init__()
        
        # Setup ZMQ socket for communication with Carla server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")
        
        # Setup Carla client to access the map for route planning
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # Define action space: steering [-1, 1], throttle/brake [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Define observation space: [distance_to_waypoint, angle_to_waypoint, speed]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initialize route and waypoint tracking
        self.route = None
        self.wp_index = 0
        
        # Variables for speed and reward computation
        self.previous_location = None
        self.previous_timestamp = None
        self.previous_distance = None
        
        # Episode step limit
        self.max_steps = 1000
        self.step_count = 0

    def reset(self):
        """Reset the environment and return the initial state."""
        # Send reset command to the server
        self.socket.send(pickle.dumps({"command": "reset"}))
        response = pickle.loads(self.socket.recv())
        if response["status"] != "reset done":
            raise RuntimeError("Server reset failed")

        # Send a dummy action to get the initial observation
        self.socket.send(pickle.dumps({"action": [0.0, 0.0]}))
        obs = pickle.loads(self.socket.recv())
        
        # Get starting position from GNSS
        start_lat = obs["gnss"]["latitude"]
        start_lon = obs["gnss"]["longitude"]
        start_loc = gnss_to_location(start_lat, start_lon)
        
        # Choose a random goal and generate route
        goal_lat, goal_lon = choose_random_coordinate()
        goal_loc = gnss_to_location(goal_lat, goal_lon)
        grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0)
        route = grp.trace_route(start_loc, goal_loc)
        self.route = [wp.transform.location for wp, _ in route]
        self.wp_index = 0
        
        # Compute initial state
        state = self.get_state(obs)
        
        # Initialize tracking variables
        self.previous_location = gnss_to_location(start_lat, start_lon)
        self.previous_timestamp = obs["timestamp"]
        self.previous_distance = self.get_distance_to_waypoint(obs)
        self.step_count = 0
        
        return state

    def step(self, action):
        """Take an action, return next state, reward, done, and info."""
        # Send action to the server
        self.socket.send(pickle.dumps({"action": action.tolist()}))
        obs = pickle.loads(self.socket.recv())
        
        # Compute new state
        state = self.get_state(obs)
        
        # Compute reward
        reward = self.compute_reward(obs)
        
        # Check if episode is done
        done = self.is_done(obs)
        
        # Increment step count and check timeout
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        
        # Update previous values
        self.previous_location = gnss_to_location(obs["gnss"]["latitude"], obs["gnss"]["longitude"])
        self.previous_timestamp = obs["timestamp"]
        self.previous_distance = self.get_distance_to_waypoint(obs)
        
        return state, reward, done, {}

    def get_state(self, obs):
        """Compute the current state from observations."""
        current_loc = gnss_to_location(obs["gnss"]["latitude"], obs["gnss"]["longitude"])
        
        if self.wp_index < len(self.route):
            next_wp_loc = self.route[self.wp_index]
            distance = current_loc.distance(next_wp_loc)
            vehicle_yaw = quaternion_to_yaw(obs["imu"]["orientation"])
            delta_x = next_wp_loc.x - current_loc.x
            delta_y = next_wp_loc.y - current_loc.y
            desired_yaw = np.arctan2(delta_y, delta_x)
            angle_to_waypoint = normalize_angle(desired_yaw - vehicle_yaw)
        else:
            distance = 0
            angle_to_waypoint = 0
        
        # Compute speed from position change
        if self.previous_location is not None and self.previous_timestamp < obs["timestamp"]:
            time_elapsed = obs["timestamp"] - self.previous_timestamp
            distance_traveled = current_loc.distance(self.previous_location)
            speed = distance_traveled / time_elapsed if time_elapsed > 0 else 0
        else:
            speed = 0
        
        return np.array([distance, angle_to_waypoint, speed], dtype=np.float32)

    def get_distance_to_waypoint(self, obs):
        """Calculate distance to the current waypoint."""
        current_loc = gnss_to_location(obs["gnss"]["latitude"], obs["gnss"]["longitude"])
        if self.wp_index < len(self.route):
            next_wp_loc = self.route[self.wp_index]
            return current_loc.distance(next_wp_loc)
        return 0

    def compute_reward(self, obs):
        """Calculate the reward based on progress and penalties."""
        current_distance = self.get_distance_to_waypoint(obs)
        progress = self.previous_distance - current_distance
        reward = progress
        
        # Penalties for collision and lane invasion
        if obs["collision"] is not None:
            reward -= 100
        if obs["lane_invaded"]["violated"]:
            reward -= 10
        
        # Speed-based reward
        speed = self.get_state(obs)[2]
        if speed < 5 or speed > 10:
            reward -= 0.1
        else:
            reward += 0.1
        
        return reward

    def is_done(self, obs):
        """Determine if the episode should terminate."""
        if self.wp_index >= len(self.route):
            return True
        
        current_distance = self.get_distance_to_waypoint(obs)
        if current_distance < 2.0:
            self.wp_index += 1
            if self.wp_index >= len(self.route):
                return True
        
        if obs["collision"] is not None or obs["lane_invaded"]["violated"]:
            return True
        
        return False

# Main execution
if __name__ == "__main__":
    # Initialize the environment
    env = CarlaEnv(port=12345)
    
    # Create and train the PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # Save the trained model
    model.save("carla_rl_agent")