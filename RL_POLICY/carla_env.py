import zmq
import time
import carla
import pickle
import numpy as np
import gymnasium as gym

# === GNSS → CARLA LOCATION ===
def gnss_to_location(lat, lon):
    x = -14418.6285 * lat + 111279.5690 * lon - 3.19252014
    y = -109660.6210 * lat +    4.33686914 * lon + 0.367254638
    return carla.Location(x=x, y=y, z=0.0)

# === GNSS DISTANCE IN SIM SPACE ===
def gnss_distance(coord1, coord2):
    loc1 = gnss_to_location(*coord1)
    loc2 = gnss_to_location(*coord2)
    return loc1.distance(loc2)

# === STUCK DETECTION ===
def is_stuck(info, prev_info, speed_threshold=0.1, time_threshold=2.0):
    if prev_info is None:
        return False
    t1 = info["timestamp"]
    t0 = prev_info["timestamp"]
    if t1 - t0 == 0:
        return False
    dx = info["gnss"][0] - prev_info["gnss"][0]
    dy = info["gnss"][1] - prev_info["gnss"][1]
    dist = np.sqrt(dx**2 + dy**2)
    speed = dist / (t1 - t0)
    return speed < speed_threshold and (t1 - prev_info["timestamp"]) > time_threshold

# === MAIN GYM ENVIRONMENT ===
class CarlaEnv(gym.Env):
    def __init__(self, goal_gnss, timeout_sec=60):
        super(CarlaEnv, self).__init__()
        
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:6501")

        # Action: [steer, throttle] ∈ [-1,1], [0,1]
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0]),
                                           high=np.array([1.0, 1.0]),
                                           dtype=np.float32)
        # Observation: 84x84 grayscale
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(1, 84, 84), dtype=np.uint8)

        self.prev_info = None
        self.target_gnss = goal_gnss
        self.timeout_sec = timeout_sec
        self.reset_time = time.time()

    def step(self, action):
        data = {"action": action.tolist()}
        self.socket.send(pickle.dumps(data))
        response = pickle.loads(self.socket.recv())

        obs = self._process_obs(response["image"])
        reward, done = self._calculate_reward(response)
        info = response
        self.prev_info = response

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.socket.send(pickle.dumps({"command": "reset"}))
        _ = self.socket.recv()

        self.socket.send(pickle.dumps({"action": [0.0, 0.0]}))
        response = pickle.loads(self.socket.recv())

        obs = self._process_obs(response["image"])
        self.prev_info = response
        self.reset_time = time.time()

        return obs, {}

    def _process_obs(self, image_bytes):
        img = np.frombuffer(image_bytes, dtype=np.uint8).reshape((84, 84))
        return np.expand_dims(img, axis=0)  # shape = (1, 84, 84)

    def _calculate_reward(self, info):
        reward = 0
        done = False

        # === COLLISION ===
        if info["collision"] is not None:
            return -100, True

        # === LANE INVASION ===
        if info["lane_invaded"]["violated"]:
            return -50, True

        # === GOAL REACHED ===
        distance = gnss_distance(info["gnss"], self.target_gnss)
        if distance < 2.0:
            return 100, True

        # === STUCK ===
        if is_stuck(info, self.prev_info):
            reward -= 10

        # === TIMEOUT ===
        if time.time() - self.reset_time > self.timeout_sec:
            return -50, True

        # === PROGRESS REWARD ===
        prev_dist = gnss_distance(self.prev_info["gnss"], self.target_gnss) if self.prev_info else distance + 1
        reward += (prev_dist - distance) * 2  # forward movement

        return reward, done

    def render(self):
        pass

    def close(self):
        self.socket.close()
