import os
import cv2
import sys
import time
import carla
import random
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from Imitation_Learning_RL.process_frame import process_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pre_process(rgb_rgb: np.ndarray, seg_img: np.ndarray) -> np.ndarray:
    H, W = 128, 128
    rgb = cv2.resize(rgb_rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    seg = cv2.resize(seg_img, (W, H), interpolation=cv2.INTER_NEAREST)

    # If seg is color-coded, derive 2 channels via simple thresholds. Adjust if you have class IDs.
    lane = ((np.abs(seg[:, :, 1] - 255) < 1) & (seg[:, :, 0] < 2) & (seg[:, :, 2] < 2)).astype(np.float32)
    obs  = ((np.abs(seg[:, :, 2] - 255) < 1) & (seg[:, :, 0] < 2) & (seg[:, :, 1] < 2)).astype(np.float32)

    imgCHW = np.transpose(rgb, (2, 0, 1))                 # (3,H,W)
    segCHW = np.stack([lane, obs], axis=0)                # (2,H,W)
    out = np.concatenate([imgCHW, segCHW], axis=0)        # (5,H,W)
    return out.astype(np.float32)

def get_actor_blueprints(world, filter, generation = 'all'):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

class CarlaEnv(gym.Env):
    def __init__(self, host, port):
        super(CarlaEnv, self).__init__()

        import queue
        self._q = queue.Queue(maxsize=1)
        self._actors = []

        self.client_ = carla.Client(host, port)
        self.client_.set_timeout(10.0)
        self.world_ = self.client_.get_world()
        self._orig_settings = self.world_.get_settings()
        
        traffic_manager = self.client_.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)

        self.blueprint_library_ = self.world_.get_blueprint_library()
        self.map_ = self.world_.get_map()
        self.spawn_points_ = self.map_.get_spawn_points()
        self.start_point_ = self.spawn_points_[0] if self.spawn_points_ else carla.Transform()

        self.vehicle_bp = self.blueprint_library_.find('vehicle.volkswagen.t2_2021')

        # Camera bp prepared once; spawned in reset
        self.camera_bp_ = self.blueprint_library_.find('sensor.camera.rgb')
        self.camera_bp_.set_attribute('image_size_x', '128')
        self.camera_bp_.set_attribute('image_size_y', '128')
        self.camera_transform = carla.Transform(carla.Location(x=4.442184/2.9, y=0, z=2.2))

        self.collision_bp = self.blueprint_library_.find("sensor.other.collision")
        self.lane_inv_bp  = self.blueprint_library_.find("sensor.other.lane_invasion")

        # Action/obs spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = Dict({
            "image": Box(low=0.0, high=1.0, shape=(5, 128, 128), dtype=np.float32),
            "state": Box(
                low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32),
                high=np.array([60.0,  1.0, 1.0, 1.0, 100.0,  np.pi], dtype=np.float32),
                shape=(6,), dtype=np.float32
            )
        })

        # Runtime flags
        self.max_steps = 6000
        self.episode_step = 0
        self.collision = False
        self.lane_invasion = False
        self.lane_inv_count = 0
        self.is_stuck = False
        self.stuck_start_time = None
        self.prev_steer = 0.0
        self.previous_location = None
        self.waypoints = []
        self.current_waypoint_idx = 0

        # Placeholders for actors
        self.vehicle_ = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        self.brake_on_since = None
        self.no_move_since = None

        blueprints = get_actor_blueprints(self.world_, "vehicle.*", generation='all')
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        spawn_points = self.world_.get_map().get_spawn_points()
 
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        
        number_of_vehicles = 20
        
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            # print(blueprint) 

            blueprint.set_attribute('role_name', 'autopilot')

        #     # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            
        self.vehicles_list = []
            
        for response in self.client_.apply_batch_sync(batch, False):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

        
    # # Sensors Callbacks
    # def image_callback(self, data):
    #     array = np.frombuffer(data.raw_data, dtype=np.uint8)
    #     array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
    #     with self.image_lock:
    #         self.rgb_image = array
    #         self.rgb_count += 1
    #         self.seg_image = process_frame(self.rgb_image)
    #         self.seg_count += 1
            
    def _apply_sync_settings(self):
        s = self.world_.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 0.05
        s.no_rendering_mode = False  # set True if running headless for speed
        self.world_.apply_settings(s)

    def collision_callback(self, event):
        self.collision = True

    def lane_invasion_callback(self, event):
        self.lane_invasion = True
        self.lane_inv_count += 1

    # # call to draw waypoints on map
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
        for a in reversed(self._actors):
            try:
                if isinstance(a, carla.Sensor):
                    a.stop()
                a.destroy()
            except:
                pass
        self._actors.clear()
        self.vehicle_ = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
    def _tick_and_get_frame(self, timeout=1.0):
        
        self.world_.tick()
        img = self._q.get(timeout=timeout)  # carla.Image
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        bgr = arr[:, :, :3]                 # BGRA -> BGR
        rgb = bgr[:, :, ::-1].copy()        # to RGB
        return rgb, bgr
    
    @staticmethod
    def _signed_angle_2d(a: carla.Vector3D, b: carla.Vector3D) -> float:
        dot = a.x*b.x + a.y*b.y
        det = a.x*b.y - a.y*b.x
        
        return float(np.arctan2(det, np.clip(dot, -1.0, 1.0)))

    @staticmethod
    def _shield(obs_state, action, ttc=None, dist_lead=None):

        steer, thr, brk = map(float, action)
        if ttc is not None and ttc < 1.2:
            return np.array([np.clip(steer, -0.2, 0.2), 0.0, 1.0], dtype=np.float32), True
        
        if dist_lead is not None and dist_lead < 7.0:
            return np.array([np.clip(steer, -0.2, 0.2), 0.0, 1.0], dtype=np.float32), True
        
        return np.array([steer, thr, brk], dtype=np.float32), False
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.episode_step = 0
        self.collision = False
        self.lane_invasion = False
        self.lane_inv_count = 0
        self.is_stuck = False
        self.stuck_start_time = None
        self.prev_steer = 0.0

        # del acts
        self.destroy_actors()
        self._apply_sync_settings()

        # Spawn ego
        spawn_points = self.map_.get_spawn_points()
        sp = spawn_points[0] if spawn_points else carla.Transform()
        self.vehicle_ = self.world_.try_spawn_actor(self.vehicle_bp, sp)
        if self.vehicle_ is None:
            # fallback: random
            for spx in random.sample(spawn_points, k=len(spawn_points)):
                self.vehicle_ = self.world_.try_spawn_actor(self.vehicle_bp, spx)
                if self.vehicle_ is not None:
                    break
        if self.vehicle_ is None:
            raise RuntimeError("Failed to spawn vehicle")
        self._actors.append(self.vehicle_)

        # Camera
        self.camera_sensor = self._spawn(self.camera_bp_, self.camera_transform, attach_to=self.vehicle_)
        self.camera_sensor.listen(lambda data: (self._q.queue.clear(), self._q.put(data)) if not self._q.empty() else self._q.put(data))

        # Collision + lane invasion
        self.collision_sensor = self._spawn(self.collision_bp, carla.Transform(), attach_to=self.vehicle_)
        self.collision_sensor.listen(self.collision_callback)
        self.lane_invasion_sensor = self._spawn(self.lane_inv_bp, carla.Transform(), attach_to=self.vehicle_)
        self.lane_invasion_sensor.listen(self.lane_invasion_callback)

        # Route 
        map_ = self.world_.get_map()
        base_wp = map_.get_waypoint(sp.location if 'sp' in locals() else self.vehicle_.get_location())
        self.waypoints = [base_wp]
        for _ in range(50):
            nxt = self.waypoints[-1].next(5.0)
            self.waypoints.append(nxt[0] if nxt else self.waypoints[-1])
        self.current_waypoint_idx = 0

        self.previous_location = self.vehicle_.get_location()

        obs = self.get_observation()
        return obs, {}

    
    def _spawn(self, bp, tf, attach_to=None):
        actor = self.world_.spawn_actor(bp, tf, attach_to=attach_to)
        self._actors.append(actor)
        return actor
    
    def get_observation(self, timeout=2.0):
        t0 = time.time()
        while True:
            try:
                rgb_rgb, bgr_raw = self._tick_and_get_frame(timeout=timeout)
                break
            except Exception:
                if time.time() - t0 > timeout:
                    raise TimeoutError("Sensor timeout")
                continue

        seg = process_frame(bgr_raw)

        img = pre_process(rgb_rgb, seg)  # (5,128,128) float32 in [0,1]

        # State features
        vel = self.vehicle_.get_velocity()
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))  # m/s

        ctrl = self.vehicle_.get_control()
        brake = 1.0 if ctrl.brake >= 0.5 else 0.0
        throttle_eff = 0.0 if brake == 1.0 else max(0.4, float(ctrl.throttle))
        steering = float(ctrl.steer)

        loc = self.vehicle_.get_location()
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            dist = float(loc.distance(wp.transform.location))
            ang = self._signed_angle_2d(wp.transform.get_forward_vector(),
                                        self.vehicle_.get_transform().get_forward_vector())
        else:
            dist, ang = 0.0, 0.0

    
        state = np.array([speed, steering, throttle_eff, brake, dist, ang], dtype=np.float32)
        state = np.clip(
            state,
            a_min=np.array([0.0, -1.0, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32),
            a_max=np.array([60.0,  1.0, 1.0, 1.0, 100.0,  np.pi], dtype=np.float32),
        )
        return {"image": img, "state": state}

    def step(self, action):
        self.episode_step += 1

        a = np.asarray(action, dtype=np.float32)
        steer_gate = float(np.clip(a[0], -1.0, 1.0))
        thr_gate   = float(np.clip(a[1],  0.0, 1.0))
        brk_gate   = float(np.clip(a[2],  0.0, 1.0))

        # binary brake
        brake = 1.0 if brk_gate > 0.7 else 0.0
        # throttle rule
        throttle = 0.0 if brake == 1.0 else (0.4 + (0.6 * thr_gate))

        cmd = np.array([steer_gate, throttle, brake], dtype=np.float32)
        cmd, _ = self._shield(None, cmd)

        # re-enforce invariants after shield
        brake    = 1.0 if cmd[2] >= 0.7 else 0.0
        throttle = 0.0 if brake == 1.0 else max(0.4, float(cmd[1]))
        steer    = float(np.clip(cmd[0], -1.0, 1.0))

        vc = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False)
        self.vehicle_.apply_control(vc)

        # Observe
        obs = self.get_observation()

        # Stuck detection for termination only
        now = self.world_.get_snapshot().timestamp.elapsed_seconds
        vel = self.vehicle_.get_velocity()
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
        if self.stuck_start_time is None:
            self.stuck_start_time = now
        stuck = False
        if speed < 0.2:
            if (now - self.stuck_start_time) > 2.0:
                stuck = True
        else:
            self.stuck_start_time = now

        reward = self.compute_reward()

        terminated = bool(self.collision or self.current_waypoint_idx >= len(self.waypoints) - 1 or stuck)
        truncated  = bool(self.episode_step >= self.max_steps)

        return obs, reward, terminated, truncated, {}


    def compute_reward(self):
        r = 0.0
        loc = self.vehicle_.get_location()
        ctrl = self.vehicle_.get_control()

        vel = self.vehicle_.get_velocity()
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))

        # #### Waypoint progress
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.current_waypoint_idx]
            if loc.distance(wp.transform.location) < 2.0:
                self.current_waypoint_idx += 1
                r += 10.0
            else:
                break

        # Progress toward current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_idx]
            prev_dist = self.previous_location.distance(wp.transform.location)
            curr_dist = loc.distance(wp.transform.location)
            delta = prev_dist - curr_dist
            if delta > 0:
                r += 0.1 * float(delta)
        self.previous_location = loc

        now = self.world_.get_snapshot().timestamp.elapsed_seconds

        # brake dwell penalty when stationary
        if ctrl.brake >= 0.5:
            if self.brake_on_since is None:
                self.brake_on_since = now
            if speed < 0.2:
                r -= 0.5 * (now - self.brake_on_since)
        else:
            self.brake_on_since = None

        # throttle-without-motion penalty
        if ctrl.throttle >= 0.4 and speed < 0.2:
            if self.no_move_since is None:
                self.no_move_since = now
            r -= 1.0 * (now - self.no_move_since)
        else:
            self.no_move_since = None

        # Speed shaping
        if speed < 1.0:
            r -= 1.0
        else:
            r += 0.1 * speed

        # Steering smoothness
        r -= 0.5 * abs(ctrl.steer - self.prev_steer)
        self.prev_steer = float(ctrl.steer)

        # Lane invasion penalty (event-based, then clear)
        if self.lane_invasion:
            r -= 30.0
            self.lane_invasion = False

        # Collision penalty
        if self.collision:
            r -= 100.0

        # Small step penalty
        r -= 0.1

        return float(np.clip(r, -100.0, 50.0))