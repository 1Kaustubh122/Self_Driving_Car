import re
import os
import sys
import glob
import platform

if platform.system() == "Windows":
    sys.path.append('C:/Users/WORKSTATION2/Self_Driving_Car/')
    sys.path.append('C:/Users/WORKSTATION2/Self_Driving_Car/Imitation_Learning_RL/')
    sys.path.append('C:/Users/WORKSTATION2/Self_Driving_Car/Imitation_Learning_RL/model')
else: # Ubunutu
    sys.path.append('/home/user/Self_Driving_Car/')
    sys.path.append('/home/user/Self_Driving_Car/Imitation_Learning_RL/')
    sys.path.append('/home/user/Self_Driving_Car/Imitation_Learning_RL/model')

import gc
import torch
import multiprocessing as mp
from stable_baselines3 import SAC
from carla_gym_env import CarlaEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from custom_feature_extractor import UnifiedFeatureExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_LOGS = "tensorboard_logs"
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "sac_carla_model")
TOTAL_TIMESTEPS = 5_000_000
SAVE_INTERVAL = 10000
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

def make_env(host, port):
    def _init():
        env = CarlaEnv(host=host, port=port)
        env.reset()
        env = Monitor(env)
        return env
    return _init


pc_ip = "192.168.0.2"
# env_fns = [make_env(pc_ip, 2000 + i) for i in range(0, 19, 3)]
env_fns = [make_env(pc_ip, port) for port in [2000, 2003, 2006, 2009]]
# env_fns = [make_env(pc_ip, port) for port in [2000,2003, 2006, 2009]]


class TorchGCCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return True
    
def find_latest_checkpoint(checkpoints_dir, prefix="sac_carla_model"):
    """Find the checkpoint with the highest number of steps."""
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, f"{prefix}_*steps.zip"))
    if not checkpoint_files:
        return None

    def extract_steps(path):
        match = re.search(rf"{prefix}_(\d+)_steps\.zip", os.path.basename(path))
        return int(match.group(1)) if match else -1

    checkpoint_files = sorted(checkpoint_files, key=extract_steps)
    return checkpoint_files[-1] 

def main():

    
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOGS, exist_ok=True)
    
    # try:
    #     env=SubprocVecEnv(env_fns)
    # except Exception as e:
    #     print(f"Failed to create multi-envs: {e}")
    #     env=DummyVecEnv([make_env("localhost", 2000)])

    env=SubprocVecEnv(env_fns)
    # env=DummyVecEnv([make_env("localhost", 2000)])

    # env=DummyVecEnv(env_fns)

    # print(env.observation_space)

    
    policy_kwargs = {
        'features_extractor_class' : UnifiedFeatureExtractor,

        'features_extractor_kwargs' : {
            "features_dim" : 768
        },
  
        # 'net_arch' : dict(pi=[768, 256, 256], qf=[768, 256, 256])
        # 'net_arch' : dict(pi=[512, 256], qf=[512, 256])
        # 'net_arch' : dict(pi=[256, 128], qf=[256, 128])
        'net_arch' : dict(pi=[128, 64], qf=[128, 64])
        
    }
    
    latest_checkpoint = find_latest_checkpoint(CHECKPOINTS_DIR) 
    # latest_checkpoint = 'checkpoints\sac_carla_model_117200_steps.zip'

    
    if latest_checkpoint:
        model = SAC.load(latest_checkpoint, env=env, device=DEVICE)
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
    else:
        print("Starting new training.")
        
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=1000,
            learning_starts=5000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=TENSORBOARD_LOGS,
            verbose=1,
            device=DEVICE
        )
    
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=CHECKPOINTS_DIR, name_prefix='sac_carla_model')
    callbacks = CallbackList([checkpoint_callback, TorchGCCallback()])
    timesteps_done = model.num_timesteps
    remaining_timesteps = TOTAL_TIMESTEPS - timesteps_done

    print(f"Already trained: {timesteps_done} steps. Remaining: {remaining_timesteps} steps.")

    model.learn(total_timesteps=remaining_timesteps, callback=callbacks, progress_bar=True, reset_num_timesteps=False)

    gc.collect()  
    torch.cuda.empty_cache()
    model.save("sac_carla_model")
    
    final_model_path = os.path.join(CHECKPOINTS_DIR, f"sac_final_{TOTAL_TIMESTEPS//1000}k")
    model.save(final_model_path)
    print(f"[DONE] Final model saved to: {final_model_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()