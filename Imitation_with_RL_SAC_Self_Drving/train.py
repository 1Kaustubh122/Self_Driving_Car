#!/usr/bin/env python3
import os
import re
import glob
import sys
import gc
import argparse
import torch

# # for local imports 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from carla_gym_env import CarlaEnv
from custom_feature_extractor import UnifiedFeatureExtractor


class TorchGCCallback(BaseCallback):
    def __init__(self, interval_steps: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._interval = max(1, int(interval_steps))

    def _on_step(self) -> bool:
        if self.n_calls % self._interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True


def find_latest_checkpoint(checkpoints_dir: str, prefix: str = "sac_carla_model"):
    pattern = os.path.join(checkpoints_dir, f"{prefix}_*steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None

    def _steps(p):
        m = re.search(rf"{re.escape(prefix)}_(\d+)_steps\.zip$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    files.sort(key=_steps)
    return files[-1]


def make_env(host: str, port: int):
    env = CarlaEnv(host=host, port=port)
    env = Monitor(env)
    return env


def parse_args():
    p = argparse.ArgumentParser(description="SAC training on single CARLA env")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--tb-log-dir", type=str, default="tensorboard_logs")
    p.add_argument("--checkpoint-prefix", type=str, default="sac_carla_model")
    p.add_argument("--save-freq", type=int, default=10_000)  # env steps
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint if present")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--gc-interval", type=int, default=100)
    # SAC core hyperparams 
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)

    # Single environment
    env = make_env(args.host, args.port)

    policy_kwargs = {
        "features_extractor_class": UnifiedFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 768},
        "net_arch": dict(pi=[128, 64], qf=[128, 64]),
    }

    device = args.device

    latest = find_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix) if args.resume else None
    if latest:
        print(f"[resume] Loading checkpoint: {latest}")
        model = SAC.load(latest, env=env, device=device)
    else:
        print("[new] Starting fresh training")
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            tensorboard_log=args.tb_log_dir,
            verbose=1,
            device=device,
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix=args.checkpoint_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    gc_cb = TorchGCCallback(interval_steps=args.gc_interval)
    callbacks = CallbackList([checkpoint_cb, gc_cb])

    # Compute remaining timesteps if resuming
    already = int(getattr(model, "num_timesteps", 0))
    remaining = max(0, int(args.total_steps) - already)
    print(f"[info] steps_done={already} remaining={remaining}")

    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    else:
        print("[info] nothing to do, already at or beyond target steps")

    # Save final
    final_stem = f"{args.checkpoint_prefix}_final_{args.total_steps // 1000}k"
    final_path = os.path.join(args.checkpoint_dir, final_stem)
    model.save(final_path)
    print(f"[done] Saved final model to: {final_path}")


if __name__ == "__main__":
    main()
