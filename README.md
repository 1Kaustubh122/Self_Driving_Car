# 🏎️ Self-Driving Car in CARLA Simulator

This project implements a self-driving car pipeline in the CARLA simulator using Imitation Learning (IL) and Reinforcement Learning (RL). The system evolves from expert data collection to Behavior Cloning (BC) with a custom ResNet18 architecture, followed by fine-tuning using Soft Actor-Critic (SAC). A ZeroMQ-based client-server setup connects the agent to the CARLA environment.

## 🚀 Project Overview

### ✔️ Stages:
1. **PPO Baseline (RL)**  
   - Initial experiments using Proximal Policy Optimization (PPO) to validate environment setup and agent control loop.

2. **Behavior Cloning (Imitation Learning)**  
   - Collected expert data with RGB + segmentation images and control actions (steer, throttle, brake).
   - Trained a custom **ResNet18** model with **5-channel input** (RGB + Seg) as a multi-output regressor.

3. **SAC Fine-Tuning (RL with IL Warm Start)**  
   - Used the pretrained BC model to warm-start a Soft Actor-Critic (SAC) agent.
   - RL fine-tuning improved robustness and long-term performance.


- ✅ Multi-stage learning: BC → SAC
- ✅ Custom ResNet18 with 5-channel input (RGB + Segmentation)
- ✅ Modular architecture using ZeroMQ for different environment communication
- ✅ Multi-output regression for steering, throttle, and brake


## 🖼️ Data Collection

Expert data includes:
- `rgb_image/`: Front camera RGB images
- `seg_image/`: Semantic segmentation images
- `logs/logs.json`: GNSS, IMU, speed, control actions (steer, throttle, brake)

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/1Kaustubh122/Self_Driving_Car
cd Self_Driving_Car

