🏎️ Self-Driving Car in CARLA Simulator
Built a self-driving car pipeline in the CARLA simulator. Started with Proximal Policy Optimization (PPO) to validate the RL setup. Then shifted to Imitation Learning (IL):

📦 Collected expert driving data (RGB + segmentation images + control actions)

🤖 Trained a multi-output Behavior Cloning (BC) model

🔁 Integrated DAgger to iteratively improve policy with expert corrections

🔧 Used the IL-trained model as a warm-start for Soft Actor-Critic (SAC) to fine-tune with reinforcement learning, improving robustness and long-term planning.

Supports sensor fusion, modular architecture, and ZeroMQ-based client-server design between CARLA and agent.

