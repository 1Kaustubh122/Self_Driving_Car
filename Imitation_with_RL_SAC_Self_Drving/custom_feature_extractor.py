import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Imitation_Learning_RL.model.ImitationResNet import ImitationResNet


class UnifiedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,  features_dim: int = 512):
        super().__init__(observation_space, features_dim=features_dim)
        self.model = ImitationResNet(pretrained=False)
        MODEL_PATH = "models/bc_model.pth"
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        
        self.model.fc = nn.Identity()
        self.model.steer_head = nn.Identity()
        self.model.throttle_head = nn.Identity()
        self.model.brake_head = nn.Identity()
        
        for name, param in self.model.named_parameters():
            if 'lstm' in name:
                param.requires_grad = False

        self.state_dim = observation_space["state"].shape[0]

        self.image_features_dim = 768
        self.combine_fc = nn.Linear(self.image_features_dim + self.state_dim, features_dim)

        self.model.to(self.device)
        self.combine_fc.to(self.device)

    def forward(self, observations: dict) -> torch.Tensor:


        image_obs = observations["image"]

        if not torch.is_tensor(image_obs):
            x = torch.tensor(image_obs, dtype=torch.float32, device=self.device)
        else:
            x = image_obs.float().to(self.device)
        
        if x.ndim == 3:
            x = x.squeeze(0)

        x = self.model(x)
        
        # print("Feature shape before flattening:", x.shape)
        x = torch.flatten(x, start_dim=1)  
        # print("Feature shape after flattening:", x.shape)

        state_obs = observations["state"]
        if not torch.is_tensor(state_obs):
            state_obs = torch.tensor(state_obs, dtype=torch.float32, device=self.device)
        else:
            state_obs = state_obs.float().to(self.device)
        
        if state_obs.ndim == 1:
            state_obs = state_obs.unsqueeze(0)

        combiner_features = torch.cat([x, state_obs], dim=1)

        output = self.combine_fc(combiner_features)

        return output





## OLD FEATURE EXTRACTOR ---- Only Images
# class CustomFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space:gym.spaces.Box, features_dim: int = 768):
#         super().__init__(observation_space, features_dim=features_dim)
#         self.model = ImitationResNet(pretrained=False)
#         MODEL_PATH = "models/bc_model.pth"
#         checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#         self.model.load_state_dict(checkpoint['model_state_dict'])
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.device = 'cpu'
        
#         self.model.fc = nn.Identity()
#         self.model.steer_head = nn.Identity()
#         self.model.throttle_head = nn.Identity()
#         self.model.brake_head = nn.Identity()
        
#         for name, param in self.model.named_parameters():
#             if 'lstm' in name:
#                 param.requires_grad = False
            
#         # self.model.to(self.device)
            
#     # def forward(self, observations):
#     #     x = torch.as_tensor(observations, dtype=torch.float32)
#     #     x = self.model.conv1(x)
#     #     x = self.model.bn1(x)
#     #     x = self.model.relu(x)
#     #     x = self.model.maxpool(x)
#     #     x = self.model.layer1(x)
#     #     x = self.model.layer2(x)
#     #     x = self.model.layer3(x)
#     #     x = self.model.layer4(x)
#     #     x = self.model.avgpool(x)
        
#     #     x = torch.flatten(x, 1)
#     #     x = x.unsqueeze(1)
#     #     x, _ = self.model.lstm(x)
#     #     x = x.squeeze(1)
#     #     return x
    
    
    
    # def forward(self, observations: torch.Tensor) -> torch.Tensor:

    #     if not torch.is_tensor(observations):
    #         x = torch.tensor(observations, dtype=torch.float32, device=self.device)
    #     else:
    #         x = observations.float().to(self.device)
        
    #     if x.ndim == 3:
    #         x = x.squeeze(0)

    #     x = self.model(x)
        
    #     # print("Feature shape before flattening:", x.shape)
    #     x = torch.flatten(x, 1)  
    #     # print("Feature shape after flattening:", x.shape)

    #     return x 