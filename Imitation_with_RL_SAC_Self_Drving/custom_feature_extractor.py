import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Imitation_Learning_RL.model.ImitationResNet import ImitationResNet


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.spaces.Box):
        super().__init__(observation_space, features_dim=768)
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
            
        # self.model.to(self.device)
            
    # def forward(self, observations):
    #     x = torch.as_tensor(observations, dtype=torch.float32)
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.relu(x)
    #     x = self.model.maxpool(x)
    #     x = self.model.layer1(x)
    #     x = self.model.layer2(x)
    #     x = self.model.layer3(x)
    #     x = self.model.layer4(x)
    #     x = self.model.avgpool(x)
        
    #     x = torch.flatten(x, 1)
    #     x = x.unsqueeze(1)
    #     x, _ = self.model.lstm(x)
    #     x = x.squeeze(1)
    #     return x
    
    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(observations):
            x = torch.tensor(observations, dtype=torch.float32, device=self.device)
        else:
            x = observations.float().to(self.device)
            
        x = self.model(x)
        
        # print("Feature shape before flattening:", x.shape)
        x = torch.flatten(x, 1)  
        # print("Feature shape after flattening:", x.shape)

        return x 