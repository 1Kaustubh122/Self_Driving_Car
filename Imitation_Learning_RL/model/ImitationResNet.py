import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ImitationResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ImitationResNet, self).__init__()

        base_model = resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 512)
        self.steer_head = nn.Linear(512, 1)
        self.throttle_head = nn.Linear(512, 1)
        self.brake_head = nn.Linear(512, 1)

        self._init_weights_from_rgb(base_model)

    def _init_weights_from_rgb(self, base_model):
        old_weights = base_model.conv1.weight.data
        new_weights = torch.zeros((64, 5, 7, 7))  # (out_channels, in_channels, H, W)
        new_weights[:, :3, :, :] = old_weights
        self.conv1.weight.data = new_weights

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)  
        x = x.squeeze(1)  
        
        x = torch.relu(self.fc(x))
        steer = torch.tanh(self.steer_head(x))  
        throttle = torch.sigmoid(self.throttle_head(x))  
        brake = self.brake_head(x)  
        
        return torch.cat([steer, throttle, brake], dim=1)